"""Bar-by-bar scoring of one symbol: Layer 1 row in -> three model scores, fused score, thresholds, state, exact contributions out.
Reproduces the batch library (c03_infer / b2_lib, the code that opened TEST) bar for bar; tests/test_golden.py proves it."""
from collections import deque
import numpy as np, pandas as pd, torch, xgboost as xgb

STATES = ['NORMAL', 'WATCH', 'ALERT']


class SymbolEngine:
    def __init__(self, export, sym, device='cpu', watch_budget=None):
        self.ex, self.sym = export, sym; self.B, self.c = export.B, export.c; self.bun = export.bundle(sym, device); self.LR = export.LR; self.gst = export.gen_stats[sym]; cfg = export.config
        self.qa = cfg['alert_q'][sym]; self.qw = cfg['watch_q'][watch_budget or cfg['watch_default_budget']][sym]; self.MODELS = self.c.MODEL_NAMES; self.modes = self.bun.modes
        self.mq = {m: export.model_q[m][sym] for m in self.c.MODEL_NAMES}; self.rl = export.decl['unc']['rolling_levels']
        self.WIN, self.MINP = self.B.WIN7, self.B.MINP; self.n0 = self.bun.n0; self.t = 0; self.primed = False

    # ------------------------------------------------------------------ warm-up (batch, exact): fills every rolling buffer
    def prime(self, F, ts=None):
        B, c = self.B, self.c; F = F[B.FEATS].reset_index(drop=True).copy(); n = len(F); assert n >= 1500, 'warm-up needs at least 1,500 bars (7 days preferred)'
        sc = self.bun.score(F, want_cols=True); cols = sc.pop('_ae_cols'); X = B.to_x(F)
        self.xbuf = deque((r for r in X[-self.WIN:]), maxlen=self.WIN); self.zbuf = {}
        for m in sorted(set(self.modes.values())): z = B.normalise(F, m, self.n0); self.zbuf[m] = deque((r for r in z[-40:]), maxlen=40)
        f = c.nanw(cols[:, 0]); self.fmax = deque(f[-10:], maxlen=10)
        self.hist = {m: deque(c.TRF[m](sc[m])[-self.WIN:], maxlen=self.WIN) for m in self.MODELS}
        rs = c.ref_stats(sc, self.gst); Xf = c.fusion_features(sc, rs); fz = c.fusion_margin(self.LR, Xf)
        Zs = np.column_stack([np.clip((c.TRF[m](sc[m]) - rs[m][0]) / rs[m][1], -15, 15) for m in self.MODELS]); self.zs5 = deque((r for r in Zs[-5:]), maxlen=5)
        self.fhist = deque(fz[-self.WIN:], maxlen=self.WIN); self.t = n; self.primed = True; self.last_ts = None if ts is None else ts[-1]
        return fz

    # ------------------------------------------------------------------ one closed bar
    def step(self, feat, ts=None):
        assert self.primed, 'prime() first'; B, c = self.B, self.c; t = self.t
        x = B.to_x(np.asarray(feat, 'float64')[None, :])[0]; win = np.asarray(self.xbuf)
        if len(win) >= self.MINP:
            q25, med, q75 = np.percentile(win, [25, 50, 75], axis=0); scl = np.maximum((q75 - q25) / 1.349, 0.1 * self.n0['scale'])
        else: med, scl = self.n0['median'], self.n0['scale']
        z1 = np.clip((x - med) / scl, -B.CLIP, B.CLIP); z0 = np.clip((x - self.n0['median']) / self.n0['scale'], -B.CLIP, B.CLIP); self.xbuf.append(x)
        for m, buf in self.zbuf.items(): buf.append(z1 if m == 'N1' else z0)
        warm = t < B.WARM; sc = {}; cols8 = None
        # AE-ctx with the A6 aggregation
        z16 = np.asarray(list(self.zbuf[self.modes[self.MODELS[0]]])[-(B.CTX + 1):])
        err = np.mean([B.ae_errors(n, z16)[-1:] for n in self.bun.ae_nets], axis=0); cols8 = B.ae_columns(err, self.bun.ae_scale).astype('float32')[0]
        fm = np.nan if warm else float(cols8[0]); self.fmax.append(fm); arr = np.asarray(self.fmax, 'float64')
        sc[self.MODELS[0]] = np.nan if warm else (0.0 if (np.isnan(arr).any() or len(arr) < 10) else float(np.float32(arr.mean())))
        # XGBoost (feature set A)
        if warm: sc[self.MODELS[1]] = np.nan
        else:
            Xa = c.build_feats(np.asarray(list(self.zbuf[self.modes[self.MODELS[1]]])))[0][-1:]; dm = xgb.DMatrix(Xa, feature_names=c.FA_NAMES)
            sc[self.MODELS[1]] = float(np.mean([b.predict(dm, output_margin=True, iteration_range=(0, bi + 1)) for b, bi in self.bun.xgb], axis=0).astype(np.float32)[0])
        # Transformer
        if warm: sc[self.MODELS[2]] = np.nan
        else:
            zt = torch.tensor(np.asarray(list(self.zbuf[self.modes[self.MODELS[2]]])[-self.bun.tr_W:], 'float32')[None]); dev = self.bun.device
            with torch.no_grad(): sc[self.MODELS[2]] = float(np.mean([n(zt.to(dev)).float().cpu().numpy() for n in self.bun.tr_nets], axis=0).astype(np.float32)[0])
        # fusion v0
        zs = []
        for m in self.MODELS:
            w = np.asarray(self.hist[m]); w = w[~np.isnan(w)]
            if len(w) >= self.MINP: q25, med_, q75 = np.percentile(w, [25, 50, 75]); scl_ = max((q75 - q25) / 1.349, 0.1 * self.gst[m][1])
            else: med_, scl_ = self.gst[m]
            v = c.TRF[m](np.float64(sc[m])); zs.append(float(np.clip((v - med_) / scl_, -15, 15)))
            self.hist[m].append(float(v))
        self.zs5.append(np.asarray(zs)); Z = np.asarray(zs); fz = np.nan; fused_feats = None; contrib = None
        if not np.isnan(Z).any():
            roll = np.nanmax(np.asarray(self.zs5), axis=0); fused_feats = np.concatenate([Z, [Z.max(), Z.mean()], roll]).astype(np.float32); fz = float(self.LR.decision_function(fused_feats[None])[0])
            ct = self.LR.contributions(fused_feats[None])[0]; contrib = {'ae': float(ct[0] + ct[5]), 'xgboost': float(ct[1] + ct[6]), 'transformer': float(ct[2] + ct[7]), 'shared': float(ct[3] + ct[4])}
        fh = np.asarray(self.fhist); fh = fh[~np.isnan(fh)]
        ta, tw = (np.quantile(fh, self.qa), np.quantile(fh, self.qw)) if len(fh) >= self.MINP else (np.inf, np.inf)
        unus = None; agree = None; conf = None; plevel = None
        if not np.isnan(fz) and len(fh) >= self.MINP:
            unus = float(100.0 * np.mean(fh < fz)); plevel = float(np.mean(fh >= fz))
            above = []
            for k_, m in zip(['ae', 'xgboost', 'transformer'], self.MODELS):
                w = np.asarray(self.hist[m]); w = w[~np.isnan(w)]
                if len(w) >= self.MINP and np.quantile(w, self.mq[m]) < c.TRF[m](np.float64(sc[m])): above.append(k_)
            agree = above
        self._last = {'Xa': None if warm else Xa, 'cols8': cols8, 'z6': np.asarray(list(self.zbuf[self.modes[self.MODELS[1]]])[-1]), 'fused_feats': fused_feats}
        self.fhist.append(fz); self.t += 1
        state = 'NORMAL' if np.isnan(fz) else ('ALERT' if fz > ta else ('WATCH' if fz > tw else 'NORMAL'))
        if state == 'ALERT' and agree is not None:
            lv = self.rl[2] if len(self.rl) > 2 else 0.001; many = len(agree) >= 2; rare = plevel is not None and plevel <= lv
            conf = 'HIGH' if (many and rare) else ('MEDIUM' if (many or rare) else 'LOW')
        return {'ts': ts, 'bar_index': t, 'unusualness': unus, 'models_above': agree, 'agreement': None if agree is None else len(agree), 'confidence': conf, 'plevel': plevel, 'warmup': bool(warm or len(fh) < self.MINP), 'scores': {'ae': sc[self.MODELS[0]], 'xgboost': sc[self.MODELS[1]], 'transformer': sc[self.MODELS[2]]}, 'fusion': fz, 'thr_alert': float(ta), 'thr_watch': float(tw), 'state': state, 'contrib': contrib, 'z': {'ae': zs[0], 'xgboost': zs[1], 'transformer': zs[2]}}
