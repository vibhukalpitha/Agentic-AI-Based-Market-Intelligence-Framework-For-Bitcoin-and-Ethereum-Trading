"""
live_engine.py - runs the TRAINED models inside the dashboard.

The dashboard no longer has to read pre-computed result files. For every minute that is replayed, the recorded Binance bar
(six Layer-1 features) is passed through the frozen engine (backend/c03_engine: Autoencoder + XGBoost + Transformer -> fusion v0
-> thresholds -> state, uncertainty, reasons, SHAP), exactly the code that the live API uses. Results are stored per minute in
the arrays that the page draws.

Needs the backend folder (c03_engine) and the engine export (models, config, replay bars). Nothing here trains or changes a model.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

AGREE_BITS = {'ae': 1, 'xgboost': 2, 'transformer': 4}
STATE_NUM = {'NORMAL': 0, 'WATCH': 1, 'ALERT': 2}
FIRST = 1500            # warm-up bars used to fill the engine's rolling windows (same as the dashboard bundle)


def find_backend(here):
    """returns (backend_dir, export_dir) or (None, reason)"""
    here = Path(here)
    cands = []
    if os.environ.get('C03_BACKEND'):
        cands.append(Path(os.environ['C03_BACKEND']))
    cands += [here.parent / 'backend', here / 'backend', here.parent / 'component_03' / 'backend']
    for b in cands:
        if (b / 'c03_engine' / 'core.py').exists():
            exp = Path(os.environ['C03_EXPORT']) if os.environ.get('C03_EXPORT') else b / 'export'
            if not (exp / 'manifest.json').exists():
                return None, f'engine export not found at {exp} (need manifest.json, models/, config/, replay/)'
            return (b, exp), None
    return None, 'backend folder with c03_engine not found next to this folder'


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), n)


def xai_detail(eng):
    """per-feature detail of the flagged bar from the frozen explain code (TreeSHAP shares, z-scores, AE errors, Transformer gradient x input)"""
    import xgboost as xgb
    from c03_engine.explain import _groups
    c, last, out = eng.c, eng._last, {}
    if last['Xa'] is not None:
        dm = xgb.DMatrix(last['Xa'], feature_names=c.FA_NAMES)
        con = np.mean([b.predict(dm, pred_contribs=True, iteration_range=(0, bi + 1))[0, :-1] for b, bi in eng.bun.xgb], axis=0)
        G = _groups(c)
        grp = np.array([con[G[f]].sum() for f in c.FEATS])
        pos = grp[grp > 0].sum() or 1.0
        out['shap'] = [_r(float(g / pos), 3) for g in grp]
        out['z'] = [_r(float(v), 2) for v in last['z6']]
    if last['cols8'] is not None:
        out['ae_err'] = [_r(float(v), 3) for v in last['cols8'][2:]]
    try:
        out.update(_tr_detail(eng))
    except Exception:
        pass
    return out


def _tr_detail(eng):
    import torch
    b = eng.bun
    m = eng.modes[eng.MODELS[2]]
    win = np.asarray(list(eng.zbuf[m])[-b.tr_W:], 'float32')
    if len(win) < b.tr_W:
        return {}
    x = torch.tensor(win[None], device=b.device)
    x.requires_grad_(True)
    y = 0.0
    for n in b.tr_nets:
        y = y + n(x).sum() / len(b.tr_nets)
    y.backward()
    g = x.grad[0].detach().cpu().numpy().astype('float64')
    ai = np.abs(g * win.astype('float64'))
    tot = ai.sum() or 1.0
    return {'tr_feat': [_r(float(v), 3) for v in ai.sum(axis=0) / tot], 'tr_time': [_r(float(v), 3) for v in ai.sum(axis=1) / tot]}


class LiveSymbol(SimpleNamespace):
    """Same attributes as the stored-bundle object used by the page, but filled minute by minute by the real models."""

    def advance(self, upto):
        """run the engine on every not-yet-processed bar up to and including `upto`"""
        upto = min(int(upto), self.n - 1)
        while self.done <= upto:
            self._one(self.done)
            self.done += 1
        self._windows()

    def _one(self, k):
        row = self.df.iloc[k]
        status, _ = self.svc.on_bar(row[self.feats].to_numpy('float64'), row.ts, {'close': float(row.close)})
        r = self.cap['r']
        self.state[k] = STATE_NUM[r['state']]
        self.fusion[k] = np.nan if r['fusion'] is None else r['fusion']
        self.thr_alert[k], self.thr_watch[k] = r['thr_alert'], r['thr_watch']
        self.ae[k], self.xgb[k], self.tr[k] = r['scores']['ae'], r['scores']['xgboost'], r['scores']['transformer']
        ct = r['contrib']
        if ct:
            self.c_ae[k], self.c_xgb[k], self.c_tr[k], self.c_shared[k] = ct['ae'], ct['xgboost'], ct['transformer'], ct['shared']
        self.plev[k] = np.nan if r['plevel'] is None else r['plevel']
        un = np.nan if r['unusualness'] is None else r['unusualness']
        self.integ[k] = 100.0 - un
        self.unus_engine[k] = un
        self.agree[k] = -1 if r['models_above'] is None else sum(AGREE_BITS[m] for m in r['models_above'])
        self.conf[k] = r['confidence']
        if status['reasons']:
            self.reasons[k] = [{'type': x['type'], 'text': x['text'], 'weight': _r(x['weight'], 3), 'feature': x.get('feature')} for x in status['reasons']]
            self.xai[k] = xai_detail(self.svc.eng)
        # flagged-minute episodes (gap <= 5), same rule as the stored-bundle loader
        if self.state[k] >= 1:
            if self._o is not None and k - self._o[1] <= 5:
                self._o[1] = k
            else:
                self._o = [k, k]
                self._eps.append(self._o)
            self.eps = [tuple(e) for e in self._eps]
            self.alert_eps = [e for e in self.eps if (self.state[e[0]:e[1] + 1] == 2).any()]
        self.t_engine += 0.0

    def _windows(self):
        for h in ('15m', '1h'):
            for w in self.svc.windows[h]:
                m = int(pd.Timestamp(w['window_start_utc']).value // 60_000_000_000)
                if m not in self.wmap[h]:
                    self.wmap[h][m] = {'start': w['window_start_utc'], 'end': w['window_end_utc'], 'peak_unus': _r(w['peak_unusualness'], 2), 'tier': w['tier'], 'flagged_min': w['flagged_minutes']}

    def parity(self):
        """independent check: engine output versus the reference scores the notebooks stored in the replay pack (exp_* columns)"""
        k0, k1 = self.first, self.done
        if k1 <= k0 or 'exp_fusion' not in self.df.columns:
            return None
        ref = self.df.exp_fusion.to_numpy('float64')[k0:k1]
        mine = self.fusion[k0:k1]
        m = np.isfinite(ref) & np.isfinite(mine)
        ref_state = self.df.exp_state.map(STATE_NUM).to_numpy()[k0:k1]
        return {'bars': int(k1 - k0), 'fusion_max_abs_diff': float(np.abs(ref[m] - mine[m]).max()) if m.any() else None,
                'fusion_median_abs_diff': float(np.median(np.abs(ref[m] - mine[m]))) if m.any() else None,
                'state_differs': int((ref_state != self.state[k0:k1]).sum())}


def make_live(ex, sym, cal=None):
    """build one LiveSymbol: loads the replay bars, primes the engine with the first FIRST bars (fills the rolling windows)"""
    from c03_engine.service import SymbolService
    df = pd.read_parquet(Path(ex.root) / 'replay' / f'{sym}_replay_2024-08-01_to_08-10.parquet')
    df = df.reset_index(drop=True)
    n = len(df)
    nan = lambda: np.full(n, np.nan)
    svc = SymbolService(ex, sym)
    cap = {}
    orig = svc.eng.step

    def step(feat, ts):
        r = orig(feat, ts)
        cap['r'] = r
        return r
    svc.eng.step = step
    feats = list(ex.B.FEATS)
    svc.prime(df.iloc[:FIRST][feats], list(df.ts[:FIRST]))
    cfg = ex.config
    qa = cfg['alert_q'][sym]
    qw = cfg['watch_q'][cfg['watch_default_budget']][sym]
    ts = [str(x) for x in df.ts]
    j = LiveSymbol(
        symbol=sym, sym=sym.replace('USDT', '/USDT'), n=n, first=FIRST, done=FIRST, ts=ts, df=df, feats=feats, svc=svc, cap=cap,
        dt=pd.to_datetime(pd.Series(ts), utc=True).dt.tz_localize(None).to_numpy(), t0=int(pd.Timestamp(ts[0]).value // 60_000_000_000),
        close=df.close.to_numpy('float64'), fusion=nan(), thr_alert=nan(), thr_watch=nan(), ae=nan(), xgb=nan(), tr=nan(), c_ae=nan(), c_xgb=nan(), c_tr=nan(), c_shared=nan(),
        plev=nan(), integ=nan(), unus_engine=nan(), state=np.zeros(n, int), agree=np.full(n, -1), conf=[None] * n, xai={}, reasons={}, features=feats,
        eps=[], alert_eps=[], _eps=[], _o=None, wmap={'15m': {}, '1h': {}}, cal=cal, _ev={}, _evid={}, t_engine=0.0,
        bandA=100.0 * (1 - qa), bandW=100.0 * (1 - qw), conf_labels=ex.decl['unc']['labels'], thr_windows=ex.decl['mh']['thresholds'][sym],
        alert_budget_per_day=cfg.get('alert_budget_per_day', 2), bundle_version='live engine (trained models run in this session)')
    return j


def advance_all(B, upto, progress=None, chunk=120):
    """advance every symbol to `upto` (both run in parallel threads for long jumps). progress(done_fraction, text) is optional."""
    todo = [j for j in B.values() if j.done <= upto]
    if not todo:
        return
    start = min(j.done for j in todo)
    total = max(1, upto - start + 1)
    if total <= 16:
        for j in todo:
            j.advance(upto)
        return
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(todo)) as pool:
        while any(j.done <= upto for j in todo):
            tgt = min(upto, min(j.done for j in todo) + chunk - 1)
            list(pool.map(lambda j: j.advance(tgt), todo))
            if progress:
                f = min(1.0, (min(j.done for j in todo) - start) / total)
                progress(f, f'Running the trained models: {int(f * total):,} of {total:,} minutes ({time.time() - t0:.0f} s)')


def advance_until(B, sym, cond, limit, progress=None, chunk=60):
    """run forward (both symbols) until cond(B[sym]) is true or `limit` is reached"""
    j = B[sym]
    while not cond(j) and j.done <= limit:
        advance_all(B, min(limit, j.done + chunk - 1), None)
        if progress:
            progress(min(1.0, (j.done - j.first) / max(1, limit - j.first)), f'Running the trained models: minute {j.done:,} of {limit:,}')
    return cond(j)
