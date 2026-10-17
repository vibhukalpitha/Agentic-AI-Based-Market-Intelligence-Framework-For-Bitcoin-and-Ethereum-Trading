"""Build the dashboard bundle: runs the REAL engine (same code as live/replay) over the saved replay bars and stores, per bar,
the uncertainty fields (rolling p-level, which models agree, confidence), the 15-min / 1-hour window tiers and the explanation
(TreeSHAP + model shares) of every flagged bar. Output: <out>/<SYMBOL>_dash.json  (= the replay json + extra fields).

    python tools/make_dashboard_bundle.py --export export --out export/dashboard
Takes about 4-5 minutes per symbol on a normal CPU (17 ms per bar)."""
import argparse, json, sys, time, glob
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c03_engine.store import Export
from c03_engine.service import SymbolService, HORIZONS

AGREE_BITS = {'ae': 1, 'xgboost': 2, 'transformer': 4}


def r_(x, n=4):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), n)


def xai_detail(eng):
    """Per-feature detail of the flagged bar from the SAME frozen explain code paths (nothing new is trained or tuned):
    shap = signed TreeSHAP share per feature (relative to the sum of the positive ones), z = feature z-score, ae_err = AE per-feature error (secondary, lower confidence)."""
    import xgboost as xgb
    from c03_engine.explain import _groups
    c, last, out = eng.c, eng._last, {}
    if last['Xa'] is not None:
        dm = xgb.DMatrix(last['Xa'], feature_names=c.FA_NAMES)
        con = np.mean([b.predict(dm, pred_contribs=True, iteration_range=(0, bi + 1))[0, :-1] for b, bi in eng.bun.xgb], axis=0)
        G = _groups(c); grp = np.array([con[G[f]].sum() for f in c.FEATS]); pos = grp[grp > 0].sum() or 1.0
        out['shap'] = [r_(float(g / pos), 3) for g in grp]; out['z'] = [r_(float(v), 2) for v in last['z6']]
    if last['cols8'] is not None: out['ae_err'] = [r_(float(v), 3) for v in last['cols8'][2:]]
    try: out.update(tr_detail(eng))
    except Exception as e:
        if not getattr(xai_detail, 'warned', False): print('  note: Transformer attribution skipped:', repr(e)); xai_detail.warned = True
    return out


def tr_detail(eng):
    """Transformer gradient x input (the method of notebook 51, applied to the engine's own input window; nothing is trained or changed).
    tr_feat = share of attribution per feature (sum over the window of |grad x input|), tr_time = share per bar of the window (oldest -> newest). Indicative only."""
    import torch
    b = eng.bun; m = eng.modes[eng.MODELS[2]]; win = np.asarray(list(eng.zbuf[m])[-b.tr_W:], 'float32')
    if len(win) < b.tr_W: return {}
    x = torch.tensor(win[None], device=b.device); x.requires_grad_(True); y = 0.0
    for n in b.tr_nets: y = y + n(x).sum() / len(b.tr_nets)
    y.backward(); g = x.grad[0].detach().cpu().numpy().astype('float64'); ai = np.abs(g * win.astype('float64')); tot = ai.sum() or 1.0
    return {'tr_feat': [r_(float(v), 3) for v in ai.sum(axis=0) / tot], 'tr_time': [r_(float(v), 3) for v in ai.sum(axis=1) / tot]}


def build(ex, root, sym, out, i0=1500, until=None):
    f = sorted(glob.glob(str(Path(root) / 'replay' / f'{sym}_replay_*.parquet')))[0]
    df = pd.read_parquet(f); n = len(df) if until is None else min(len(df), until)
    base = json.load(open(Path(root) / 'replay' / f'{sym}_replay.json'))
    svc = SymbolService(ex, sym); cap = {}
    orig = svc.eng.step

    def step(feat, ts):
        r = orig(feat, ts); cap['r'] = r; return r
    svc.eng.step = step
    svc.prime(df.iloc[:i0][ex.B.FEATS], list(df.ts[:i0]))
    N = len(df)
    plev = [None] * N; agree = [None] * N; conf = [None] * N; unus = [None] * N; bstate = [None] * N; ta_e = [None] * N; tw_e = [None] * N
    reasons = {}; caveats = {}; xai = {}
    t0 = time.time()
    for k in range(i0, n):
        row = df.iloc[k]
        status, _ = svc.on_bar(row[ex.B.FEATS].to_numpy('float64'), row.ts, {'close': float(row.close)})
        r = cap['r']
        plev[k] = r_(r['plevel'], 5); unus[k] = r_(r['unusualness'], 3); conf[k] = r['confidence']; bstate[k] = r['state']
        ta_e[k] = r_(r['thr_alert'], 4); tw_e[k] = r_(r['thr_watch'], 4)
        agree[k] = None if r['models_above'] is None else sum(AGREE_BITS[m] for m in r['models_above'])
        if status['reasons']:
            reasons[k] = [{'type': x['type'], 'text': x['text'], 'weight': r_(x['weight'], 3), 'feature': x.get('feature')} for x in status['reasons']]
            if status['caveats']: caveats[k] = status['caveats']
            xai[k] = xai_detail(svc.eng)
        if (k - i0) % 2000 == 0: print(f'  {sym}: {k - i0}/{n - i0} bars  ({time.time() - t0:.0f}s)', flush=True)
    wins = {h: [{'start': w['window_start_utc'], 'end': w['window_end_utc'], 'peak_unus': r_(w['peak_unusualness'], 2), 'tier': w['tier'], 'flagged_min': w['flagged_minutes']} for w in svc.windows[h]] for h in HORIZONS}
    base.update({'plev': plev, 'agree': agree, 'conf': conf, 'unus_engine': unus, 'bstate': bstate, 'thr_alert_engine': ta_e, 'thr_watch_engine': tw_e, 'reasons': reasons, 'xai': xai, 'feature_names': list(svc.eng.c.FEATS), 'caveats': caveats, 'windows': wins,
                 'thr_windows': ex.decl['mh']['thresholds'][sym], 'conf_labels': ex.decl['unc']['labels'], 'bundle_from_bar': i0, 'bundle_version': 1})
    mism = sum(1 for k in range(i0, n) if base['state'][k] is not None and bstate[k] is not None and ['NORMAL', 'WATCH', 'ALERT'].index(bstate[k]) != base['state'][k])
    print(f'  {sym}: state differences vs notebook replay: {mism} of {n - i0} bars')
    # diagnosis: are the differences explained by the engine's rolling 7-day window still filling up (only i0 bars of history at the start)?
    diff = [k for k in range(i0, n) if base['state'][k] is not None and bstate[k] is not None and ['NORMAL', 'WATCH', 'ALERT'].index(bstate[k]) != base['state'][k]]
    early = [k for k in diff if k - i0 < 10080 - i0]
    print(f'  {sym}: differing bars while engine window still filling (first {10080 - i0} bars after start): {len(early)} of {len(diff)}')
    for k in diff:
        f = base['fusion'][k]; ra, rw = base['thr_alert'][k], base['thr_watch'][k]
        print(f'    bar {k} {base["ts"][k]}  fusion={f:.4f}  notebook thr A/W={ra:.4f}/{rw:.4f}  engine thr A/W={ta_e[k]}/{tw_e[k]}  notebook={base["state"][k]} engine={bstate[k]}')
    dA = [abs(ta_e[k] - base['thr_alert'][k]) for k in range(i0, n) if ta_e[k] is not None and base['thr_alert'][k] is not None and ta_e[k] != float('inf')]
    if dA: print(f'  {sym}: |engine - notebook| alert line: median {np.median(dA):.4f}, max {np.max(dA):.4f}; after day 7: max {np.max(dA[10080 - i0:]) if len(dA) > 10080 - i0 else float("nan"):.4f}')
    p = Path(out) / f'{sym}_dash.json'; json.dump(base, open(p, 'w'), separators=(',', ':')); print('  wrote', p, f'({p.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--export', required=True); ap.add_argument('--out', default=None); ap.add_argument('--until', type=int, default=None, help='testing only: stop after this bar index')
    a = ap.parse_args(); out = a.out or str(Path(a.export) / 'dashboard'); Path(out).mkdir(parents=True, exist_ok=True)
    ex = Export(a.export, verify=True)
    for s in ex.symbols: build(ex, a.export, s, out, until=a.until)
