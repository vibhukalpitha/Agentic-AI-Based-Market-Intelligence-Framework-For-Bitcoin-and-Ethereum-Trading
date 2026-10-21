"""Per-bar explanation detail (XGBoost TreeSHAP + interventional Shapley, AE errors, Transformer gradient x input) and the history-query descriptor, moved from the Streamlit page into the engine so the API can serve them in replay AND live mode.
Nothing here changes a model, feature, threshold or score: it only reads the engine's last step."""
import os
from pathlib import Path
import numpy as np

_BG = {}


def _r(x, n=4): return None if x is None or (isinstance(x, float) and x != x) else round(float(x), n)


def xgb_background(sym, folder=None):
    """K x 53 background rows from notebook 59 (explain_background_v1.npz in the assets folder); None when absent"""
    key = (sym, folder or os.environ.get('C03_ASSETS'))
    if key in _BG: return _BG[key]
    v = None; d = key[1]
    try:
        if d and (Path(d) / 'explain_background_v1.npz').exists():
            z = np.load(Path(d) / 'explain_background_v1.npz')
            if f'{sym}_xgb_rows' in z.files: v = np.asarray(z[f'{sym}_xgb_rows'], 'float64')
    except Exception: v = None
    _BG[key] = v
    return v


def transformer_detail(eng):
    import torch
    b = eng.bun; m = eng.modes[eng.MODELS[2]]; win = np.asarray(list(eng.zbuf[m])[-b.tr_W:], 'float32')
    if len(win) < b.tr_W: return {}
    x = torch.tensor(win[None], device=b.device); x.requires_grad_(True); y = 0.0
    for n in b.tr_nets: y = y + n(x).sum() / len(b.tr_nets)
    y.backward(); g = x.grad[0].detach().cpu().numpy().astype('float64'); ai = np.abs(g * win.astype('float64')); tot = ai.sum() or 1.0
    return {'tr_feat': [_r(float(v), 3) for v in ai.sum(axis=0) / tot], 'tr_time': [_r(float(v), 3) for v in ai.sum(axis=1) / tot]}


def xai_detail(eng, folder=None):
    """detail of the engine's LAST scored bar. Keys (all optional): shap (TreeSHAP shares per feature, indicative), shap_int / shap_int_group (interventional Shapley; group level validated, nb59),
    z (signed z of the six features), ae_err (AE error per feature), tr_feat / tr_time (Transformer gradient x input shares)."""
    import xgboost as xgb
    from .explain import _groups
    c, last, out = eng.c, eng._last, {}
    if last['Xa'] is not None:
        dm = xgb.DMatrix(last['Xa'], feature_names=c.FA_NAMES)
        con = np.mean([b.predict(dm, pred_contribs=True, iteration_range=(0, bi + 1))[0, :-1] for b, bi in eng.bun.xgb], axis=0)
        G = _groups(c); grp = np.array([con[G[f]].sum() for f in c.FEATS]); pos = grp[grp > 0].sum() or 1.0
        out['shap'] = [_r(float(g / pos), 3) for g in grp]
        try:
            bgx = xgb_background(eng.sym, folder)
            if bgx is not None:
                from . import explain_int as EI
                mf = lambda X: np.mean([b_.predict(xgb.DMatrix(X, feature_names=c.FA_NAMES), output_margin=True, iteration_range=(0, bi + 1)) for b_, bi in eng.bun.xgb], axis=0)
                ri = EI.interventional(mf, last['Xa'][0], bgx, G, list(c.FEATS))
                out['shap_int'] = [_r(float(v), 3) for v in EI.shares(ri['phi'])]
                gs = EI.group_shares(ri['phi'], list(c.FEATS)); tot = sum(v for v in gs.values() if v > 0) or 1.0
                out['shap_int_group'] = {k: _r(float(v / tot), 3) for k, v in gs.items()}
        except Exception: pass
        out['z'] = [_r(float(v), 2) for v in last['z6']]
    if last['cols8'] is not None: out['ae_err'] = [_r(float(v), 3) for v in last['cols8'][2:]]
    try: out.update(transformer_detail(eng))
    except Exception: pass
    out['features'] = list(c.FEATS)
    return out


_HIDX = {}


def default_history():
    """HistoryIndex from the assets folder (env C03_ASSETS); an empty index (ok = False) when the folder or files are missing"""
    from .history import HistoryIndex
    f = os.environ.get('C03_ASSETS')
    if f not in _HIDX:
        try: _HIDX[f] = HistoryIndex(f)
        except Exception: _HIDX[f] = HistoryIndex(None)
    return _HIDX[f]
