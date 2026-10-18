"""Interventional (background-marginalised) Shapley attribution for the XGBoost reason (protocol E4 v1 / notebook 59: H-E1y, H-E2y, H-E3y held on held-out TEST events).
For every set S of the six Layer-1 features, v(S) = mean over K background rows of the booster margin of the event's 53-feature row in which the derived columns of every feature NOT in S come from the background row
(cross-feature columns 'x' stay at the event's values). Exact Shapley values over the six features (64 coalitions). The background rows are data (explain_background_v1.npz, not committed).
Additive output; no frozen model, feature, threshold or score changes."""
from math import factorial
import numpy as np

NF = 6
MASKS = np.arange(2 ** NF)
KEEP = ((MASKS[:, None] >> np.arange(NF)) & 1).astype(bool)
POP = KEEP.sum(axis=1)
WSH = np.array([factorial(k) * factorial(NF - k - 1) / factorial(NF) for k in range(NF)])
FULL = 2 ** NF - 1
GROUPS3 = {'price': ('log_returns', 'hl_spread_ratio'), 'flow': ('trade_flow_imbalance', 'rush_order_density'), 'size_volume': ('large_trade_count', 'volume_acceleration')}


def coalition_rows(x53, bgx, gidx, feats):
    """rows (64*K, 53): coalition by background row; gidx maps feature name -> its derived columns"""
    K = len(bgx)
    R = np.repeat(np.asarray(x53, 'float64')[None, :], len(MASKS) * K, axis=0).reshape(len(MASKS), K, -1).copy()
    for j, f in enumerate(feats):
        cols = gidx[f]; drop = ~KEEP[:, j]
        R[np.ix_(drop, np.arange(K), cols)] = np.asarray(bgx, 'float64')[None, :, cols]
    return R.reshape(len(MASKS) * K, -1)


def shapley_from_v(v):
    phi = np.zeros(NF)
    for j in range(NF):
        for mk in MASKS[~KEEP[:, j]]:
            phi[j] += WSH[POP[mk]] * (v[mk | (1 << j)] - v[mk])
    return phi


def interventional(margin_fn, x53, bgx, gidx, feats):
    """margin_fn(rows) -> margins (mean over the declared boosters). Returns dict(phi=(6,), total=v(all)-v(empty))"""
    y = np.asarray(margin_fn(coalition_rows(x53, bgx, gidx, feats)), 'float64').reshape(len(MASKS), len(bgx)).mean(axis=1)
    return {'phi': shapley_from_v(y), 'total': float(y[FULL] - y[0])}


def shares(phi):
    """same display convention as the TreeSHAP shares: divide by the sum of the positive values"""
    phi = np.asarray(phi, 'float64'); pos = phi[phi > 0].sum() or 1.0
    return phi / pos


def group_shares(phi, feats):
    return {g: float(sum(phi[feats.index(f)] for f in fs)) for g, fs in GROUPS3.items()}
