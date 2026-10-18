import numpy as np
from c03_engine import explain_int as E

FEATS = ['log_returns', 'trade_flow_imbalance', 'hl_spread_ratio', 'rush_order_density', 'volume_acceleration', 'large_trade_count']


def _gidx():
    rng = np.random.default_rng(0); p = rng.permutation(53)
    g = {f: sorted(p[k * 8:(k + 1) * 8].tolist()) for k, f in enumerate(FEATS)}; g['x'] = sorted(p[48:].tolist()); return g


def test_rows_structure():
    g = _gidx(); x = np.arange(53.); b = -np.arange(53.)[None, :] - 1; R = E.coalition_rows(x, b, g, FEATS)
    fc = sorted(sum([g[f] for f in FEATS], []))
    assert np.array_equal(R[E.FULL], x) and np.array_equal(R[0][fc], b[0][fc]) and np.array_equal(R[0][g['x']], x[g['x']])
    r = R[E.FULL & ~(1 << 2)]; assert sorted(np.where(r != x)[0].tolist()) == sorted(g[FEATS[2]])


def test_linear_game_and_efficiency():
    g = _gidx(); rng = np.random.default_rng(1); w = rng.normal(size=53); x = rng.normal(size=53); bg = rng.normal(size=(9, 53))
    res = E.interventional(lambda X: X @ w, x, bg, g, FEATS)
    for j, f in enumerate(FEATS):
        assert abs(res['phi'][j] - float(w[g[f]] @ (x[g[f]] - bg[:, g[f]].mean(axis=0)))) < 1e-9
    assert abs(res['phi'].sum() - res['total']) < 1e-9


def test_shares_and_groups():
    phi = np.array([2., -1, 1, 0, 1, 0]); s = E.shares(phi); assert abs(s[phi > 0].sum() - 1) < 1e-12
    gs = E.group_shares(phi, FEATS); assert abs(sum(gs.values()) - phi.sum()) < 1e-12
