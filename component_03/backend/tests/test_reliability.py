"""Reliability layer: pure functions by hand, and bar-by-bar equality with a pandas re-implementation of the notebook-49 rules."""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c03_engine import reliability as R


def test_p_values_by_hand():
    cal = np.array([1., 2., 3., 4.])
    assert R.p_abnormal(0.5, cal) == 1 / 5 and R.p_abnormal(2.0, cal) == 3 / 5 and R.p_abnormal(9, cal) == 1.0
    assert R.p_normal(9, cal) == 1 / 5 and R.p_normal(2.0, cal) == 4 / 5 and R.p_normal(0, cal) == 1.0


def test_level_rule():
    L = R.level_code
    assert L(False, True, 1, 1, 1) == 0 and L(True, True, 1, 1, 1) == 3 and L(True, True, 1, 1, 0) == 2 and L(True, True, 1, 0, 0) == 1 and L(True, False, 1, 1, 1) == 1 and L(True, True, 0, 0, 0) == 1


def test_tracker_equals_pandas():
    rng = np.random.default_rng(0); WIN, MINP, n = 600, 100, 1500
    ident = lambda x: x; trf = {'a': ident, 'x': ident, 't': ident}; mq = {'a': .98, 'x': .98, 't': .98}; qa = 0.995
    S = {k: rng.standard_normal(n) for k in 'axt'}
    for k in 'xt':
        S[k][1000:1010] += 6
    S['x'][1200:1210] += 6                       # xgb alone -> conflict
    fz = (S['a'] + S['x'] + S['t']) / 3 + rng.standard_normal(n) * .1; fz[1000:1010] += 4; fz[1200:1210] += 4
    prime = 800; tr = R.ReliabilityTracker(['a', 'x', 't'], trf, qa, mq, WIN, MINP, None, 'S')
    tr.prime(fz[:prime], {k: S[k][:prime] for k in 'axt'}, good_run=15)
    f = pd.Series(fz)
    thr = f.rolling(WIN, min_periods=MINP).quantile(qa).shift(1); rare_t = f.rolling(WIN, min_periods=MINP).quantile(.999).shift(1)
    alert = (f > thr)
    ab, lo = {}, {}
    for k in 'axt':
        s = pd.Series(S[k]); ab[k] = s > s.rolling(WIN, min_periods=MINP).quantile(mq[k]).shift(1); lo[k] = s <= s.rolling(WIN, min_periods=MINP).quantile(.9).shift(1)
    pers = alert.rolling(5, min_periods=1).sum() >= 3; nc = ~((ab['x'] & lo['t']) | (ab['t'] & lo['x'])); rare = f > rare_t
    bad = 0; seen = {1: 0, 2: 0, 3: 0}
    for t in range(prime, n):
        o = tr.update(fz[t], {k: S[k][t] for k in 'axt'}, 'ALERT' if alert[t] else 'NORMAL')
        exp = R.level_code(bool(alert[t]), True, bool(rare[t]), bool(pers[t]), bool(nc[t]))
        bad += int(o['level_code'] != exp); seen[exp] = seen.get(exp, 0) + (exp > 0)
        assert o['checks']['persistent'] == bool(pers[t])
    assert bad == 0 and sum(seen.values()) > 0, (bad, seen)


def test_gate_closes_after_rejected_minute():
    tr = R.ReliabilityTracker(['a', 'x', 't'], {k: (lambda x: x) for k in 'axt'}, .99, {k: .98 for k in 'axt'}, 300, 50, None, 'S')
    rng = np.random.default_rng(1); v = rng.standard_normal(200)
    tr.prime(v, {k: v for k in 'axt'}); tr.note_bad()
    for i in range(14):
        assert not tr.update(0., {k: 0. for k in 'axt'}, 'NORMAL')['gate']['ok']
    assert tr.update(0., {k: 0. for k in 'axt'}, 'NORMAL')['gate']['ok']


def test_conformal_sets_and_assets(tmp_path):
    np.savez(tmp_path / 'conformal_tables_v1.npz', S_abn=np.linspace(2, 8, 200), S_norm=np.linspace(-3, 1, 200))
    A = R.Assets(tmp_path); assert A.error is None and len(A.abn['S']) == 200
    tr = R.ReliabilityTracker(['a', 'x', 't'], {k: (lambda x: x) for k in 'axt'}, .99, {k: .98 for k in 'axt'}, 300, 50, A, 'S')
    v = np.random.default_rng(2).standard_normal(250); tr.prime(v, {k: v for k in 'axt'}, 20)
    z = {k: 0. for k in 'axt'}
    lo = tr.update(-1.0, z, 'NORMAL')['conformal']; hi = tr.update(5.0, z, 'NORMAL')['conformal']; far = tr.update(1.5, z, 'NORMAL')['conformal']
    assert lo['set'] == 'normal' and hi['set'] == 'abnormal' and far['set'] in ('cannot_tell', 'neither', 'normal', 'abnormal')
    assert R.Assets(tmp_path / 'nope').error is not None or R.Assets(tmp_path / 'nope').abn == {}
