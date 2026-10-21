import numpy as np, pandas as pd
from tools import p_eval as pe


def _ts(n):
    return (np.datetime64("2024-07-01T00:00") + np.arange(n).astype("timedelta64[m]"))


def test_episodes_merge_gap():
    a = np.zeros(40, bool); a[[3, 4, 9, 20]] = True   # 4->9 gap 5 merges, 9->20 splits
    ep = pe.episodes(a, _ts(40), gap=5)
    assert ep.tolist() == [[3, 9], [20, 20]]


def test_episodes_use_timestamps_for_missing_minutes():
    ts = _ts(10).copy(); ts[5:] += np.timedelta64(30, "m")   # 30-min hole between idx 4 and 5
    a = np.zeros(10, bool); a[[4, 5]] = True
    assert len(pe.episodes(a, ts, gap=5)) == 2


def test_eps_per_day():
    n = 2880; a = np.zeros(n, bool); a[[10, 500, 1500]] = True
    assert abs(pe.genuine_eps_per_day(a, _ts(n)) - 1.5) < 1e-9


def test_t1_past_only():
    rng = np.random.default_rng(0); s = rng.normal(size=6000)
    t1 = pe.t1_threshold(s, 0.99, window=2000)
    s2 = s.copy(); s2[4000:] += 100                      # change the future
    t2 = pe.t1_threshold(s2, 0.99, window=2000)
    assert np.allclose(t1[:4001], t2[:4001], equal_nan=True)
    assert np.isnan(t1[:1440]).all() and not np.isnan(t1[1440])


def test_t1_matches_plain_loop():
    rng = np.random.default_rng(1); s = rng.normal(size=3000); w = 1500
    t = pe.t1_threshold(s, 0.9, window=w)
    for i in (1440, 2000, 2999):
        assert abs(t[i] - np.quantile(s[max(0, i - w):i], 0.9)) < 1e-9


def test_t1c_uses_genuine_threshold():
    rng = np.random.default_rng(2); g = rng.normal(size=4000); inj = g.copy(); inj[3000:3010] += 50
    al = pe.t1c_alarms(inj, g, 0.99, window=2000)
    assert al[3000:3010].all() and al[:1440].sum() == 0
    # injected values must not leak into the threshold
    assert np.allclose(pe.t1_threshold(g, 0.99, 2000)[3005], pe.t1_threshold(g, 0.99, 2000)[3005])


def test_recall_plain_loop():
    ev = pd.DataFrame({"family": ["a", "a", "b", "b", "b"], "start": [0, 10, 20, 30, 40], "end": [2, 12, 22, 32, 42]})
    al = np.zeros(50, bool); al[[1, 31, 42]] = True
    r = pe.recall_by_family(al, ev)
    exp = {"a": 1 / 2, "b": 2 / 3}
    for f, v in exp.items():
        assert abs(r["per_family"][f] - v) < 1e-12
    assert abs(r["macro"] - np.mean(list(exp.values()))) < 1e-12
    loop = []
    for _, e in ev.iterrows():
        loop.append(al[e.start:e.end + 1].any())
    assert (pe.event_hits(al, ev) == np.array(loop)).all()


def test_bootstrap_seed_and_sign():
    fam = np.array(["a"] * 30 + ["b"] * 30)
    ha = np.ones(60, bool); hb = np.zeros(60, bool); hb[::3] = True
    f = pe.diff_macro_recall(ha, hb, fam)
    r1 = pe.paired_bootstrap(f, fam, n=300, seed=5); r2 = pe.paired_bootstrap(f, fam, n=300, seed=5)
    assert r1 == r2 and r1["lo"] > 0 and abs(r1["point"] - (1 - 1 / 3)) < 1e-9


def test_lofo():
    fam = np.array(["a", "a", "b"])
    r = pe.lofo(fam, lambda held: np.array([True, False, True]))
    assert r["per_family"] == {"a": 0.5, "b": 1.0}


# ---- equivalence with the reference functions of the old notebooks (b2_lib.py, nb28/29b), copied verbatim ----
def _ref_episodes(alarm, gap=5):
    a = np.flatnonzero(alarm)
    if len(a) == 0: return 0
    return int(1 + np.sum(np.diff(a) > gap))


def _ref_eps_per_day(alarm, valid):
    a = alarm & valid; days = valid.sum() / 1440.0
    return _ref_episodes(a) / max(days, 1e-9)


def _ref_rolling_thr(score, q):
    return pd.Series(score).rolling(10080, min_periods=1440).quantile(q).shift(1).to_numpy()


def _ref_event_hits(alarm, ev):
    c = np.concatenate([[0], np.cumsum(alarm.astype(np.int64))]); t0 = ev.t0.to_numpy(); du = ev.dur.to_numpy()
    return (c[t0 + du] - c[t0]) > 0


def test_equals_old_b2_lib():
    rng = np.random.default_rng(3); n = 20000; s = rng.normal(size=n); s[rng.integers(0, n, 60)] += 6
    ts = _ts(n); valid = np.zeros(n, bool); valid[30:] = True
    for q in (0.99, 0.999):
        thr = _ref_rolling_thr(s, q); ref = s > np.where(np.isnan(thr), np.inf, thr)
        mine = pe.t1_alarms(s, q)
        assert (ref == mine).all()
        assert abs(_ref_eps_per_day(ref, valid) - pe.genuine_eps_per_day(mine, ts, valid=valid)) < 1e-12
    ev = pd.DataFrame({"t0": [100, 5000, 9000], "dur": [5, 10, 1], "family": [1, 2, 3]})
    al = np.zeros(n, bool); al[[104, 5010, 9001]] = True       # 104 inside, 5010 is t0+dur (outside), 9001 outside
    assert (pe.event_hits(al, pe.events_from_t0dur(ev)) == _ref_event_hits(al, ev)).all()
    assert pe.event_hits(al, pe.events_from_t0dur(ev)).tolist() == [True, False, False]


def test_t1_q_grid():
    rng = np.random.default_rng(4); n = 30000; s = rng.normal(size=n); valid = np.ones(n, bool); valid[:30] = False
    q = pe.t1_q(s, _ts(n), valid, 2.0)
    assert q in pe.QGRID and pe.genuine_eps_per_day(pe.t1_alarms(s, q), _ts(n), valid=valid) <= 2.0
