"""Historical context: mechanics of the notebook-53 retrieval on a toy index (past-only, gap, nearest first, exact match). The parity test against the real index is tests/test_history_parity.py (needs the files)."""
import json, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c03_engine import history as H


def toy(tmp, n=6000, seed=0):
    rng = np.random.default_rng(seed); cols = np.r_[0:12, 18:30]; mean = rng.normal(size=32); std = rng.uniform(.5, 2, 32)
    t = np.arange(H.DEVN - 3000, H.DEVN + n); Z = rng.normal(size=(len(t), 24)).astype('float32')
    ts = (H.AUG0_UTC + (t - H.DEVN) * 60_000_000_000).astype('int64')
    for s in ('BTCUSDT', 'ETHUSDT'):
        np.savez(tmp / f'history_index_v2_{s}.npz', Z=Z, t=t, PK=rng.normal(size=len(t)), MA=rng.integers(0, 5, len(t)).astype(float), ts=ts, mean=mean, std=std, cols=cols, weights=np.ones(24))
    (tmp / 'history_index_v2.json').write_text(json.dumps({'spec': {'candidate': 'c2'}, 'version_sha256': {}}))
    return Z, t


def test_past_only_gap_and_order(tmp_path):
    Z, t = toy(tmp_path); I = H.HistoryIndex(tmp_path); assert I.ok
    qe = H.DEVN + 4000; res = I.find('BTCUSDT', np.zeros(24) + I.sym['BTCUSDT']['mean'][I.sym['BTCUSDT']['cols']], I.sym['BTCUSDT']['ts'][t == qe][0])
    assert len(res) == 5 and [r['distance'] for r in res] == sorted(r['distance'] for r in res)
    tt = [int((r['end_ts_ns'] - H.AUG0_UTC) // 60_000_000_000 + H.DEVN) for r in res]
    assert all(x + H.OUTW < qe - H.W + 1 for x in tt) and all(abs(a - b) >= H.GAP for i, a in enumerate(tt) for b in tt[i + 1:])


def test_exact_window_is_nearest(tmp_path):
    Z, t = toy(tmp_path); I = H.HistoryIndex(tmp_path); a = I.sym['BTCUSDT']
    row = 1000; raw = Z[row] / a['weights'] * a['std'][a['cols']] + a['mean'][a['cols']]      # the descriptor that standardises to archive row 1000
    res = I.find('BTCUSDT', raw, a['ts'][t == H.DEVN + 5000][0])
    assert res[0]['end_ts_ns'] == int(a['ts'][row]) and res[0]['distance'] < 1e-4


def test_missing_folder_is_reported_not_raised(tmp_path):
    I = H.HistoryIndex(tmp_path / 'nope'); assert not I.ok and I.error
    assert H.descriptor_raw([np.zeros(60)] * 4, np.zeros((60, 6))).shape == (24,) and H.descriptor_raw([np.full(60, np.nan)] * 4, np.zeros((60, 6))) is None


def test_trace_of_a_matched_hour(tmp_path):
    n = 3000; ts = (H.AUG0_UTC - (n - np.arange(n)) * 60_000_000_000).astype('int64'); F = np.arange(n, dtype='float64'); thr = np.full(n, 5.0)
    np.savez(tmp_path / 'history_traces_v1.npz', BTCUSDT_ts=ts, BTCUSDT_F=F, BTCUSDT_thr=thr)
    I = H.HistoryIndex(tmp_path); assert not I.ok and 'BTCUSDT' in I.tr
    f, t = I.trace('BTCUSDT', int(ts[500]))
    assert len(f) == len(t) == 120 and f[59] == 500 and f[0] == 441 and f[-1] == 560            # 60 minutes ending at the match, then the 60 after it
    assert I.trace('BTCUSDT', int(ts[10])) is None and I.trace('BTCUSDT', int(ts[-5])) is None   # not enough minutes before / after
    assert I.trace('BTCUSDT', int(ts[500]) + 1) is None and I.trace('ETHUSDT', int(ts[500])) is None
    F2 = F.copy(); F2[520] = np.nan; np.savez(tmp_path / 'history_traces_v1.npz', BTCUSDT_ts=ts, BTCUSDT_F=F2, BTCUSDT_thr=thr)
    assert H.HistoryIndex(tmp_path).trace('BTCUSDT', int(ts[500])) is None                       # a gap in the trace is never drawn
