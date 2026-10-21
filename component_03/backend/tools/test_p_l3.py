import os, sys, numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import p_l3 as G, p_feat3 as F
from test_p_feat3 import table


def base(n=6000):
    d = table(n, 1); r = np.random.default_rng(2); p = 60000 * np.exp(np.cumsum(r.normal(0, 3e-4, n)))
    d['open'] = np.r_[p[0], p[:-1]]; d['close'] = p; d['high'] = np.maximum(d.open, d.close) * 1.0003; d['low'] = np.minimum(d.open, d.close) * .9997
    d['taker_buy_base'] = d.volume * .5; d['n_trades'] = d.n_agg * 2; d['large_count'] = r.integers(0, 5, n); d['large_qty'] = d['large_count'] * .2
    d['q_p90'] = d.q_p50 * 5; d['q_p99'] = d.q_p50 * 20; d['q_max'] = d.q_p50 * 100; d['sec_max_agg'] = np.minimum(d.sec_max_agg, d.n_agg); return d


def test_generate_labels_deterministic_and_untouched_outside():
    b = base(); df, ev, meta = G.generate(b, 'BTCUSDT', 'val', 40); df2, ev2, _ = G.generate(b, 'BTCUSDT', 'val', 40)
    assert df.equals(df2) and ev.equals(ev2) and len(ev) == 40 and set(ev.family) == set(G.FAMILIES) and ev.unseen.sum() == 10 and meta['generator_sha16'] == G.GENERATOR_SHA16
    out = df['l3_event'] == -1; cols = [c for c in b.columns if c != 'ts']; assert np.allclose(df.loc[out, cols].to_numpy(float), b.loc[out, cols].to_numpy(float))
    assert (ev.start_idx.to_numpy()[1:] - ev.end_idx.to_numpy()[:-1] >= 90).all() and ev.start_idx.min() >= 180


def test_rows_are_consistent_and_events_actually_change():
    b = base(); df, ev, _ = G.generate(b, 'ETHUSDT', 'final', 40); e = df[df.l3_event >= 0]
    assert (e.high >= e[['open', 'close']].max(axis=1) - 1e-9).all() and (e.low <= e[['open', 'close']].min(axis=1) + 1e-9).all() and (e.taker_buy_base <= e.volume + 1e-9).all()
    assert (e.large_count <= e.n_agg).all() and (e.sec_max_agg <= e.n_agg).all() and (e.same_size_max <= e.n_agg).all() and (e.q_p50 <= e.q_p90).all() and (e.q_p90 <= e.q_p99).all() and (e.q_p99 <= e.q_max).all() and (e.sec_active <= 60).all()
    f0, f1 = F.features_v3(b), F.features_v3(df)
    for fam, col, sign in [('wash_cluster_v3', 'same_size_share', 1), ('volume_burst_v3', 'volume_burst', 1), ('periodic_pinging', 'second_concentration', 1), ('iceberg_absorption', 'same_size_share', 1)]:
        m = (df.l3_family == fam).to_numpy(); assert (f1.loc[m, col] - f0.loc[m, col]).median() * sign > 0, fam


def test_seeds_differ_and_final_is_single_use(tmp_path):
    b = base(); a, _, _ = G.generate(b, 'BTCUSDT', 'val', 30); c, _, _ = G.generate(b, 'BTCUSDT', 'final', 30); assert not a.equals(c)
    with pytest.raises(ValueError): G.generate(b.iloc[:300], 'BTCUSDT', 'val', 40)
    p = tmp_path / 'b.parquet'; b.to_parquet(p); sys.argv = ['p_l3', '--set', 'final', '--symbol', 'BTCUSDT', '--base', str(p), '--out', str(tmp_path), '--n-events', '16', '--open-final']; G.main()
    with pytest.raises(SystemExit): G.main()
    sys.argv = ['p_l3', '--set', 'final', '--symbol', 'ETHUSDT', '--base', str(p), '--out', str(tmp_path), '--n-events', '16']
    with pytest.raises(SystemExit): G.main()
