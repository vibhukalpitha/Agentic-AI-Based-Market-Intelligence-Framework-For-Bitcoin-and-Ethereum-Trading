import numpy as np, pandas as pd
from tools import p_inj_train as J


def toy(n=9000, seed=0, hole=None):
    rng = np.random.default_rng(seed)
    ts = pd.date_range('2024-01-01', periods=n, freq='1min')
    if hole: ts = ts[:hole].append(ts[hole:] + pd.Timedelta(minutes=500))
    vol = rng.gamma(2, 3, n) + 0.5; buy = vol * np.clip(rng.normal(0.5, 0.1, n), 0.05, 0.95)
    tr = np.maximum(3, np.round(vol * rng.uniform(40, 80, n))); ag = np.maximum(1, np.round(tr * rng.uniform(0.5, 0.9, n)))
    lg = np.minimum(np.round(vol * rng.uniform(0.5, 2, n)), ag)
    c = 40000 * np.exp(np.cumsum(rng.normal(0, 3e-4, n))); o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + rng.uniform(1e-4, 4e-4, n)); lo = np.minimum(o, c) * (1 - rng.uniform(1e-4, 4e-4, n))
    q50 = rng.uniform(0.001, 0.01, n); q90 = q50 * 3; q99 = q50 * 10; qmx = q50 * 30
    sa = np.minimum(60, ag); sm = np.maximum(np.ceil(ag / sa), np.round(ag / sa * 1.5)); sm = np.minimum(sm, ag)
    ss = np.minimum(ag, np.maximum(1, np.round(ag * 0.1))); sp = np.minimum(ag, np.round(ag * 0.05))
    br = np.maximum(1, np.round(ag * 0.1)); sr = np.maximum(1, np.round(ag * 0.1)); nr = np.clip(np.round(ag * 0.5), 1, ag)
    return pd.DataFrame({'ts': ts, 'open': o, 'high': hi, 'low': lo, 'close': c, 'volume': vol, 'taker_buy_base': buy, 'n_trades': tr, 'n_agg': ag,
                         'large_count': lg, 'large_qty': np.minimum(lg * 0.2, vol), 'q_p50': q50, 'q_p90': q90, 'q_p99': q99, 'q_max': qmx,
                         'same_size_max': ss, 'same_size_prev': sp, 'dup_share': rng.uniform(0, 0.2, n), 'buy_run_max': br, 'sell_run_max': sr,
                         'n_runs': nr, 'sec_active': sa, 'sec_max_agg': sm})


ALL = list(J.FAMS)


def test_same_seed_identical_and_replay():
    d = toy(); a, ea = J.build_replica(d, 'BTCUSDT', 0, len(d), ALL * 2, 5); b, eb = J.build_replica(d, 'BTCUSDT', 0, len(d), ALL * 2, 5)
    assert a.equals(b) and ea.equals(eb)
    c, _ = J.inject(d, 'BTCUSDT', ea, 6); assert c.equals(a)           # replay of the stored events (incl. sign)


def test_placement_rules():
    d = toy(); f, ev = J.build_replica(d, 'BTCUSDT', 0, len(d), ALL * 3, 5); e = ev.sort_values('t0')
    assert len(e) == 24 and set(e.family) == set(ALL) and e.amp.between(0.5, 2.0).all()
    assert all(J.DUR[r.family][0] <= r.dur <= J.DUR[r.family][1] for r in e.itertuples())
    assert all(e.t0.iloc[i + 1] >= e.t0.iloc[i] + e.dur.iloc[i] + J.TAIL + J.GAP for i in range(len(e) - 1))
    assert (f.label == -1).sum() == J.TAIL * len(e)


def test_nothing_changes_outside_events():
    d = toy(); f, ev = J.build_replica(d, 'BTCUSDT', 0, len(d), ALL * 3, 5)
    b0, b1 = J.base6(d, 'BTCUSDT'), J.base6(f, 'BTCUSDT'); out = (f.label == 0).to_numpy()
    for c in ['log_returns', 'trade_flow_imbalance', 'hl_spread_ratio', 'rush_order_density', 'large_trade_count']:
        assert np.allclose(b0[c][out], b1[c][out], rtol=1e-9, atol=1e-12), c
    for c in J.NUM:
        if c in ('open', 'high', 'low', 'close'): continue
        assert np.array_equal(d[c].to_numpy()[out], f[c].to_numpy()[out]), c
    # volume_acceleration may move for 15 bars after an event (look-back window): compare it only where no event/tail lies in the previous 15 bars
    nonclean = (f.label != 0).astype(int).to_numpy(); c = np.r_[0, np.cumsum(nonclean)]; idx = np.arange(len(f))
    far = (c[idx + 1] - c[np.maximum(idx - 15, 0)]) == 0
    assert np.allclose(b0.volume_acceleration[far], b1.volume_acceleration[far], rtol=1e-9, atol=1e-12)


def test_invariants_hold_everywhere():
    d = toy(); f, _ = J.build_replica(d, 'BTCUSDT', 0, len(d), ALL * 3, 5)
    assert (f.volume >= 0).all() and (f.taker_buy_base <= f.volume + 1e-9).all() and (f.taker_buy_base >= -1e-12).all()
    assert (f.n_agg <= f.n_trades).all() and (f.large_count <= f.n_agg).all() and (f.large_qty <= f.volume + 1e-9).all()
    assert (f.q_p50 <= f.q_p90 + 1e-12).all() and (f.q_p90 <= f.q_p99 + 1e-12).all() and (f.q_p99 <= f.q_max + 1e-12).all()
    assert (f.same_size_max <= f.n_agg).all() and (f.same_size_prev <= f.n_agg).all() and f.dup_share.between(0, 1).all()
    assert (f.buy_run_max + f.sell_run_max <= f.n_agg).all() and (f.n_runs <= f.n_agg).all() and (f.n_runs >= 1).all()
    assert f.sec_active.between(1, 60).all() and (f.sec_max_agg <= f.n_agg).all() and (f.sec_max_agg * f.sec_active >= f.n_agg - 1e-9).all()
    assert (f.high >= f[['open', 'close']].max(axis=1) - 1e-9).all() and (f.low <= f[['open', 'close']].min(axis=1) + 1e-9).all() and (f.close > 0).all()


def test_no_event_over_a_time_hole():
    d = toy(hole=4500); f, ev = J.build_replica(d, 'BTCUSDT', 0, len(d), ALL * 3, 5)
    for r in ev.itertuples():
        a, b = r.t0 - J.LOOK, r.t0 + r.dur + J.TAIL
        assert not (a < 4500 <= b), (a, b)


def test_family_signatures():
    d = toy(n=40000); f, ev = J.build_replica(d, 'BTCUSDT', 0, len(d), ALL * 12, 9)
    def med(col, fam):
        return float(np.median([f[col].iloc[r.t0:r.t0 + r.dur].mean() / d[col].iloc[r.t0:r.t0 + r.dur].mean() for r in ev[ev.family == fam].itertuples()]))
    for fam in (1, 2, 3, 5, 7, 8): assert med('volume', fam) > 1.0, fam
    assert med('sec_max_agg', 3) > 1.0 and med('same_size_max', 5) > 1.0 and med('q_max', 3) > 1.0
    assert med('sec_active', 6) < 1.0 and med('n_runs', 4) < 1.0
    b1 = J.base6(f, 'BTCUSDT'); b0 = J.base6(d, 'BTCUSDT')
    hr = np.median([b1.hl_spread_ratio.iloc[r.t0:r.t0 + r.dur].mean() / b0.hl_spread_ratio.iloc[r.t0:r.t0 + r.dur].mean() for r in ev[ev.family == 6].itertuples()])
    assert hr > 1.0


def test_set_builder_old_only_and_count():
    d = toy(n=30000); big, evs = J.build_set(d, 'BTCUSDT', 0, len(d), 60, J.OLD, 3)
    assert len(evs) >= 60 and evs.family.isin(J.OLD).all() and not big.label.isin(J.NEW).any()
