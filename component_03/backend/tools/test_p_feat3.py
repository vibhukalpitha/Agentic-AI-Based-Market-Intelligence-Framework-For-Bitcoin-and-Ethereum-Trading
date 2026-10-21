import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
import p_feat3 as P


def table(n=200, seed=0):
    r = np.random.default_rng(seed); na = r.integers(100, 400, n)
    return pd.DataFrame({'ts': pd.date_range('2024-01-01', periods=n, freq='min'), 'volume': r.gamma(2, 3, n) + .1, 'n_agg': na, 'large_qty': r.gamma(1, .2, n), 'q_p50': .002 + r.random(n) * 1e-3, 'q_p90': .02 + r.random(n) * 1e-2,
        'q_p99': .1 + r.random(n) * .05, 'q_max': .5 + r.random(n), 'same_size_max': r.integers(2, 20, n), 'same_size_prev': r.integers(0, 30, n), 'dup_share': r.random(n) * .5, 'buy_run_max': r.integers(1, 12, n),
        'sell_run_max': r.integers(1, 12, n), 'n_runs': (na * .5).astype(int), 'sec_active': r.integers(30, 61, n), 'sec_max_agg': r.integers(5, 30, n)})


def test_shape_and_formulas():
    d = table(); f = P.features_v3(d); assert list(f.columns) == ['ts'] + P.FEATURES_V3 and len(f) == len(d)
    i = 100; assert np.isclose(f.loc[i, 'size_skew'], np.log((d.q_p99[i] + 1e-8) / (d.q_p50[i] + 1e-8)))
    assert np.isclose(f.loc[i, 'same_size_share'], d.same_size_max[i] / d.n_agg[i]); assert np.isclose(f.loc[i, 'second_coverage'], d.sec_active[i] / 60)
    assert np.isclose(f.loc[i, 'run_imbalance'], (d.buy_run_max[i] - d.sell_run_max[i]) / (d.buy_run_max[i] + d.sell_run_max[i] + 1))
    base = np.median(d.volume[i - 60:i]); assert np.isclose(f.loc[i, 'volume_burst'], np.log((d.volume[i] + 1e-8) / (base + 1e-8)))


def test_past_only_and_nan_head():
    d = table(); f = P.features_v3(d); assert f.loc[:29, 'volume_burst'].isna().all() and f.loc[30:, 'volume_burst'].notna().all()
    d2 = d.copy(); d2.loc[150:, 'volume'] *= 100; f2 = P.features_v3(d2)      # changing the future never changes earlier rows
    assert np.allclose(f.loc[:149, P.FEATURES_V3].fillna(0), f2.loc[:149, P.FEATURES_V3].fillna(0))
    assert f.loc[150:, 'volume_burst'].iloc[0] != f2.loc[150:, 'volume_burst'].iloc[0]


def test_missing_column_and_sha():
    import pytest
    with pytest.raises(ValueError): P.features_v3(table().drop(columns=['q_max']))
    assert len(P.SPEC_SHA16) == 16
