import os, numpy as np, pandas as pd
from tools import p_build_train as B, p_inj_train as J, p_feat3 as F
try:
    from tests.test_p_inj_train import toy
except ImportError:                                  # tests\ without __init__.py
    from test_p_inj_train import toy


def _archive(tmp_path, n_months=3, per=4000):
    d = toy(n=per * n_months, seed=1)
    for i in range(n_months):                                   # each toy 'month' starts on the 1st of a real month
        m = d.iloc[i * per:(i + 1) * per].copy(); m['ts'] = pd.date_range(f'2024-0{4 + i}-01', periods=per, freq='1min')
        m.to_parquet(tmp_path / f'BTCUSDT_2024-0{4 + i}.parquet')   # month token inside the file name
    return d


def test_loader_and_feature_table(tmp_path):
    _archive(tmp_path); t, files = B.load_tables(str(tmp_path), 'BTCUSDT', ['2024-04', '2024-05'])
    assert len(t) == 8000 and t.ts.is_monotonic_increasing and set(files) == {'2024-04', '2024-05'}
    ft = B.feature_table(t, 'BTCUSDT'); assert list(ft.columns) == ['ts'] + list(J.base6(t, 'BTCUSDT').columns) + F.FEATURES_V3
    assert ft[F.FEATURES_V3].iloc[:30].isna().any(axis=1).all() and not ft[F.FEATURES_V3].iloc[100:].isna().any().any()


def test_missing_file_is_a_clear_error(tmp_path):
    _archive(tmp_path)
    try: B.find_month_file(str(tmp_path), 'ETHUSDT', '2024-04'); assert False
    except SystemExit as e: assert 'expected exactly 1' in str(e)


def test_build_symbol_outputs(tmp_path, monkeypatch):
    _archive(tmp_path); out = tmp_path / 'out'
    B.build_symbol(str(tmp_path), str(out), 'BTCUSDT', '2024-04', '2024-05', '2024-04', '2024-06', 30.0)
    g = pd.read_parquet(out / 'BTCUSDT_genuine_features_2024-04_2024-06.parquet'); assert len(g) == 12000
    tr = pd.read_parquet(out / 'BTCUSDT_train_r0.parquet'); assert len(tr) == 8000 and (tr.label > 0).sum() > 0 and not tr.label.isin(J.NEW).any()
    ev = pd.read_csv(out / 'BTCUSDT_train_events.csv'); assert not ev.family.isin(J.NEW).any()
    import json; m = json.load(open(out / 'manifest_BTCUSDT.json')); assert m['p_feat3_spec_sha16'] == F.SPEC_SHA16 and m['rows_train'] == 8000
    # July/June (outside the train months) is never injected: genuine table has 12000 rows but replicas only 8000
    B.build_symbol(str(tmp_path), str(out), 'BTCUSDT', '2024-04', '2024-05', '2024-04', '2024-06', 30.0)    # second call: skipped, no overwrite


def test_v3_features_respond_to_the_families():
    d = toy(n=40000); f, ev = J.build_replica(d, 'BTCUSDT', 0, len(d), list(J.FAMS) * 12, 9)
    a = F.features_v3(d); b = F.features_v3(f)
    def med(col, fam):
        return float(np.nanmedian([np.nanmean(b[col].iloc[r.t0:r.t0 + r.dur]) - np.nanmean(a[col].iloc[r.t0:r.t0 + r.dur]) for r in ev[ev.family == fam].itertuples()]))
    assert med('same_size_share', 5) > 0 and med('dup_share', 5) > 0           # wash-like
    assert med('second_concentration', 3) > 0 and med('size_skew', 3) > 0       # volume burst
    assert med('second_coverage', 6) < 0                                        # liquidity stress
    assert abs(med('same_size_share', 4)) < 0.02 and abs(med('second_concentration', 4)) < 0.02   # trade-flow imbalance does not touch them
    assert med('volume_burst', 3) > 0 and med('trade_burst', 3) > 0


def test_loader_finds_subfolder_layout_and_explains_failures(tmp_path):
    (tmp_path / 'BTCUSDT').mkdir(); toy(n=500).to_parquet(tmp_path / 'BTCUSDT' / '2023_01.parquet')
    assert B.find_month_file(str(tmp_path), 'BTCUSDT', '2023-01').endswith('2023_01.parquet')
    try: B.find_month_file(str(tmp_path), 'BTCUSDT', '2023-02'); assert False
    except SystemExit as e: assert 'first ones' in str(e) and '2023_01.parquet' in str(e)
    try: B.find_month_file(str(tmp_path / 'nope'), 'BTCUSDT', '2023-01'); assert False
    except SystemExit as e: assert 'folder not found' in str(e)


def test_clean_ts_accepts_tz_aware_naive_and_epoch():
    base = pd.Series(pd.date_range('2024-04-01', periods=3, freq='1min'))
    want = base.astype('datetime64[ns]')
    sec = ((base - pd.Timestamp('1970-01-01')).dt.total_seconds()).astype('int64')      # independent of the datetime unit of this pandas version
    for s in (base.dt.tz_localize('UTC'), base, sec * 1000, sec):
        out = B.clean_ts(s); assert out.dtype == np.dtype('datetime64[ns]') and (out.to_numpy() == want.to_numpy()).all()


def test_loader_with_tz_aware_ts(tmp_path):
    d = toy(n=3000); d['ts'] = pd.date_range('2023-01-01', periods=3000, freq='1min', tz='UTC'); (tmp_path / 'BTCUSDT').mkdir()
    d.to_parquet(tmp_path / 'BTCUSDT' / 'BTCUSDT_2023-01.parquet'); t, _ = B.load_tables(str(tmp_path), 'BTCUSDT', ['2023-01'])
    assert t.ts.dtype == np.dtype('datetime64[ns]') and len(B.feature_table(t, 'BTCUSDT')) == 3000


def test_only_train_needs_no_july_then_only_genuine(tmp_path):
    _archive(tmp_path); out = tmp_path / 'out'                  # months 04,05,06 exist; 'July' (07) does not
    B.build_symbol(str(tmp_path), str(out), 'BTCUSDT', '2024-04', '2024-05', '2024-04', '2024-07', 30.0, only='train')
    assert (out / 'BTCUSDT_train_r0.parquet').exists() and not list(out.glob('*genuine*'))
    try: B.build_symbol(str(tmp_path), str(out), 'BTCUSDT', '2024-04', '2024-05', '2024-04', '2024-07', 30.0, only='genuine'); assert False
    except SystemExit as e: assert '2024-07' in str(e)
    B.build_symbol(str(tmp_path), str(out), 'BTCUSDT', '2024-04', '2024-05', '2024-04', '2024-06', 30.0, only='genuine')
    assert (out / 'BTCUSDT_genuine_features_2024-04_2024-06.parquet').exists()
