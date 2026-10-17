"""What happens when data is missing, garbage, all zeros, out of order or too short (viva: 'user inputs garbage').
Part 1 needs no export (pure input checks). Part 2 needs C03_EXPORT (real engine): a bad minute must leave NO trace in the engine memory."""
import json, os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from c03_engine import validate as V

ROOT = os.environ.get('C03_EXPORT')
GOOD = [0.0008, 0.12, 0.0033, 12.0, 0.4, 3.0]


def codes(p): return [x['code'] for x in p]


# ---------------------------------------------------------------- part 1: input checks (no export needed)
def test_good_minute_passes():
    assert V.check_features(GOOD) == []


@pytest.mark.parametrize('vec,code', [
    ([np.nan] + GOOD[1:], 'not_a_number'), ([np.inf] + GOOD[1:], 'not_a_number'), ([-np.inf] + GOOD[1:], 'not_a_number'),
    ([None] * 6, 'not_a_number'), (['a', 'b', 'c', 'd', 'e', 'f'], 'not_a_number'),
    ([0.0] * 6, 'all_zero'), ([1, 2, 3], 'bad_shape'), ([], 'bad_shape'),
])
def test_garbage_features_are_rejected(vec, code):
    assert code in codes(V.check_features(vec))


@pytest.mark.parametrize('i,val', [(0, 0.9), (1, 1.5), (2, 0.9), (2, -0.1), (3, -1.0), (4, 1e12), (5, -5.0)])
def test_impossible_values_are_rejected(i, val):
    v = list(GOOD); v[i] = val
    assert 'out_of_range' in codes(V.check_features(v))


def test_largest_real_values_are_accepted():
    # the largest values seen in the Aug 2024 BTC/ETH data must never be rejected
    assert V.check_features([-0.0878, 0.9925, 0.1102, 683.1, 95.6, 3082]) == []


GOOD_BAR = dict(open=60000., high=60100., low=59900., close=60050., volume=10., trades=100, taker_buy_base=5.)


def test_good_bar_passes():
    assert V.check_bar(GOOD_BAR) == []


@pytest.mark.parametrize('chg,code', [
    (dict(open=0., high=0., low=0., close=0., volume=0., trades=0, taker_buy_base=0.), 'non_positive_price'),
    (dict(close=float('nan')), 'not_a_number'), (dict(close=-1.), 'non_positive_price'),
    (dict(high=59000., low=61000.), 'high_below_low'), (dict(high=59000.), 'high_below_low'),
    (dict(volume=-5.), 'negative_volume'), (dict(trades=-1), 'negative_trades'),
    (dict(taker_buy_base=99.), 'taker_gt_volume'), (dict(volume=0., taker_buy_base=0., trades=100), 'zero_volume_with_trades'),
    (dict(trades=None), 'missing_field'), (dict(close='abc'), 'not_a_number'),
])
def test_garbage_bars_are_rejected(chg, code):
    b = dict(GOOD_BAR); b.update(chg)
    assert code in codes(V.check_bar(b))


def test_not_a_bar_at_all():
    assert codes(V.check_bar(None)) == ['missing_field']
    assert codes(V.check_bar('picture.jpg')) == ['missing_field']


def _hist(n=1600):
    rg = np.random.default_rng(0)
    return pd.DataFrame({f: np.abs(rg.normal(0.05, 0.02, n)) + 0.01 for f in V.FEATS})   # small positive values: all inside the allowed ranges


def test_warmup_too_short_has_a_clear_message():
    with pytest.raises(V.DataError) as e: V.check_warmup(_hist(500))
    assert e.value.code == 'too_few_bars' and '1,500' in e.value.message and '500' in e.value.message


def test_warmup_missing_columns_names_them():
    with pytest.raises(V.DataError) as e: V.check_warmup(_hist().drop(columns=['volume_acceleration', 'large_trade_count']))
    assert e.value.code == 'missing_columns' and 'volume_acceleration' in e.value.message


def test_warmup_bad_rows_are_counted():
    h = _hist(); h.iloc[10, 2] = np.nan; h.iloc[20, :] = 0.0; h.iloc[30, 4] = 1e9
    with pytest.raises(V.DataError) as e: V.check_warmup(h)
    assert e.value.code == 'bad_rows' and '3 of 1,600' in e.value.message


def test_warmup_text_cells():
    h = _hist().astype(object); h.iloc[5, 1] = 'oops'
    with pytest.raises(V.DataError) as e: V.check_warmup(h)
    assert e.value.code == 'bad_rows'


def test_clean_warmup_passes():
    assert V.check_warmup(_hist()) == 1600


# ---------------------------------------------------------------- part 2: the real engine (needs C03_EXPORT)
needs_export = pytest.mark.skipif(not ROOT, reason='set C03_EXPORT')
SYM = 'BTCUSDT'


@pytest.fixture(scope='module')
def env():
    from c03_engine.store import Export
    ex = Export(ROOT); df = pd.read_parquet(os.path.join(ROOT, 'replay', f'{SYM}_replay_2024-08-01_to_08-10.parquet')).reset_index(drop=True)
    return ex, df, list(ex.B.FEATS)


def fresh(env):
    from c03_engine.service import SymbolService
    ex, df, feats = env; s = SymbolService(ex, SYM); s.prime(df.iloc[:1500][feats], list(df.ts[:1500])); return s


def feed(env, svc, idx, override=None):
    ex, df, feats = env; out = []
    for j, k in enumerate(idx):
        row = df.loc[k, feats].to_numpy('float64')
        if override and j in override: row = override[j]
        out.append(svc.on_bar(row, str(df.ts[k]), {'close': float(df.close[k])}))
    return out


BAD = {'nan': [np.nan, 0.1, 0.003, 10, 0.2, 1], 'inf': [0.001, np.inf, 0.003, 10, 0.2, 1], 'zeros': [0] * 6, 'huge': [0.001, 0.1, 0.003, 10, 1e12, 1],
       'negative_count': [0.001, 0.1, 0.003, 10, 0.2, -5], 'all_nan': [np.nan] * 6, 'short': [1, 2, 3], 'text': ['a'] * 6}


@needs_export
@pytest.mark.parametrize('name', list(BAD))
def test_bad_minute_leaves_no_trace(env, name):
    idx = list(range(1500, 1530)); bad_at = 5
    a = fresh(env); ra = feed(env, a, idx, {bad_at: BAD[name]})           # garbage at position 5
    b = fresh(env); rb = feed(env, b, [k for j, k in enumerate(idx) if j != bad_at])  # same minutes, that minute simply never arrived
    st = ra[bad_at][0]
    assert st['state'] == 'NO_DATA' and st['fused_score'] is None and st['integrity_score'] is None and st['data_quality']['ok'] is False and st['data_quality']['problems']
    assert ra[bad_at][1][0]['type'] == 'bad_input'
    fa = [r[0]['fused_score'] for j, r in enumerate(ra) if j != bad_at]; fb = [r[0]['fused_score'] for r in rb]
    assert all(x == y for x, y in zip(fa, fb)), 'a bad minute changed later scores'              # exactly equal: nothing leaked into the memory
    assert all(r[0]['state'] != 'NO_DATA' for j, r in enumerate(ra) if j != bad_at)
    json.dumps(__import__('c03_engine.api', fromlist=['clean']).clean(st))                       # the API can send it as JSON


@needs_export
def test_bad_minute_never_shown_as_normal_and_never_raises_alert(env):
    s = fresh(env); r = feed(env, s, [1500, 1501], {1: BAD['inf']})
    assert r[1][0]['state'] == 'NO_DATA' and s.state in ('NORMAL', 'WATCH', 'ALERT') and s.episodes == []


@needs_export
def test_duplicate_and_out_of_order_minutes_are_rejected(env):
    ex, df, feats = env; s = fresh(env)
    ok = feed(env, s, [1500, 1501, 1502]); assert all(r[0]['state'] != 'NO_DATA' for r in ok)
    dup = feed(env, s, [1502]); back = feed(env, s, [1500])
    assert dup[0][0]['state'] == 'NO_DATA' and 'duplicate_or_out_of_order' in codes(dup[0][0]['data_quality']['problems'])
    assert back[0][0]['state'] == 'NO_DATA'
    assert feed(env, s, [1503])[0][0]['state'] != 'NO_DATA'                                      # recovers by itself


@needs_export
def test_missing_minutes_warn_but_do_not_invent_data(env):
    s = fresh(env); feed(env, s, [1500, 1501])
    r = feed(env, s, [1505])[0][0]                                                               # 1502..1504 never arrived
    assert r['state'] != 'NO_DATA' and r['data_quality']['ok'] and any('3 minute(s) are missing' in w for w in r['data_quality']['warnings'])


@needs_export
def test_many_bad_minutes_in_a_row_then_recovery(env):
    s = fresh(env); r = feed(env, s, list(range(1500, 1520)), {j: BAD['zeros'] for j in range(2, 12)})
    assert [x[0]['state'] == 'NO_DATA' for x in r] == [j in range(2, 12) for j in range(20)]
    assert r[11][0]['data_quality']['consecutive_bad'] == 10 and r[12][0]['data_quality']['bad_minutes_total'] == 10


@needs_export
def test_service_prime_gives_clear_errors(env):
    from c03_engine.service import SymbolService
    ex, df, feats = env
    with pytest.raises(V.DataError, match='1,500'): SymbolService(ex, SYM).prime(df.iloc[:500][feats], list(df.ts[:500]))
    with pytest.raises(V.DataError, match='volume_acceleration'): SymbolService(ex, SYM).prime(df.iloc[:1500][feats[:4]], list(df.ts[:1500]))
    bad = df.iloc[:1500][feats].copy(); bad.iloc[100, 0] = np.nan
    with pytest.raises(V.DataError, match='1 of 1,500'): SymbolService(ex, SYM).prime(bad, list(df.ts[:1500]))


@needs_export
def test_good_data_is_unchanged_by_the_checks(env):
    """the checks must not reject a single real minute: all 14,400 minutes of both symbols pass"""
    ex, df, feats = env
    A = df[feats].to_numpy('float64')
    assert sum(1 for r in A if V.check_features(r)) == 0
