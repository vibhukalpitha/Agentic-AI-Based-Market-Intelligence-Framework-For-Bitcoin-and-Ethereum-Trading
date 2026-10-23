"""v0.2.2: the newest warm-up minutes are scored bar by bar after priming (prime_and_seed). Check on the replay pack: the page has bars at once, the 60-minute window for 'similar past'
is full, and the seeded scores equal the expected (batch) scores of the replay pack."""
import os, sys, numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from c03_engine.store import Export
from c03_engine.service import SymbolService
from c03_engine.live import prime_and_seed

ROOT = os.environ.get('C03_EXPORT')
pytestmark = pytest.mark.skipif(not ROOT, reason='set C03_EXPORT to the engine_export_v1 folder')


@pytest.mark.parametrize('sym', ['BTCUSDT', 'ETHUSDT'])
def test_seed(sym):
    ex = Export(ROOT); df = pd.read_parquet(os.path.join(ROOT, 'replay', f'{sym}_replay_2024-08-01_to_08-10.parquet')); n = 11600
    d = df.iloc[:n]; rows = d[ex.B.FEATS].to_numpy('float64').tolist(); tss = list(d.ts); ks = [{'close': float(c)} for c in d.close]
    svc = SymbolService(ex, sym); seeded = prime_and_seed(svc, rows, tss, ks, seed=180)
    assert seeded == 180 and len(svc.bars) == 180
    assert len(svc.dwin) == 180                                   # every seeded minute is kept for the 'similar past' query (newest, or any held minute via `at`)
    got = np.array([b['fused_score'] for b in svc.bars], 'float64'); exp = d.exp_fusion.to_numpy('float64')[-180:]
    assert np.allclose(got, exp, atol=1e-2, equal_nan=True), np.nanmax(np.abs(got - exp))     # same tolerance spirit as the golden test (float tie in XGBoost, see replay parity)
    assert [b['bar_open_utc'] for b in svc.bars][-1].startswith(str(d.ts.iloc[-1])[:16].replace(' ', 'T'))
    assert svc.last_m is not None
    # the next live minute must be accepted after the seeded ones
    nxt = df.iloc[n]; st, ev = svc.on_bar(np.asarray(nxt[ex.B.FEATS], 'float64'), nxt.ts, {'close': float(nxt.close)})
    assert len(svc.bars) == 181


def test_similar_at():
    """/similar with `at`: the newest minute gives the same answer as without `at`; an older held minute gives its own answer; an unheld or bad time says so."""
    sym = 'BTCUSDT'; ex = Export(ROOT); df = pd.read_parquet(os.path.join(ROOT, 'replay', f'{sym}_replay_2024-08-01_to_08-10.parquet')); n = 11600
    d = df.iloc[:n]; rows = d[ex.B.FEATS].to_numpy('float64').tolist(); tss = list(d.ts); ks = [{'close': float(c)} for c in d.close]
    svc = SymbolService(ex, sym); prime_and_seed(svc, rows, tss, ks, seed=180)
    if not svc.hidx.ok:
        pytest.skip('history index not loaded (set C03_ASSETS)')
    a, b = svc.similar(5), svc.similar(5, svc.bars[-1]['bar_open_utc'])
    assert a['status'] == 'ok' and a == b
    old = svc.similar(5, svc.bars[-100]['bar_open_utc'])
    assert old['status'] == 'ok' and old['query_end_utc'] == svc.bars[-100]['bar_open_utc'] and old['matches'] != a['matches']
    assert svc.similar(5, svc.bars[-121]['bar_open_utc'])['status'] == 'ok'          # still has 60 minutes before it
    assert svc.similar(5, svc.bars[-125]['bar_open_utc'])['status'] == 'collecting'   # fewer than 60 minutes held before it
    assert svc.similar(5, '2020-01-01T00:00:00Z')['status'] == 'unavailable'
    assert svc.similar(5, 'garbage')['status'] == 'unavailable'
