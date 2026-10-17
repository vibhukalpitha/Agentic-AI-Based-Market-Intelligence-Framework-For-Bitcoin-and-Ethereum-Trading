"""Golden test: priming on the first 11,500 replay bars and streaming the rest bar by bar must reproduce every expected column of the replay pack
(three model scores, fused score, thresholds, state) - the same numbers the notebooks computed in batch with the code that opened TEST."""
import os, sys, numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from c03_engine.store import Export
from c03_engine.core import SymbolEngine

ROOT = os.environ.get('C03_EXPORT')
pytestmark = pytest.mark.skipif(not ROOT, reason='set C03_EXPORT to the engine_export_v1 folder')
NPRIME, NSTEP = 11500, int(os.environ.get('C03_GOLDEN_STEPS', '600'))


@pytest.fixture(scope='module')
def ex(): return Export(ROOT)


@pytest.mark.parametrize('sym', ['BTCUSDT', 'ETHUSDT'])
def test_stream_equals_batch(ex, sym):
    df = pd.read_parquet(os.path.join(ROOT, 'replay', f'{sym}_replay_2024-08-01_to_08-10.parquet')); eng = SymbolEngine(ex, sym)
    eng.prime(df.iloc[:NPRIME][ex.B.FEATS], list(df.ts[:NPRIME])); out = []
    for k in range(NPRIME, NPRIME + NSTEP): out.append(eng.step(df.iloc[k][ex.B.FEATS].to_numpy('float64'), df.ts[k]))
    sl = df.iloc[NPRIME:NPRIME + NSTEP]
    for key, col, tol in [('ae', 'exp_ae', 1e-5), ('xgboost', 'exp_xgb', 1e-4), ('transformer', 'exp_tr', 2e-3)]:
        got = np.array([o['scores'][key] for o in out]); assert np.allclose(got, sl[col].to_numpy('float64'), atol=tol, equal_nan=True), (key, np.nanmax(np.abs(got - sl[col].to_numpy())))
    fz = np.array([o['fusion'] for o in out]); assert np.allclose(fz, sl.exp_fusion.to_numpy('float64'), atol=1e-3, equal_nan=True), np.nanmax(np.abs(fz - sl.exp_fusion.to_numpy()))
    ta = np.array([o['thr_alert'] for o in out]); e = sl.exp_thr_alert.to_numpy('float64'); assert np.allclose(ta, e, atol=1e-3, equal_nan=True)
    agree = np.mean([o['state'] for o in out] == sl.exp_state.to_numpy()); assert agree >= 0.999, agree
