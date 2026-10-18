"""PARITY of the live descriptor with the saved notebook-53 index (needs the files from Colab: history_index_v2*.npz).
Set C03_ASSETS to the folder with the files and C03_EXPORT to engine_export_v1, then:   python -m pytest tests/test_history_parity.py -q -s
What it checks: for replay bars of 2024-08-01.., the 24-number descriptor computed live by the engine (scores + signed z) and standardised with the index's own mean/std must equal the stored index row of the
same minute (same window, same candidate c2). It writes tests/history_parity_result.json (do not commit)."""
import json, os, sys
from pathlib import Path
import numpy as np, pytest
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'frontend_streamlit'))
A = os.environ.get('C03_ASSETS'); EX = os.environ.get('C03_EXPORT')
pytestmark = pytest.mark.skipif(not (A and EX and (Path(A) / 'history_index_v2_BTCUSDT.npz').exists()), reason='needs C03_ASSETS (history index) and C03_EXPORT')
N_BARS = int(os.environ.get('C03_PARITY_BARS', 3500))


@pytest.mark.parametrize('sym', ['BTCUSDT', 'ETHUSDT'])
def test_descriptor_parity(sym):
    import live_engine as LE
    from c03_engine.store import Export
    from c03_engine import history as H
    ex = Export(EX, verify=True); j = LE.make_live(ex, sym); I = j.hidx; assert I.ok, I.error
    j.advance(min(j.n - 1, LE.FIRST + N_BARS)); a = I.sym[sym]; row = {int(t): r for r, t in enumerate(a['t'])}
    diffs, checked = [], 0
    for k in range(LE.FIRST + 80, j.done, 23):
        d = LE.window_descriptor(j, k)
        r = row.get(H.DEVN + k)
        if d is None or r is None:
            continue
        mine = I.standardise(sym, d); diffs.append(np.abs(mine - a['Z'][r].astype('float64'))); checked += 1
    assert checked >= 20, f'only {checked} comparable windows'
    D = np.array(diffs); res = {'symbol': sym, 'windows_compared': checked, 'max_abs_diff_std_units': float(D.max()), 'median_abs_diff': float(np.median(D)), 'share_within_1e-3': float((D < 1e-3).mean()),
                                'worst_dimension': H.NAMES[int(np.argmax(D.max(axis=0)))] if D.shape[1] == len(H.NAMES) else int(np.argmax(D.max(axis=0)))}
    out = Path(__file__).with_name('history_parity_result.json'); old = json.loads(out.read_text()) if out.exists() else {}; old[sym] = res; out.write_text(json.dumps(old, indent=1)); print(res)
    assert res['median_abs_diff'] < 1e-5 and res['max_abs_diff_std_units'] < 0.01, res
