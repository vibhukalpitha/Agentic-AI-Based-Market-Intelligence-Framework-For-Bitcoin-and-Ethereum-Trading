"""Quick checks of the dashboard.  Run:  python -m pytest -q test_app.py
Live-model mode needs ../backend (c03_engine + export). The stored-results tests need data/*_dash.json and are skipped without them."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import live_engine as LE

LIVE = LE.find_backend(HERE)[0] is not None
STORED = (HERE / 'data' / 'BTCUSDT_dash.json').exists()


def fresh(stored=False):
    at = AppTest.from_file(str(HERE / 'app.py'), default_timeout=900)
    if stored:
        at.session_state['_data_dir'] = str(HERE / 'data')
    at.run()
    return at


def btn(at, label):
    return next(b for b in at.button if label in b.label)


@pytest.mark.skipif(not LIVE, reason='backend/export not found')
def test_live_mode_runs_the_models():
    at = fresh()
    assert not at.exception and len(at.tabs) == 4
    j = at.session_state['_B']['BTCUSDT']
    assert isinstance(j, LE.LiveSymbol) and j.done > j.first
    i0 = at.session_state['i']
    btn(at, '1 bar \u25b6').click().run()
    assert not at.exception and at.session_state['i'] == i0 + 1


@pytest.mark.skipif(not LIVE, reason='backend/export not found')
def test_live_symbol_switch_and_step_back():
    at = fresh()
    i0 = at.session_state['i']
    btn(at, '\u25c0 1 bar').click().run()
    assert at.session_state['i'] == i0 - 1
    at.session_state['sym_sel'] = 'ETH/USDT'
    at.run()
    assert not at.exception and at.session_state['sym'] == 'ETHUSDT'


@pytest.mark.skipif(not STORED, reason='stored bundles not in data/')
def test_stored_mode_fallback():
    at = fresh(stored=True)
    assert not at.exception and len(at.tabs) == 4
    btn(at, 'Aug 5').click().run()
    assert not at.exception and '2024-08-05' in at.session_state['_B']['BTCUSDT'].ts[at.session_state['i'] + 30]


# ---------------------------------------------------------------------------- bad data (viva: "what happens with garbage / missing data")
@pytest.mark.skipif(not LIVE, reason='backend/export not found')
def test_bad_minute_in_the_recording_is_skipped_and_shown():
    import numpy as np
    at = fresh()
    j = at.session_state['_B']['BTCUSDT']
    k = j.done + 1
    j.df.loc[k, 'trade_flow_imbalance'] = np.nan                    # one empty value inside the recording
    at.session_state['i'] = k
    at.run()
    assert not at.exception and k in j.bad and j.state[k] == 0 and np.isnan(j.fusion[k]) and np.isnan(j.integ[k])
    assert any('No score for this minute' in w.value for w in at.warning)
    assert any('skipped' in c.value for c in at.caption)
    at.session_state['i'] = k + 1
    at.run()
    assert not at.exception and np.isfinite(j.fusion[k + 1]) and (k + 1) not in j.bad       # the next valid minute is scored normally


@pytest.mark.skipif(not LIVE, reason='backend/export not found')
def test_data_check_panel_rejects_garbage_and_accepts_a_real_minute():
    at = fresh()
    sb = next(s for s in at.selectbox if s.key == 'dq_pick')
    sb.select('All six values are 0').run()
    assert not at.exception and any('Rejected' in e.value for e in at.error)
    next(s for s in at.selectbox if s.key == 'dq_pick').select('Text instead of numbers').run()
    assert not at.exception and any('Rejected' in e.value for e in at.error)
    next(s for s in at.selectbox if s.key == 'dq_pick').select('A normal minute (copied from the recording)').run()
    assert not at.exception and any('Accepted' in s_.value for s_ in at.success) and not any('Rejected' in e.value for e in at.error)
