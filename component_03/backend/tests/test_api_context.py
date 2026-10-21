"""/reliability, /explain, /similar: served from the same engine in replay mode (live mode uses the identical SymbolService). Toy history index; no model or score is changed."""
import os, sys, time, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.environ.get('C03_EXPORT'); pytestmark = pytest.mark.skipif(not ROOT, reason='set C03_EXPORT')


def test_context_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import test_history as TH
    TH.toy(tmp_path); monkeypatch.setenv('C03_ASSETS', str(tmp_path))
    from c03_engine.api import create_app
    app = create_app(ROOT, mode='replay', verify=False, replay_default={'from': '2024-08-05T00:00:00', 'speed': 20000})
    with TestClient(app) as c:
        t0 = time.time(); n = 0
        while time.time() - t0 < 120 and n < 75:
            time.sleep(1)
            r = c.get('/api/c03/bars', params={'symbol': 'BTCUSDT', 'limit': 500})
            n = len(r.json()) if r.status_code == 200 else 0
        assert n >= 75, n
        r = c.get('/api/c03/reliability', params={'symbol': 'BTCUSDT', 'limit': 30}).json()
        assert r['current'] is not None and len(r['series']) == 30 and {'level', 'checks_met', 'conformal_set'} <= set(r['series'][0])
        e = c.get('/api/c03/explain', params={'symbol': 'BTCUSDT'})
        assert e.status_code == 200 and len(e.json()['detail']['shap']) == 6 and len(e.json()['detail']['z']) == 6
        assert c.get('/api/c03/explain', params={'symbol': 'BTCUSDT', 'at': '2001-01-01T00:00:00Z'}).status_code == 404
        s = c.get('/api/c03/similar', params={'symbol': 'BTCUSDT', 'k': 3}).json()
        assert s['status'] == 'ok' and len(s['matches']) == 3 and len(s['now_fused']) == 60 and s['matches'][0]['distance'] <= s['matches'][-1]['distance']
        assert c.get('/api/c03/reliability', params={'symbol': 'XXX'}).status_code == 404
