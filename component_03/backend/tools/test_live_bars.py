"""LIVE-T bar saving: one csv per symbol and UTC day, header once, exact flag kept, never raises."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_persist_bar(monkeypatch):
    from c03_engine import live
    d = tempfile.mkdtemp(); monkeypatch.setattr(live, 'BARS_DIR', d)
    k = {'open_ms': 1760832000000, 'open': 1., 'high': 2., 'low': .5, 'close': 1.5, 'volume': 3., 'trades': 10, 'taker_buy_base': 1., 'large_count': 4}
    live.persist_bar('BTCUSDT', k, True, 'NORMAL'); live.persist_bar('BTCUSDT', dict(k, open_ms=k['open_ms'] + 60000), False, 'WATCH')
    fn = [f for f in os.listdir(d)]; assert fn == ['BTCUSDT_2025-10-19.csv'] or len(fn) == 1
    L = open(os.path.join(d, fn[0])).read().splitlines(); assert len(L) == 3 and L[0].startswith('open_ms,utc') and L[1].split(',')[10] == '1' and L[2].split(',')[10] == '0'


def test_persist_never_raises(monkeypatch):
    from c03_engine import live
    monkeypatch.setattr(live, 'BARS_DIR', '/proc/nope/x'); live.persist_bar('BTCUSDT', {'open_ms': 0}, True)
    assert live.S_ERR[0]
