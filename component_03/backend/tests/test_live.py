"""Live source test with a local fake Binance (no internet): synthetic klines + trades in the real payload format; the engine must score exactly the closed bars, count large trades per minute, and fill a gap."""
import asyncio, json, os, sys, time, numpy as np, pandas as pd, pytest, websockets
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
ROOT = os.environ.get('C03_EXPORT'); pytestmark = pytest.mark.skipif(not ROOT, reason='set C03_EXPORT')


def test_live_pipeline():
    from c03_engine import live
    from c03_engine.api import create_app
    from c03_engine.store import Export
    ex = Export(ROOT); rg = np.random.default_rng(0); t0 = (int(time.time()) // 60) * 60000 - 1700 * 60000; price = {'BTCUSDT': 60000.0, 'ETHUSDT': 3000.0}
    def kl(sym, i):
        p = price[sym] * (1 + rg.normal(0, 5e-4)); v = float(rg.gamma(2, 3) + .1); return [t0 + i * 60000, p, p * 1.0004, p * .9996, p, v, 0, 0, int(rg.integers(80, 300)), v * .5, 0, 0]
    hist = {s: [kl(s, i) for i in range(1700)] for s in price}
    async def fake_fetch(sym, start_ms=None, end_ms=None, n=None):
        ks = hist[sym]
        if n: ks = ks[-n:]
        else: ks = [k for k in ks if start_ms <= k[0] <= end_ms]
        return [{'open_ms': k[0], 'open': k[1], 'high': k[2], 'low': k[3], 'close': k[4], 'volume': k[5], 'trades': k[8], 'taker_buy_base': k[9], 'large_count': 0} for k in ks]
    live.fetch_klines = fake_fetch; live.WARMUP_BARS = 1650
    async def run():
        sent = []
        async def server(ws):
            for s in price:     # a few big trades in the next minute, then the closed kline of that minute; then skip 2 minutes (gap) and send another closed kline
                for j in range(3): await ws.send(json.dumps({'stream': 'x', 'data': {'e': 'trade', 's': s, 'q': '5.0', 'T': t0 + 1700 * 60000 + 1000 * (j + 1)}}))
            for i in (1700, 1703):
                for s in price:
                    k = kl(s, i); hist[s].append(k) if i == 1700 else hist[s].extend([kl(s, 1701), kl(s, 1702), k])
                    await ws.send(json.dumps({'stream': 'x', 'data': {'e': 'kline', 's': s, 'k': {'t': k[0], 'o': str(k[1]), 'h': str(k[2]), 'l': str(k[3]), 'c': str(k[4]), 'v': str(k[5]), 'n': k[8], 'V': str(k[9]), 'x': True}}}))
                await asyncio.sleep(0.3)
            await asyncio.sleep(2)
        srv = await websockets.serve(server, 'localhost', 8765); live.WS_URL = 'ws://localhost:8765'
        class Hub:
            async def send(self, m): sent.append(m)
        S = {}; task = asyncio.create_task(live.run_live(ex, S, Hub(), lambda: {s: __import__('c03_engine.service', fromlist=['x']).SymbolService(ex, s) for s in ex.symbols}))
        await asyncio.sleep(25); task.cancel(); srv.close(); return sent, S
    sent, S = asyncio.run(run())
    bars = [m for m in sent if m['type'] == 'bar_scored' and m['data']['symbol'] == 'BTCUSDT']; types = {m['type'] for m in sent}
    assert len(bars) >= 4 and 'data_gap' in types, (len(bars), types)       # bar 1700, gap-filled 1701, 1702, then 1703
    assert S['svc']['BTCUSDT'].bars[-1]['features']['large_trade_count'] == 0 and any(b['features']['large_trade_count'] == 3 for b in S['svc']['BTCUSDT'].bars)
