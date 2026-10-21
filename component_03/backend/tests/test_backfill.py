"""large_trade_count history: raw archive day (exact), aggTrades estimate (paging, 429 retry), and the mix with the cache."""
import asyncio, os, sys, tempfile, zipfile, datetime as dt, httpx, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_agg_estimate_paging_and_429(monkeypatch):
    from c03_engine import live
    live.MAX_CALLS_PER_MIN = 60000; calls = {'n': 0, 'hit429': False}
    T0 = 1_700_000_040_000
    trades = [{'a': i, 'q': '0.2' if i % 5 == 0 else '0.01', 'T': T0 + i * 10} for i in range(2500)]
    def handler(req):
        calls['n'] += 1
        if calls['n'] == 2 and not calls['hit429']: calls['hit429'] = True; return httpx.Response(429, headers={'Retry-After': '0'})
        q = dict(req.url.params); out = []
        for t in trades:
            if 'fromId' in q and t['a'] < int(q['fromId']): continue
            if 'startTime' in q and not (int(q['startTime']) <= t['T'] <= int(q['endTime'])): continue
            out.append(t)
            if len(out) == int(q['limit']): break
        return httpx.Response(200, json=out)
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    r = asyncio.run(live._count_range_agg('BTCUSDT', T0, T0 + 60000))
    assert r.get(T0, 0) == 500 and calls['hit429']


def test_raw_day_and_mix(tmp_path, monkeypatch):
    from c03_engine import live
    live.CACHE_DIR = str(tmp_path); live._RAW.clear()
    day = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2)).strftime('%Y-%m-%d'); d0 = int(dt.datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    rows = [(i, 1.0, q, q, d0 + 5 * 60000 + i * 100, True, True) for i, q in enumerate([0.05, 0.1, 0.1, 0.2, 0.09, 0.5])]
    def fake_dl(url, dest):
        assert f'trades/BTCUSDT/BTCUSDT-trades-{day}.zip' in url
        with zipfile.ZipFile(dest, 'w') as z: z.writestr('x.csv', pd.DataFrame(rows).to_csv(header=False, index=False))
    monkeypatch.setattr(live, '_download', fake_dl)
    r = live.raw_day_counts('BTCUSDT', day); assert r == {str(d0 + 5 * 60000): 4}       # raw fills >= 0.1 (>= includes the two of exactly 0.1)
    live._RAW.clear(); live.CACHE_DIR = str(tmp_path)
    async def fake_agg(sym, a, b): return {m: 9 for m in range(a, b + 1, 60000)}
    monkeypatch.setattr(live, '_count_range_agg', fake_agg)
    now_min = int(dt.datetime.now(dt.timezone.utc).timestamp() // 60 * 60000); live._remember_live('BTCUSDT', now_min - 60000, 3)
    out, napprox = asyncio.run(live.fetch_large_counts('BTCUSDT', d0 + 4 * 60000, d0 + 6 * 60000))
    assert out == {d0 + 4 * 60000: 0, d0 + 5 * 60000: 4, d0 + 6 * 60000: 0} and napprox == 0
    out, napprox = asyncio.run(live.fetch_large_counts('BTCUSDT', now_min - 120000, now_min - 60000)); assert out[now_min - 60000] == 3 and out[now_min - 120000] == 9 and napprox == 1
