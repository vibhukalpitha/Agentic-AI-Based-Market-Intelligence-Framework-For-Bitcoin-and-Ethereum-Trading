"""Live source: Binance Spot PUBLIC market data (kline_1m + trade streams, REST klines for warm-up and gap fill). No keys, no orders, no private endpoints."""
import asyncio, json, os, time, datetime as dt
from collections import defaultdict
import numpy as np, websockets
from .features import FeatureBuilder, LARGE, ORDER
from .service import iso
from .validate import check_bar, problem

WS_URL = os.environ.get('C03_BINANCE_WS', 'wss://stream.binance.com:9443/stream')
REST_URL = os.environ.get('C03_BINANCE_REST', 'https://api.binance.com')
WARMUP_BARS = 10080 + 60


def ms_iso(ms): return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


async def fetch_klines(sym, start_ms=None, end_ms=None, n=None):
    """closed 1-minute klines, oldest first, as dicts. n = number of most recent closed bars (paged)."""
    import httpx
    out = []; now = int(time.time() * 1000)
    async with httpx.AsyncClient(timeout=20) as cl:
        if n:
            end = (now // 60000) * 60000 - 1
            while len(out) < n:
                r = await cl.get(f'{REST_URL}/api/v3/klines', params={'symbol': sym, 'interval': '1m', 'limit': 1000, 'endTime': end}); r.raise_for_status(); k = r.json()
                if not k: break
                out = k + out; end = k[0][0] - 1
            out = out[-n:]
        else:
            t = start_ms
            while t <= end_ms:
                r = await cl.get(f'{REST_URL}/api/v3/klines', params={'symbol': sym, 'interval': '1m', 'limit': 1000, 'startTime': t, 'endTime': end_ms}); r.raise_for_status(); k = r.json()
                if not k: break
                out += k; t = k[-1][0] + 60000
    return [{'open_ms': int(x[0]), 'open': float(x[1]), 'high': float(x[2]), 'low': float(x[3]), 'close': float(x[4]), 'volume': float(x[5]), 'trades': int(x[8]), 'taker_buy_base': float(x[9]), 'large_count': 0} for x in out]


def parse_kline(k):
    return {'open_ms': int(k['t']), 'open': float(k['o']), 'high': float(k['h']), 'low': float(k['l']), 'close': float(k['c']), 'volume': float(k['v']), 'trades': int(k['n']), 'taker_buy_base': float(k['V'])}


async def run_live(ex, S, hub, fresh):
    svc = fresh(); S['svc'] = svc; S['mode'] = 'live'; fb = {}; last_open = {}; B = ex.B
    for sym in ex.symbols:                      # 1) warm-up from REST (large_trade_count history is not in klines: 0 until the live trade stream fills it)
        ks = await fetch_klines(sym, n=WARMUP_BARS); fb[sym] = FeatureBuilder(sym); rows, tss = [], []
        for k in ks:
            f = fb[sym].push(k)
            if f is not None: rows.append(f); tss.append(ms_iso(k['open_ms']))
        import pandas as pd
        await asyncio.to_thread(svc[sym].prime, pd.DataFrame(rows, columns=ORDER), tss); last_open[sym] = ks[-1]['open_ms']
    S['warmup_note'] = 'large_trade_count history before start-up is unavailable from klines and was set to 0 (live counts from the trade stream only)'
    counts = defaultdict(int); last_msg = {'t': time.time()}; gap_sent = {'v': False}

    async def handle(sym, k):
        lo = last_open[sym]
        if k['open_ms'] <= lo: return
        if k['open_ms'] > lo + 60000:           # gap: fill from REST so the engine never scores a bar built from incomplete data
            await hub.send({'type': 'data_gap', 'data': {'symbol': sym, 'since_utc': iso(ms_iso(lo + 60000))}})
            for m in await fetch_klines(sym, start_ms=lo + 60000, end_ms=k['open_ms'] - 60000): await score(sym, m)
        await score(sym, k)

    async def score(sym, k):
        k = dict(k); k['large_count'] = counts.pop((sym, k['open_ms']), k.get('large_count', 0))
        bad = check_bar(k)                      # a bar that fails the checks is never scored and never shown as NORMAL; the 15-bar history is re-built from clean bars
        if bad:
            fb[sym] = FeatureBuilder(sym); last_open[sym] = k['open_ms']
            for e in (await asyncio.to_thread(svc[sym].reject, ms_iso(k['open_ms']), bad, None, k))[1]: await hub.send(e)
            return
        f = fb[sym].push(k)
        if f is None:
            last_open[sym] = k['open_ms']
            for e in (await asyncio.to_thread(svc[sym].reject, ms_iso(k['open_ms']), [problem('rebuilding_features')], None, k))[1]: await hub.send(e)
            return
        status, events = await asyncio.to_thread(svc[sym].on_bar, np.asarray(f, 'float64'), ms_iso(k['open_ms']), k); last_open[sym] = k['open_ms']; S['last_bar'] = time.time()
        for e in events: await hub.send(e)

    async def watchdog():
        while True:
            await asyncio.sleep(15)
            if time.time() - last_msg['t'] > 90 and not gap_sent['v']:
                gap_sent['v'] = True
                for sym in ex.symbols: await hub.send({'type': 'data_gap', 'data': {'symbol': sym, 'since_utc': iso(ms_iso(last_open[sym] + 60000))}})
    wd = asyncio.create_task(watchdog()); backoff = 1
    streams = '/'.join(f'{s.lower()}@kline_1m/{s.lower()}@trade' for s in ex.symbols)
    try:
        while True:
            try:
                async with websockets.connect(f'{WS_URL}?streams={streams}', ping_interval=20) as ws:
                    backoff = 1
                    async for raw in ws:
                        last_msg['t'] = time.time(); gap_sent['v'] = False; d = json.loads(raw).get('data', {})
                        if d.get('e') == 'trade':
                            if float(d['q']) >= LARGE[d['s']]: counts[(d['s'], int(d['T']) // 60000 * 60000)] += 1
                        elif d.get('e') == 'kline' and d['k']['x']: await handle(d['s'], parse_kline(d['k']))
            except asyncio.CancelledError: raise
            except Exception as e:
                S['last_error'] = repr(e); await asyncio.sleep(backoff); backoff = min(60, backoff * 2)
    finally: wd.cancel()
