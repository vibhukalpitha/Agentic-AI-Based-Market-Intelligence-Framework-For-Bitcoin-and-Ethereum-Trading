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


CACHE_DIR = os.environ.get('C03_CACHE', os.path.join('data_local', 'live_cache'))
ARCHIVE_BASE = os.environ.get('C03_ARCHIVE_BASE', 'https://data.binance.vision/data/spot')
MAX_CALLS_PER_MIN = int(os.environ.get('C03_BACKFILL_CALLS_PER_MIN', '600'))   # aggTrades costs weight 4 per call; 600 calls = 2,400 of Binance's 6,000 weight per minute
# Training definition (p_data_large_def.py, 2026-10-19): large_trade_count = number of RAW fills (every trade) of the minute with quantity >= LARGE[symbol]; matched the recorded values 100% for BTC and ETH.
# Exact sources: the daily raw-trade archive (complete UTC days) and the live @trade stream. Approximate source (flagged): aggTrades REST (counts aggregated trades; close to the raw count but not equal).
_RAW = {}


def _raw_path(sym): return os.path.join(CACHE_DIR, f'large_raw_{sym}.json')


def _load_raw(sym):
    if sym in _RAW: return _RAW[sym]
    c = {'threshold': LARGE[sym], 'source': 'raw', 'days': {}, 'live': {}}
    try:
        j = json.load(open(_raw_path(sym)))
        if j.get('threshold') == LARGE[sym] and j.get('source') == 'raw': c = j
    except Exception: pass
    _RAW[sym] = c; return c


def _save_raw(sym):
    c = _load_raw(sym)
    try:
        cut = int(time.time() * 1000) - 15 * 86400000
        c['live'] = {k: v for k, v in c['live'].items() if int(k) >= cut}
        c['days'] = dict(sorted(c['days'].items())[-15:])
        os.makedirs(CACHE_DIR, exist_ok=True); tmp = _raw_path(sym) + '.tmp'; json.dump(c, open(tmp, 'w')); os.replace(tmp, _raw_path(sym))
    except Exception: pass                      # the cache only saves time; never fatal


def _download(url, dest, tries=5):
    """resumable download of a public archive file (HTTP Range); FileNotFoundError when the file is not published (yet)"""
    import urllib.request, urllib.error
    for k in range(tries):
        try:
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, 'ab' if have and r.status == 206 else 'wb') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416: return
            if e.code == 404: raise FileNotFoundError(url)
            time.sleep(5 * (k + 1))
        except Exception: time.sleep(5 * (k + 1))
    raise RuntimeError(f'download failed: {url}')


def raw_day_counts(sym, day):
    """exact large_trade_count of every minute of one complete UTC day from the raw-trade archive (blocking; run in a thread). Returns {minute_ms(str): count} for minutes with at least one large trade."""
    import zipfile
    import pandas as pd
    os.makedirs(os.path.join(CACHE_DIR, 'tmp'), exist_ok=True); zp = os.path.join(CACHE_DIR, 'tmp', f'{sym}-trades-{day}.zip'); _download(f'{ARCHIVE_BASE}/daily/trades/{sym}/{sym}-trades-{day}.zip', zp)
    acc = {}; thr = LARGE[sym]
    with zipfile.ZipFile(zp) as z:
        with z.open([n for n in z.namelist() if n.endswith('.csv')][0]) as f:
            for ch in pd.read_csv(f, header=None, names=['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match'], usecols=['qty', 'time'], chunksize=4_000_000):
                t = ch['time'].to_numpy('int64'); t = np.where(t > 10 ** 14, t // 1000, t); m = t // 60000 * 60000; mask = ch['qty'].to_numpy('float64') >= thr
                if mask.any():
                    u, c = np.unique(m[mask], return_counts=True)
                    for a_, b_ in zip(u.tolist(), c.tolist()): acc[str(a_)] = acc.get(str(a_), 0) + b_
    os.remove(zp)
    if not acc: raise RuntimeError(f'{sym} {day}: no large trades found in the archive file (unexpected)')
    return acc


async def _count_range_agg(sym, a_ms, b_ms):
    """APPROXIMATE large counts from aggregated trades (public aggTrades REST, rate-limited): minute_ms -> count of aggregated trades with quantity >= LARGE"""
    import httpx
    thr = LARGE[sym]; gap = 60.0 / MAX_CALLS_PER_MIN; last_call = [0.0]; counts = {}
    async with httpx.AsyncClient(timeout=30) as cl:
        async def call(params):
            for attempt in range(8):
                wait = gap - (time.time() - last_call[0])
                if wait > 0: await asyncio.sleep(wait)
                last_call[0] = time.time()
                r = await cl.get(f'{REST_URL}/api/v3/aggTrades', params=params)
                if r.status_code in (418, 429): await asyncio.sleep(min(120, float(r.headers.get('Retry-After', 5)) + attempt * 5)); continue
                r.raise_for_status(); return r.json()
            raise RuntimeError('Binance rate limit: aggTrades backfill gave up after 8 tries')
        t = a_ms
        while t <= b_ms:
            w_end = min(t + 3600000 - 1, b_ms); params = {'symbol': sym, 'startTime': t, 'endTime': w_end, 'limit': 1000}
            while True:
                rows = await call(params)
                for x in rows:
                    if float(x['q']) >= thr: mm = (int(x['T']) // 60000) * 60000; counts[mm] = counts.get(mm, 0) + 1
                if len(rows) < 1000: break
                params = {'symbol': sym, 'fromId': int(rows[-1]['a']) + 1, 'limit': 1000}
                if int(rows[-1]['T']) > w_end: break
            t = w_end + 1
    return counts


async def fetch_large_counts(sym, start_ms, end_ms, progress=None):
    """large_trade_count for the minute-open times start_ms..end_ms (inclusive). Returns (counts {minute_ms: n}, n_approximate_minutes).
    Complete UTC days come EXACT from the raw-trade archive (cached per day); minutes the live stream counted completely are exact (cached); anything else (today so far, an unpublished archive day, a gap) is APPROXIMATED from aggTrades and counted in the second value."""
    c = _load_raw(sym); out = {}; need = []; today = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    first_day = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc).date(); last_day = dt.datetime.fromtimestamp(end_ms / 1000, dt.timezone.utc).date(); d = first_day
    while d <= last_day:
        key = str(d)
        if key < today and key not in c['days']:
            if progress: progress(f'{sym}: raw trade archive {key}')
            try: c['days'][key] = await asyncio.to_thread(raw_day_counts, sym, key); _save_raw(sym)
            except FileNotFoundError: pass          # not published yet -> approximated below
        d += dt.timedelta(days=1)
    for m in range(start_ms, end_ms + 1, 60000):
        key = dt.datetime.fromtimestamp(m / 1000, dt.timezone.utc).strftime('%Y-%m-%d')
        if key in c['days']: out[m] = int(c['days'][key].get(str(m), 0))
        elif str(m) in c['live']: out[m] = int(c['live'][str(m)])
        else: need.append(m)
    if need:                                      # contiguous runs of unknown minutes -> aggTrades (approximate)
        runs = []; s0 = prev = need[0]
        for m in need[1:]:
            if m != prev + 60000: runs.append((s0, prev)); s0 = m
            prev = m
        runs.append((s0, prev))
        for (x, y) in runs:
            if progress: progress(f'{sym}: aggTrades estimate up to {ms_iso(y)}')
            part = await _count_range_agg(sym, x, y + 59999)
            for m in range(x, y + 1, 60000): out[m] = part.get(m, 0)
    return out, len(need)


def _remember_live(sym, minute_ms, n):
    c = _load_raw(sym); c['live'][str(minute_ms)] = int(n)


BARS_DIR = os.environ.get('C03_LIVE_BARS', os.path.join('data_local', 'live_bars'))
BAR_COLS = ['open_ms', 'utc', 'open', 'high', 'low', 'close', 'volume', 'trades', 'taker_buy_base', 'large_count', 'large_exact', 'state', 'fused']


def persist_bar(sym, k, exact, state=None, fused=None):
    """LIVE-T: append one scored bar to data_local/live_bars/{SYM}_{UTC day}.csv (unlabelled real market data for burden/stability checks; never used for tuning). Never raises: saving must not stop scoring."""
    try:
        day = dt.datetime.fromtimestamp(k['open_ms'] / 1000, dt.timezone.utc).strftime('%Y-%m-%d'); os.makedirs(BARS_DIR, exist_ok=True); path = os.path.join(BARS_DIR, f'{sym}_{day}.csv'); new = not os.path.exists(path)
        row = [k['open_ms'], ms_iso(k['open_ms']) + 'Z', k['open'], k['high'], k['low'], k['close'], k['volume'], k['trades'], k['taker_buy_base'], k.get('large_count'), int(bool(exact)), state if state is not None else '', '' if fused is None else fused]
        with open(path, 'a', encoding='utf-8') as fh:
            if new: fh.write(','.join(BAR_COLS) + '\n')
            fh.write(','.join(str(x) for x in row) + '\n')
    except Exception as e: S_ERR[0] = 'live bar save failed: ' + repr(e)[:200]


S_ERR = [None]


def parse_kline(k):
    return {'open_ms': int(k['t']), 'open': float(k['o']), 'high': float(k['h']), 'low': float(k['l']), 'close': float(k['c']), 'volume': float(k['v']), 'trades': int(k['n']), 'taker_buy_base': float(k['V'])}


SEED_MINUTES = int(os.environ.get('C03_SEED_MINUTES', '180'))     # v0.2.2: the newest warm-up minutes are scored one by one (as a live minute would be) so the page has history at once; 0 = off


def prime_and_seed(service, rows, tss, ks_rows, seed=None):
    """prime the engine on all warm-up minutes but the last `seed`, then score those last minutes bar by bar with the SAME code path as a live minute (identical to the replay path).
    Nothing is written to the live_bars files and no event is sent: these are past minutes. rows: feature rows, tss: their timestamps, ks_rows: the Binance kline of each row (price for the chart)."""
    import pandas as pd
    seed = SEED_MINUTES if seed is None else seed
    n = len(rows); seed = max(0, min(int(seed), n - 1500))               # the engine needs >= 1,500 minutes to prime
    service.prime(pd.DataFrame(rows[:n - seed], columns=ORDER), tss[:n - seed])
    for f, ts, k in zip(rows[n - seed:], tss[n - seed:], ks_rows[n - seed:]):
        service.on_bar(np.asarray(f, 'float64'), ts, k)
    return seed


async def run_live(ex, S, hub, fresh):
    svc = fresh(); S['mode'] = 'live'; S['svc'] = {}; fb = {}; last_open = {}; B = ex.B; S['warmup'] = {'stage': 'starting', 'done': 0, 'total': len(ex.symbols), 'error': None}; S['warnings'] = []
    import pandas as pd
    for attempt in range(1000):                 # 1) warm-up: REST klines + the real large_trade_count history (aggTrades). A failure is shown in /health and retried, never silent.
        try:
            for i, sym in enumerate(ex.symbols):
                S['warmup'].update(stage=f'{sym}: klines', done=i, error=None); ks = await fetch_klines(sym, n=WARMUP_BARS)
                S['warmup']['stage'] = f'{sym}: large-trade history (first start downloads daily archive files and can take 10-25 minutes; saved for next time)'
                lc, napprox = await fetch_large_counts(sym, ks[0]['open_ms'], ks[-1]['open_ms'], lambda msg: S['warmup'].update(stage=msg)); S['warmup'].setdefault('approx_minutes', {})[sym] = napprox
                for k in ks: k['large_count'] = lc.get(k['open_ms'], 0)
                if sum(k['large_count'] for k in ks) == 0: raise RuntimeError(f'{sym}: large_trade_count history is all zero - the models would be scored on a wrong baseline')
                fb[sym] = FeatureBuilder(sym); rows, tss, krow = [], [], []
                for k in ks:
                    f = fb[sym].push(k)
                    if f is not None: rows.append(f); tss.append(ms_iso(k['open_ms'])); krow.append(k)
                S['warmup']['stage'] = f'{sym}: priming the engine'; await asyncio.to_thread(prime_and_seed, svc[sym], rows, tss, krow); last_open[sym] = ks[-1]['open_ms']
            break
        except asyncio.CancelledError: raise
        except Exception as e:
            S['warmup']['error'] = repr(e)[:300]; S['last_error'] = repr(e)[:300]; await asyncio.sleep(min(300, 10 * (attempt + 1)))
    S['svc'] = svc; S['warmup'].update(stage='done', done=len(ex.symbols), error=None)
    S['warmup_note'] = 'large_trade_count = raw fills with quantity >= %s (the training definition). History: exact from the raw-trade archive for complete days; today so far estimated from aggTrades (see warmup.approx_minutes); live minutes counted exactly from the trade stream.' % LARGE
    counts = defaultdict(int); last_msg = {'t': time.time()}; gap_sent = {'v': False}; conn = {'t': None}; saved = {'n': 0}

    async def handle(sym, k):
        lo = last_open[sym]
        if k['open_ms'] <= lo: return
        if k['open_ms'] > lo + 60000:           # gap: fill from REST so the engine never scores a bar built from incomplete data
            await hub.send({'type': 'data_gap', 'data': {'symbol': sym, 'since_utc': iso(ms_iso(lo + 60000))}})
            ms = await fetch_klines(sym, start_ms=lo + 60000, end_ms=k['open_ms'] - 60000)
            try: lc = (await fetch_large_counts(sym, ms[0]['open_ms'], ms[-1]['open_ms']))[0] if ms else {}
            except Exception as e: lc = {}; S['last_error'] = 'gap backfill of large_trade_count failed: ' + repr(e)[:200]
            for m in ms: m['_lc'] = lc.get(m['open_ms'], 0); await score(sym, m)
        await score(sym, k)

    async def score(sym, k):
        k = dict(k); n_stream = counts.pop((sym, k['open_ms']), 0); exact = False
        if '_lc' in k: k['large_count'] = k.pop('_lc')                                       # gap-filled minute: counted by fetch_large_counts
        elif conn['t'] is not None and conn['t'] <= k['open_ms']: k['large_count'] = n_stream; exact = True; _remember_live(sym, k['open_ms'], n_stream)       # the stream was connected for the whole minute: exact
        else:                                                                                  # connected after the minute began: the stream count is incomplete -> estimate this minute
            try: k['large_count'] = (await fetch_large_counts(sym, k['open_ms'], k['open_ms']))[0].get(k['open_ms'], n_stream)
            except Exception: k['large_count'] = n_stream
            S['approx_live_minutes'] = S.get('approx_live_minutes', 0) + 1
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
        status, events = await asyncio.to_thread(svc[sym].on_bar, np.asarray(f, 'float64'), ms_iso(k['open_ms']), k); last_open[sym] = k['open_ms']; S['last_bar'] = time.time(); saved['n'] += 1
        persist_bar(sym, k, exact, status if isinstance(status, str) else getattr(status, 'get', lambda *_: '')('state', ''))
        if S_ERR[0]: S['last_error'] = S_ERR[0]; S_ERR[0] = None
        if saved['n'] % 10 == 0:
            for s_ in ex.symbols: _save_raw(s_)
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
                    backoff = 1; conn['t'] = int(time.time() * 1000)
                    async for raw in ws:
                        last_msg['t'] = time.time(); gap_sent['v'] = False; d = json.loads(raw).get('data', {})
                        if d.get('e') == 'trade':
                            if float(d['q']) >= LARGE[d['s']]: counts[(d['s'], int(d['T']) // 60000 * 60000)] += 1
                        elif d.get('e') == 'kline' and d['k']['x']: await handle(d['s'], parse_kline(d['k']))
            except asyncio.CancelledError: raise
            except Exception as e:
                S['last_error'] = repr(e); await asyncio.sleep(backoff); backoff = min(60, backoff * 2)
    finally: wd.cancel()
