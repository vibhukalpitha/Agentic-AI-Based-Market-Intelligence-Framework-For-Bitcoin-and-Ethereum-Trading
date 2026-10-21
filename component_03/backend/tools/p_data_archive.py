"""P-DATA (protocol pack v1, approved 2026-10-19): Binance public archive (data.binance.vision) aggTrades -> one row per minute.

Streams monthly (or daily) aggTrades zips, never keeps the raw trades: it writes one small parquet per symbol-month with the per-minute statistics below. The statistics are a fixed superset, declared BEFORE any result:
  Layer-1 inputs : open, high, low, close, volume, taker_buy_base, n_trades (= sum(last_id - first_id + 1), the same number a kline reports as 'trades'), large_count (aggregated trades with quantity >= LARGE[symbol])
  P-FEAT3 inputs : n_agg, q_p50, q_p90, q_p99, q_max (quantiles of aggregated-trade quantity), same_size_max (largest number of aggregated trades with identical quantity),
                   same_size_prev (aggregated trades whose quantity equals the previous trade's), dup_share ((n_agg - n_unique_qty) / n_agg), buy_run_max / sell_run_max (longest run of consecutive
                   aggregated trades on one side), n_runs, sec_active (seconds with a trade), sec_max_agg (most aggregated trades in one second), large_qty (sum of quantity of large trades)
Resumable: a finished symbol-month is skipped. A month that fails is reported and the others continue.
Run:  python p_data_archive.py --symbols BTCUSDT ETHUSDT --months 2023-01 2024-06 --out D:\\data\\archive_v1      (works on a laptop or on Colab; no keys, public files only)
"""
import argparse, hashlib, io, json, os, sys, time, zipfile, urllib.request
import numpy as np, pandas as pd

LARGE = {'BTCUSDT': 0.1, 'ETHUSDT': 2.0}
BASE = os.environ.get('C03_ARCHIVE_BASE', 'https://data.binance.vision/data/spot')
COLS = ['agg_id', 'price', 'qty', 'first_id', 'last_id', 'time', 'is_buyer_maker', 'is_best_match']
OUT_COLS = ['ts', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'n_trades', 'n_agg', 'large_count', 'large_qty', 'q_p50', 'q_p90', 'q_p99', 'q_max',
            'same_size_max', 'same_size_prev', 'dup_share', 'buy_run_max', 'sell_run_max', 'n_runs', 'sec_active', 'sec_max_agg']


def minute_stats(df, sym):
    """df: aggregated trades of whole minutes (columns COLS, time in ms, sorted by agg_id). Returns one row per minute."""
    thr = LARGE[sym]; t = df['time'].to_numpy('int64'); mi = t // 60000; q = df['qty'].to_numpy('float64'); p = df['price'].to_numpy('float64')
    maker = df['is_buyer_maker'].to_numpy(bool); buy = ~maker                # taker buy = buyer is not the maker
    nt = (df['last_id'].to_numpy('int64') - df['first_id'].to_numpy('int64') + 1)
    qr = np.round(q, 8); side = buy.astype('int8')
    work = pd.DataFrame({'m': mi, 'p': p, 'q': q, 'qr': qr, 'buy': buy, 'nt': nt, 'sec': t // 1000, 'large': q >= thr, 'side': side})
    work['prev_same_q'] = (work['qr'] == work['qr'].shift(1)) & (work['m'] == work['m'].shift(1))
    work['new_run'] = (work['side'] != work['side'].shift(1)) | (work['m'] != work['m'].shift(1))
    work['run_id'] = work['new_run'].cumsum()
    runs = work.groupby('run_id').agg(m=('m', 'first'), side=('side', 'first'), n=('q', 'size'))
    rb = runs[runs.side == 1].groupby('m')['n'].max(); rs = runs[runs.side == 0].groupby('m')['n'].max(); nr = runs.groupby('m').size()
    g = work.groupby('m')
    out = pd.DataFrame({'open': g['p'].first(), 'high': g['p'].max(), 'low': g['p'].min(), 'close': g['p'].last(), 'volume': g['q'].sum(), 'n_trades': g['nt'].sum(), 'n_agg': g.size(),
                        'q_p50': g['q'].quantile(.5), 'q_p90': g['q'].quantile(.9), 'q_p99': g['q'].quantile(.99), 'q_max': g['q'].max(), 'same_size_prev': g['prev_same_q'].sum(),
                        'sec_active': g['sec'].nunique()})
    out['taker_buy_base'] = work[work.buy].groupby('m')['q'].sum(); out['taker_buy_base'] = out['taker_buy_base'].fillna(0.0)
    out['large_count'] = work[work.large].groupby('m').size(); out['large_count'] = out['large_count'].fillna(0).astype('int64')
    out['large_qty'] = work[work.large].groupby('m')['q'].sum(); out['large_qty'] = out['large_qty'].fillna(0.0)
    out['same_size_max'] = work.groupby(['m', 'qr']).size().groupby('m').max()
    out['dup_share'] = 1.0 - work.groupby('m')['qr'].nunique() / out['n_agg']
    out['buy_run_max'] = rb; out['buy_run_max'] = out['buy_run_max'].fillna(0).astype('int64'); out['sell_run_max'] = rs; out['sell_run_max'] = out['sell_run_max'].fillna(0).astype('int64'); out['n_runs'] = nr
    out['sec_max_agg'] = work.groupby(['m', 'sec']).size().groupby('m').max()
    out = out.reset_index().rename(columns={'m': 'minute'}); out['ts'] = pd.to_datetime(out['minute'] * 60000, unit='ms', utc=True); out = out.drop(columns='minute')
    return out[OUT_COLS]


def stream_minutes(fileobj, sym, chunk_rows=3_000_000):
    """read a CSV of aggregated trades in chunks; a minute is only processed when complete (the last, possibly cut, minute is carried into the next chunk)"""
    carry = None; frames = []
    for ch in pd.read_csv(fileobj, header=None, names=COLS, usecols=range(8), chunksize=chunk_rows, dtype={'is_buyer_maker': 'object'}):
        ch['is_buyer_maker'] = ch['is_buyer_maker'].astype(str).str.lower().isin(('true', '1'))
        if ch['time'].iloc[0] > 10 ** 14: ch['time'] = ch['time'] // 1000      # microseconds (spot archive from 2025) -> ms
        if carry is not None: ch = pd.concat([carry, ch], ignore_index=True)
        last_min = int(ch['time'].iloc[-1] // 60000); done = ch[ch['time'] // 60000 < last_min]; carry = ch[ch['time'] // 60000 >= last_min]
        if len(done): frames.append(minute_stats(done, sym))
    if carry is not None and len(carry): frames.append(minute_stats(carry, sym))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUT_COLS)


def download(url, dest, tries=5):
    """resumable download (HTTP Range) of a public archive file"""
    for k in range(tries):
        try:
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, 'ab' if have and r.status == 206 else 'wb') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 416: return True            # already complete
            if e.code == 404: raise FileNotFoundError(url)
            time.sleep(5 * (k + 1))
        except Exception: time.sleep(5 * (k + 1))
    raise RuntimeError(f'download failed: {url}')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def process_zip(zip_path, sym):
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith('.csv')][0]
        with z.open(name) as f: return stream_minutes(f, sym)


def month_list(a, b):
    r = pd.period_range(a, b, freq='M'); return [str(x) for x in r]


def run(symbols, months, out, daily=None, keep_zip=False, tmp=None):
    os.makedirs(out, exist_ok=True); manifest_p = os.path.join(out, 'manifest.json')
    man = json.load(open(manifest_p)) if os.path.exists(manifest_p) else {'files': {}}
    items = [(s, m, 'monthly') for s in symbols for m in months] if not daily else [(s, d, 'daily') for s in symbols for d in daily]
    fails = []
    for sym, key, kind in items:
        od = os.path.join(out, sym); os.makedirs(od, exist_ok=True); dest = os.path.join(od, f'{sym}_{key}_minute.parquet')
        if os.path.exists(dest): print('skip', dest); continue
        url = f'{BASE}/{kind}/aggTrades/{sym}/{sym}-aggTrades-{key}.zip'; os.makedirs(tmp or od, exist_ok=True); zp = os.path.join(tmp or od, f'{sym}-aggTrades-{key}.zip')
        t0 = time.time()
        try:
            download(url, zp); df = process_zip(zp, sym)
            df.to_parquet(dest + '.tmp', index=False); os.replace(dest + '.tmp', dest)
            man['files'][f'{sym}/{key}'] = {'rows': int(len(df)), 'sha256': sha256_file(dest), 'source_url': url, 'seconds': round(time.time() - t0, 1), 'first': str(df.ts.iloc[0]) if len(df) else None, 'last': str(df.ts.iloc[-1]) if len(df) else None}
            json.dump(man, open(manifest_p, 'w'), indent=1)
            if not keep_zip: os.remove(zp)
            print(f'ok {sym} {key}: {len(df):,} minutes in {time.time() - t0:.0f}s', flush=True)
        except Exception as e:
            fails.append((sym, key, repr(e))); print('FAILED', sym, key, repr(e), flush=True)
    if fails: json.dump(fails, open(os.path.join(out, 'failed.json'), 'w'), indent=1)
    return fails


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT']); ap.add_argument('--months', nargs=2, default=['2023-01', '2024-06'])
    ap.add_argument('--days', nargs='*', help='daily files instead of monthly, e.g. 2024-08-01 2024-08-02 (used by the definition check)'); ap.add_argument('--out', required=True); ap.add_argument('--keep-zip', action='store_true'); ap.add_argument('--tmp', help='folder for the downloaded zip files (on Colab use /content/tmp so the Drive is not filled with multi-GB zips)')
    a = ap.parse_args(); f = run(a.symbols, month_list(*a.months), a.out, a.days, a.keep_zip, a.tmp); sys.exit(1 if f else 0)
