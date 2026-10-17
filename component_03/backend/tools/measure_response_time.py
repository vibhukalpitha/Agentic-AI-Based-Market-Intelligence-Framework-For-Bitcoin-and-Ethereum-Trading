"""C03 IntegritySense: how fast is the engine? (measurement only; nothing is scored differently, nothing is shown in the dashboard)

Measures on THIS computer, with the real frozen export and the real recorded bars:
  1. start-up: loading the export (sha256 check of all files) and warming up one symbol (1,500 bars)
  2. engine time per closed bar (SymbolService.on_bar = checks + features + 3 models + fusion + state), median / p95 / p99 / max
  3. optional: the API round trip (GET /status) if the server is running (--api http://127.0.0.1:8000)
  4. optional: the live delay (data_age_sec from /health) if the server runs in live mode

Run (from component_03\\backend):
  $env:C03_EXPORT="...\\component_03\\backend\\engine_export_v1"
  python tools\\measure_response_time.py --bars 1000
  python tools\\measure_response_time.py --bars 1000 --api http://127.0.0.1:8000
Writes tools\\response_time_result.json so the numbers can be quoted in the viva.
"""
import argparse, json, os, platform, sys, time, urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
BAR_INTERVAL_MS = 60_000


def stats(ms):
    a = np.asarray(ms, 'float64')
    return {'n': int(a.size), 'median_ms': round(float(np.median(a)), 2), 'p95_ms': round(float(np.percentile(a, 95)), 2),
            'p99_ms': round(float(np.percentile(a, 99)), 2), 'max_ms': round(float(a.max()), 2)}


def get_json(url, timeout=10):
    t = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return (time.perf_counter() - t) * 1000.0, body


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--export', default=os.environ.get('C03_EXPORT'), help='engine_export_v1 folder (or set C03_EXPORT)')
    ap.add_argument('--symbol', default='BTCUSDT', choices=['BTCUSDT', 'ETHUSDT'])
    ap.add_argument('--bars', type=int, default=1000, help='how many bars to time (after 100 untimed bars to warm the caches)')
    ap.add_argument('--api', default=None, help='base URL of a running API, e.g. http://127.0.0.1:8000')
    a = ap.parse_args()
    if not a.export:
        sys.exit('Give the export folder: --export <path> or $env:C03_EXPORT')
    import pandas as pd
    from c03_engine.store import Export
    from c03_engine.service import SymbolService

    out = {'machine': {'cpu': platform.processor() or platform.machine(), 'cores': os.cpu_count(), 'python': platform.python_version(), 'os': platform.platform()},
           'symbol': a.symbol, 'bar_interval_ms': BAR_INTERVAL_MS}
    t = time.perf_counter(); ex = Export(a.export); out['load_export_sha256_check_s'] = round(time.perf_counter() - t, 2)
    rp = next(Path(a.export, 'replay').glob(f'{a.symbol}_replay_*.parquet'))
    df = pd.read_parquet(rp).reset_index(drop=True); feats = list(ex.B.FEATS)
    need = 1500 + 100 + a.bars
    if len(df) < need:
        sys.exit(f'The recorded file has {len(df)} bars, {need} are needed. Use a smaller --bars.')
    t = time.perf_counter(); s = SymbolService(ex, a.symbol); s.prime(df.iloc[:1500][feats], list(df.ts[:1500])); out['warmup_1500_bars_s'] = round(time.perf_counter() - t, 2)

    def one(k):
        t0 = time.perf_counter(); s.on_bar(df.loc[k, feats].to_numpy('float64'), str(df.ts[k]), {'close': float(df.close[k])}); return (time.perf_counter() - t0) * 1000.0

    for k in range(1500, 1600): one(k)                                   # untimed: first calls load libraries and fill caches
    ms = [one(k) for k in range(1600, 1600 + a.bars)]
    out['engine_per_bar'] = stats(ms)
    out['share_of_bar_interval_pct'] = round(100.0 * out['engine_per_bar']['p99_ms'] / BAR_INTERVAL_MS, 3)
    out['bars_per_second_possible'] = round(1000.0 / out['engine_per_bar']['median_ms'], 1)
    out['replay_jump_estimate_s_per_1000_bars'] = round(out['engine_per_bar']['median_ms'], 1)  # 1,000 bars x median ms / 1000

    if a.api:
        base = a.api.rstrip('/') + '/api/c03'
        try:
            lat = [get_json(base + '/status')[0] for _ in range(50)]
            out['api_status_round_trip'] = stats(lat)
            h = get_json(base + '/health')[1]
            out['api_health'] = {'mode': h.get('mode'), 'data_age_sec': h.get('data_age_sec'), 'status': h.get('status')}
        except Exception as e:                                           # the API is optional: say why it was not measured
            out['api_status_round_trip'] = f'not measured: {type(e).__name__}: {e}'

    (HERE / 'response_time_result.json').write_text(json.dumps(out, indent=2))
    e = out['engine_per_bar']
    print(f"\nMachine: {out['machine']['cores']} cores, Python {out['machine']['python']}")
    print(f"Start-up: export check {out['load_export_sha256_check_s']} s, warm-up (1,500 bars) {out['warmup_1500_bars_s']} s")
    print(f"Engine per closed bar ({e['n']} bars, {a.symbol}): median {e['median_ms']} ms | p95 {e['p95_ms']} ms | p99 {e['p99_ms']} ms | max {e['max_ms']} ms")
    print(f"A new bar arrives every {BAR_INTERVAL_MS:,} ms, so the engine uses {out['share_of_bar_interval_pct']} % of the time between bars (p99).")
    print(f"Replay fast-forward: about {out['replay_jump_estimate_s_per_1000_bars']} s per 1,000 bars per symbol.")
    if a.api:
        print('API /status round trip:', out['api_status_round_trip']); print('API health:', out.get('api_health'))
    print('Saved:', HERE / 'response_time_result.json')


if __name__ == '__main__':
    main()
