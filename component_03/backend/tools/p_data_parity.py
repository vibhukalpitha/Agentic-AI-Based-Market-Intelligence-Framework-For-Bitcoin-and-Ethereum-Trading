"""Definition check (not a model, not training): do the Layer-1 features the engine was trained on equal the features computed from Binance aggTrades?
Reads the minute table written by p_data_archive.py for 2024-08-01.. (daily files) and compares it with the recorded replay bars of the export (same minutes). The engine's live large_trade_count (aggTrades, quantity >= 0.1 BTC / 2.0 ETH)
is trusted only if this check passes. Writes parity_result.json.
Run:  python p_data_archive.py --days 2024-08-01 2024-08-02 --out D:\\data\\parity_days   then   python p_data_parity.py --minutes D:\\data\\parity_days --export ..\\engine_export_v1
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from c03_engine.features import FeatureBuilder


def layer1(df, sym):
    fb = FeatureBuilder(sym); rows = []
    for r in df.itertuples():
        f = fb.push({'close': r.close, 'high': r.high, 'low': r.low, 'volume': r.volume, 'trades': r.n_trades, 'taker_buy_base': r.taker_buy_base, 'large_count': r.large_count}); rows.append(f if f is not None else [np.nan] * 6)
    return pd.DataFrame(rows, columns=['log_returns', 'trade_flow_imbalance', 'hl_spread_ratio', 'rush_order_density', 'volume_acceleration', 'large_trade_count'], index=df.ts.values)


def compare(a, b):
    j = a.join(b, how='inner', lsuffix='_arch', rsuffix='_rec'); out = {'minutes_compared': int(len(j))}
    for c in ['log_returns', 'trade_flow_imbalance', 'hl_spread_ratio', 'rush_order_density', 'volume_acceleration', 'large_trade_count']:
        x, y = j[c + '_arch'].to_numpy(float), j[c + '_rec'].to_numpy(float); ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10: out[c] = None; continue
        x, y = x[ok], y[ok]; out[c] = {'corr': float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None, 'share_equal_1e-6': float(np.mean(np.abs(x - y) <= 1e-6 + 1e-6 * np.abs(y))), 'median_ratio_arch_over_rec': float(np.median(x[y != 0] / y[y != 0])) if (y != 0).any() else None}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--minutes', required=True); ap.add_argument('--export', required=True); a = ap.parse_args(); res = {}
    for sym in ('BTCUSDT', 'ETHUSDT'):
        fs = sorted(glob.glob(os.path.join(a.minutes, sym, f'{sym}_*_minute.parquet')))
        if not fs: continue
        arch = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).sort_values('ts'); arch['ts'] = pd.to_datetime(arch['ts'], utc=True)
        rec = pd.read_parquet(os.path.join(a.export, 'replay', f'{sym}_replay_2024-08-01_to_08-10.parquet')); rec['ts'] = pd.to_datetime(rec['ts'], utc=True); rec = rec.set_index('ts')
        res[sym] = compare(layer1(arch, sym).set_index(pd.DatetimeIndex(arch.ts)), rec[['log_returns', 'trade_flow_imbalance', 'hl_spread_ratio', 'rush_order_density', 'volume_acceleration', 'large_trade_count']])
    json.dump(res, open('parity_result.json', 'w'), indent=1); print(json.dumps(res, indent=1))
