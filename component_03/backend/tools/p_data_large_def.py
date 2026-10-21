"""Which rule produced the training large_trade_count? (definition check, no model, no training)
p_data_parity.py showed 5 of 6 Layer-1 features match the Binance archive exactly, but large_trade_count only correlates (0.988) - the rule differs slightly.
This script counts, per minute, large trades under several candidate rules from BOTH archive files of one day (aggTrades and raw trades) and reports how often each rule equals the recorded value of the export:
  source: agg (aggregated trades) or raw (every fill) ; threshold = LARGE x {0.5, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0} with >= ; and '>' at 1.0.
Run:  python p_data_large_def.py --days 2024-08-01 2024-08-02 --out D:\\data\\large_def --export ..\\engine_export_v1
"""
import argparse, json, os, sys, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p_data_archive as P

MULTS = [0.5, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
AGG_COLS = P.COLS; RAW_COLS = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']


def variants():
    return [(f'>=x{m}', m, 'ge') for m in MULTS] + [('>x1.0', 1.0, 'gt')]


def count_zip(zip_path, sym, cols, chunk_rows=4_000_000):
    thr = P.LARGE[sym]; acc = {v[0]: {} for v in variants()}
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith('.csv')][0]
        with z.open(name) as f:
            for ch in pd.read_csv(f, header=None, names=cols, usecols=['qty', 'time'], chunksize=chunk_rows):
                t = ch['time'].to_numpy('int64'); t = np.where(t > 10 ** 14, t // 1000, t); m = t // 60000; q = ch['qty'].to_numpy('float64')
                for key, mult, op in variants():
                    mask = (q >= thr * mult) if op == 'ge' else (q > thr * mult)
                    if mask.any():
                        u, c = np.unique(m[mask], return_counts=True); d = acc[key]
                        for a, b in zip(u.tolist(), c.tolist()): d[a] = d.get(a, 0) + b
    return {k: pd.Series(v, dtype='int64') for k, v in acc.items()}


def compare(counts, rec):
    """counts: {variant: Series(minute index -> count)}, rec: Series indexed by UTC minute (recorded large_trade_count)"""
    out = {}; idx = (rec.index.astype('int64') // 10 ** 9 // 60).to_numpy()
    for k, s in counts.items():
        x = s.reindex(idx).fillna(0).to_numpy(float); y = rec.to_numpy(float)
        out[k] = {'share_equal': float(np.mean(x == y)), 'share_within_1': float(np.mean(np.abs(x - y) <= 1)), 'corr': float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else None, 'median_ratio': float(np.median(x[y > 0] / y[y > 0]))}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--days', nargs='+', required=True); ap.add_argument('--out', required=True); ap.add_argument('--export', required=True); ap.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT']); a = ap.parse_args()
    res = {}; os.makedirs(a.out, exist_ok=True)
    for sym in a.symbols:
        rec = pd.read_parquet(os.path.join(a.export, 'replay', f'{sym}_replay_2024-08-01_to_08-10.parquet')); rec['ts'] = pd.to_datetime(rec['ts'], utc=True); rec = rec.set_index('ts')['large_trade_count']
        tot = {}
        for src, cols, kind in (('agg', AGG_COLS, 'aggTrades'), ('raw', RAW_COLS, 'trades')):
            parts = {}
            for d in a.days:
                zp = os.path.join(a.out, f'{sym}-{kind}-{d}.zip'); P.download(f'{P.BASE}/daily/{kind}/{sym}/{sym}-{kind}-{d}.zip', zp)
                for k, s in count_zip(zp, sym, cols).items(): parts.setdefault(k, []).append(s)
                os.remove(zp)
            tot[src] = {k: pd.concat(v).groupby(level=0).sum() for k, v in parts.items()}
        day_idx = rec.index[[str(i.date()) in a.days for i in rec.index]]
        res[sym] = {src: compare(c, rec.loc[day_idx]) for src, c in tot.items()}
        best = max(((src, k, v['share_equal']) for src, c in res[sym].items() for k, v in c.items()), key=lambda x: x[2]); res[sym]['best'] = {'source': best[0], 'rule': best[1], 'share_equal': best[2]}
        print(sym, 'best rule:', res[sym]['best'], flush=True)
    json.dump(res, open('large_def_result.json', 'w'), indent=1); print(json.dumps(res, indent=1))
