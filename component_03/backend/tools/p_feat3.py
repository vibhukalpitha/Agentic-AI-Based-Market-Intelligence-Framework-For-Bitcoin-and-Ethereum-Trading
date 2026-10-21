"""P-FEAT3 features v3 (protocol pack v1, approved 2026-10-19). FORMULAS DECLARED HERE BEFORE ANY RESULT IS SEEN; changing them after a result has been looked at is a protocol violation (new version instead).
Input: the per-minute table written by p_data_archive.py (columns ts, open, high, low, close, volume, taker_buy_base, n_trades, n_agg, large_count, large_qty, q_p50, q_p90, q_p99, q_max,
same_size_max, same_size_prev, dup_share, buy_run_max, sell_run_max, n_runs, sec_active, sec_max_agg), ONE symbol, sorted by ts. Output: 12 features per minute (past-only), appended to the 6 Layer-1 features by the training scripts.
eps = 1e-8. 'base60(x)' = median of x over the 60 closed minutes BEFORE the current minute (current minute excluded, shifted by one), needs >= 30 values, otherwise NaN (the row is dropped by the training scripts).
 1 size_skew            = ln((q_p99 + eps) / (q_p50 + eps))                          trade-size distribution: tail vs typical trade
 2 size_tail            = ln((q_max + eps) / (q_p90 + eps))                          trade-size distribution: single largest trade vs upper decile
 3 large_qty_share      = large_qty / (volume + eps)                                 share of volume traded in large trades
 4 same_size_share      = same_size_max / max(n_agg, 1)                              largest group of identical-size trades
 5 same_size_prev_share = same_size_prev / max(n_agg, 1)                             trades repeating the previous trade's size
 6 dup_share            = dup_share                                                   share of non-unique sizes (given)
 7 run_imbalance        = (buy_run_max - sell_run_max) / (buy_run_max + sell_run_max + 1)   side asymmetry of the longest one-sided runs
 8 mean_run_len         = ln(max(n_agg, 1) / max(n_runs, 1))                         average length of one-sided runs (log)
 9 volume_burst         = ln((volume + eps) / (base60(volume) + eps))                burst ratio of volume against its trailing baseline
10 trade_burst          = ln((n_agg + 1) / (base60(n_agg) + 1))                      burst ratio of trade count against its trailing baseline
11 second_concentration = sec_max_agg / max(n_agg, 1)                                busiest second's share of the minute's trades (self-similar / bursty proxy)
12 second_coverage      = sec_active / 60                                            share of the 60 seconds with any trade (self-similar / bursty proxy)
Note: the archive's large_count/large_qty/n_agg are aggTrades-based (not the raw-fill definition of the frozen large_trade_count); features 3 and 10 inherit that; the frozen Layer-1 feature large_trade_count is NOT replaced here.
"""
import hashlib, sys
import numpy as np, pandas as pd

EPS = 1e-8
FEATURES_V3 = ['size_skew', 'size_tail', 'large_qty_share', 'same_size_share', 'same_size_prev_share', 'dup_share', 'run_imbalance', 'mean_run_len', 'volume_burst', 'trade_burst', 'second_concentration', 'second_coverage']
NEED = ['ts', 'volume', 'n_agg', 'large_qty', 'q_p50', 'q_p90', 'q_p99', 'q_max', 'same_size_max', 'same_size_prev', 'dup_share', 'buy_run_max', 'sell_run_max', 'n_runs', 'sec_active', 'sec_max_agg']
SPEC_SHA16 = hashlib.sha256((__doc__ or '').encode('utf-8')).hexdigest()[:16]      # logged with every output; changes if the declared text changes


def _base60(s):
    return s.shift(1).rolling(60, min_periods=30).median()


def features_v3(df):
    """df: one symbol, sorted by ts. Returns a DataFrame [ts] + FEATURES_V3 (NaN where the baseline has < 30 past minutes)."""
    miss = [c for c in NEED if c not in df.columns]
    if miss: raise ValueError('missing columns: %s' % miss)
    d = df.sort_values('ts').reset_index(drop=True); na = d['n_agg'].clip(lower=1).astype('float64'); out = pd.DataFrame({'ts': d['ts']})
    out['size_skew'] = np.log((d['q_p99'] + EPS) / (d['q_p50'] + EPS)); out['size_tail'] = np.log((d['q_max'] + EPS) / (d['q_p90'] + EPS))
    out['large_qty_share'] = d['large_qty'] / (d['volume'] + EPS); out['same_size_share'] = d['same_size_max'] / na; out['same_size_prev_share'] = d['same_size_prev'] / na; out['dup_share'] = d['dup_share'].astype('float64')
    br, sr = d['buy_run_max'].astype('float64'), d['sell_run_max'].astype('float64'); out['run_imbalance'] = (br - sr) / (br + sr + 1.0)
    out['mean_run_len'] = np.log(na / d['n_runs'].clip(lower=1).astype('float64'))
    out['volume_burst'] = np.log((d['volume'] + EPS) / (_base60(d['volume']) + EPS)); out['trade_burst'] = np.log((d['n_agg'] + 1.0) / (_base60(d['n_agg'].astype('float64')) + 1.0))
    out['second_concentration'] = d['sec_max_agg'] / na; out['second_coverage'] = d['sec_active'] / 60.0
    return out[['ts'] + FEATURES_V3]


if __name__ == '__main__':          # python p_feat3.py in.parquet out.parquet   (one symbol-month or a concatenation of consecutive months)
    if len(sys.argv) != 3: raise SystemExit('usage: p_feat3.py in.parquet out.parquet')
    f = features_v3(pd.read_parquet(sys.argv[1])); f.to_parquet(sys.argv[2]); print('rows', len(f), 'spec_sha16', SPEC_SHA16, 'nan_rows', int(f[FEATURES_V3].isna().any(axis=1).sum()))
