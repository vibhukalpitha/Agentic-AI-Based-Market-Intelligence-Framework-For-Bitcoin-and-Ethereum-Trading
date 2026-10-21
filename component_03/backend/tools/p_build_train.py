"""p_build_train.py - step 3 of the improvement round: genuine feature tables + injected TRAIN replicas (CPU).

Per symbol it writes (folder --out, NEW folder, nothing is overwritten unless --force):
  {SYM}_genuine_features_{from}_{to}.parquet   all genuine minutes --all-from .. --all-to (default 2023-01 .. 2024-07; use --only train now, --only genuine once July exists):
                                    ts + 6 base features (c03_engine formulas, agg-based large_trade_count) + 12 v3 features
  {SYM}_train_r{k}.parquet         injected replicas of the TRAIN months only (default 2023-01 .. 2024-06), same columns
                                    + label (0 clean, 1..6 family, -1 tail) + event_id + bar + replica
  {SYM}_train_events.csv, manifest_{SYM}.json (sha16 of every input file, of p_inj_train.py, p_feat3 SPEC_SHA16, seeds, counts)
Rules: injected families = OLD only (momentum_ignition / slow_accumulation never in training). No July 2024 minute is used
for injection (July = DEV). Event density = 600 events / 63 days of TRAIN-3 = 9.5 per day (--events-per-day).
Run (from backend):  python tools\\p_build_train.py --archive <archive_v1 folder> --out <new folder> --symbols BTCUSDT ETHUSDT
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd
try:
    from tools import p_inj_train as J, p_feat3 as F
except ImportError:                                  # run as a script from inside tools\
    import p_inj_train as J, p_feat3 as F

SEED = {'BTCUSDT': 2011, 'ETHUSDT': 2016}           # TRAIN injection seeds (never the p_l3 seeds 3101/3202/3303)


def sha16(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()[:16]


def months(a, b):
    return [str(p) for p in pd.period_range(a, b, freq='M')]


def find_month_file(folder, symbol, ym):
    """one parquet whose relative path contains the symbol and the month (2023-01 / 202301 / 2023_01). Clear error with a listing otherwise."""
    if not os.path.isdir(folder):
        raise SystemExit(f'folder not found: {folder} (is Drive mounted?)')
    allf = sorted(p for p in glob.glob(os.path.join(folder, '**', '*'), recursive=True) if os.path.isfile(p))
    rel = lambda p: os.path.relpath(p, folder)
    toks = [ym, ym.replace('-', ''), ym.replace('-', '_')]
    cand = [p for p in allf if p.lower().endswith('.parquet') and symbol.lower() in rel(p).lower() and any(t in rel(p) for t in toks)]
    if len(cand) != 1:
        raise SystemExit(f'expected exactly 1 parquet for {symbol} {ym} in {folder}, found {cand}. The folder has {len(allf)} files, first ones: '
                         + ', '.join(rel(p) for p in allf[:12]))
    return cand[0]


def clean_ts(s):
    """-> tz-naive UTC datetime64[ns]. Accepts tz-aware datetimes (the archive), naive datetimes, or epoch seconds/ms."""
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_datetime(s, unit='ms' if s.iloc[0] > 1e11 else 's', utc=True)
    else:
        s = pd.to_datetime(s, utc=True)
    return s.dt.tz_localize(None).astype('datetime64[ns]')


def load_tables(folder, symbol, ms):
    parts, files = [], {}
    for ym in ms:
        p = find_month_file(folder, symbol, ym); d = pd.read_parquet(p)
        if 'ts' not in d.columns:
            d = d.reset_index().rename(columns={d.index.name or 'index': 'ts'})
        miss = [c for c in J.REQ if c not in d.columns]
        if miss:
            raise SystemExit(f'{p} lacks columns {miss}')
        d = d[J.REQ].copy(); d['ts'] = clean_ts(d['ts']); parts.append(d); files[ym] = (os.path.basename(p), sha16(p))
    t = pd.concat(parts, ignore_index=True).sort_values('ts').reset_index(drop=True)
    assert t['ts'].is_unique, 'duplicate minutes across months'
    return t, files


def feature_table(t, symbol):
    """6 base + 12 v3 features, past-only. Rows with a v3 baseline < 30 minutes are NaN (dropped by the training scripts)."""
    b = J.base6(t, symbol); b.insert(0, 'ts', t['ts'].to_numpy()); f = F.features_v3(t)
    assert (f['ts'].to_numpy() == t['ts'].to_numpy()).all()
    return pd.concat([b, f.drop(columns='ts')], axis=1)


def build_symbol(archive, out, symbol, train_from, train_to, all_from, all_to, events_per_day, force=False, only='both'):
    """only: 'both' | 'train' (replicas, needs only the TRAIN months) | 'genuine' (feature table of all_from..all_to, e.g. incl. July 2024)."""
    t0 = time.time(); os.makedirs(out, exist_ok=True)
    gp = os.path.join(out, f'{symbol}_genuine_features_{all_from}_{all_to}.parquet'); mp = os.path.join(out, f'manifest_{symbol}.json')
    do_g, do_t = only in ('both', 'genuine'), only in ('both', 'train')
    if do_g and os.path.exists(gp) and not force:
        print('exists, skipped (use --force):', gp); do_g = False
    if do_t and os.path.exists(mp) and not force:
        print('exists, skipped (use --force):', mp); do_t = False
    if not (do_g or do_t):
        return
    allt, files = (None, {})
    if do_g:
        allt, files = load_tables(archive, symbol, months(all_from, all_to))
        gaps = int((np.diff(allt['ts'].to_numpy().astype('datetime64[m]').astype('int64')) != 1).sum())
        feature_table(allt, symbol).to_parquet(gp)
        json.dump({'symbol': symbol, 'inputs': files, 'p_feat3_spec_sha16': F.SPEC_SHA16, 'rows': int(len(allt)), 'timestamp_holes': gaps,
                   'note': 'genuine minutes only; large_trade_count is the AGG-based large_count'}, open(gp.replace('.parquet', '.manifest.json'), 'w'), indent=1)
        print('ok genuine', symbol, 'rows', len(allt), 'holes', gaps, flush=True)
    if not do_t:
        return
    lo, hi = pd.Timestamp(train_from + '-01'), (pd.Period(train_to, 'M') + 1).start_time
    if allt is not None:
        tr, tfiles = allt[(allt['ts'] >= lo) & (allt['ts'] < hi)].reset_index(drop=True), files
    else:
        tr, tfiles = load_tables(archive, symbol, months(train_from, train_to))
    gaps = int((np.diff(tr['ts'].to_numpy().astype('datetime64[m]').astype('int64')) != 1).sum())
    total = max(1, int(round(events_per_day * len(tr) / 1440.0))); info = {}; evs = []
    for f, ev in J.iter_set(tr, symbol, 0, len(tr), total, J.OLD, SEED[symbol]):
        ft = feature_table(f, symbol)
        for c in ('label', 'event_id', 'replica', 'bar', 'in_seg'):
            ft[c] = f[c].to_numpy()
        r = int(f['replica'].iloc[0]); ft.to_parquet(os.path.join(out, f'{symbol}_train_r{r}.parquet')); evs.append(ev)
        info[f'r{r}'] = {'events': int(len(ev)), 'rows': int(len(ft)), 'positive_bars': int((f['label'] > 0).sum())}
        print(symbol, 'replica', r, info[f'r{r}'], flush=True)
    ev = pd.concat(evs, ignore_index=True); assert not ev.family.isin(J.NEW).any(); ev.to_csv(os.path.join(out, f'{symbol}_train_events.csv'), index=False)
    man = {'symbol': symbol, 'inputs': tfiles, 'p_inj_train_sha16': sha16(J.__file__), 'p_feat3_spec_sha16': F.SPEC_SHA16, 'seed': SEED[symbol],
           'train_months': [train_from, train_to], 'events_per_day': events_per_day, 'events_total': int(len(ev)),
           'rows_train': int(len(tr)), 'timestamp_holes': gaps, 'replicas': info, 'seconds': round(time.time() - t0, 1),
           'note': 'large_trade_count is the AGG-based large_count (limitation); synthetic positives; July 2024 never injected'}
    json.dump(man, open(mp, 'w'), indent=1)
    print('ok train', symbol, 'events', len(ev), 'holes', gaps, f'{man["seconds"]}s')


def main():
    a = argparse.ArgumentParser(); a.add_argument('--archive', required=True); a.add_argument('--out', required=True)
    a.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT']); a.add_argument('--train-from', default='2023-01'); a.add_argument('--train-to', default='2024-06')
    a.add_argument('--all-from', default='2023-01'); a.add_argument('--all-to', default='2024-07'); a.add_argument('--events-per-day', type=float, default=9.5); a.add_argument('--force', action='store_true')
    a.add_argument('--only', choices=['both', 'train', 'genuine'], default='both')
    x = a.parse_args()
    for s in x.symbols:
        build_symbol(x.archive, x.out, s, x.train_from, x.train_to, x.all_from, x.all_to, x.events_per_day, x.force, x.only)


if __name__ == '__main__':
    main()
