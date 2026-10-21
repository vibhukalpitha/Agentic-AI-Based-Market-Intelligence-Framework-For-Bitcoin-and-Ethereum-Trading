"""P-L3: Layer 3 synthetic generator (protocol pack v1). SYNTHETIC, labelled by construction; the mismatch between this generator and real markets is unknown.
Different code and seeds from the Layer-1 injector. It edits the per-minute table of p_data_archive.py (so features v3 and the 6 Layer-1 features can both be computed from the result).
8 families: 3 new variants of known mechanisms (fake_breakout_v3, wash_cluster_v3, liquidity_vacuum_v3), 3 more known-type families (volume_burst_v3, one_sided_flow_v3, pump_dump_v3) and 2 families that
appear in NO training set (iceberg_absorption, periodic_pinging; flagged unseen=True). Frozen seeds: val 3101, final 3202, val2 3303 (val2 is only generated for repair round 3 of Amendment 1).
L3-final may be generated ONCE per symbol: the CLI needs --open-final and writes a lock file; a second request is refused. GENERATOR_SHA16 = hash of this file's text, logged in every output."""
import argparse, hashlib, json, os, sys, zlib
import numpy as np, pandas as pd

SEEDS = {'val': 3101, 'final': 3202, 'val2': 3303}
GENERATOR_SHA16 = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16]
INT_COLS = ['n_trades', 'n_agg', 'large_count', 'same_size_max', 'same_size_prev', 'buy_run_max', 'sell_run_max', 'n_runs', 'sec_active', 'sec_max_agg']
F_COLS = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'large_qty', 'q_p50', 'q_p90', 'q_p99', 'q_max', 'dup_share']
NEEDED = ['ts'] + F_COLS + INT_COLS


def _u(r, a, b): return float(r.uniform(a, b))


def _tri(n, peak=.4, end=0.0): return np.interp(np.arange(n) / max(n - 1, 1), [0, peak, 1], [0, 1, end])


def _plateau(n): return np.minimum(1.0, np.minimum(np.arange(n) + 1, n - np.arange(n)) / 3.0)


def _price(W, f, widen=1.0):
    for c in ('open', 'high', 'low', 'close'): W[c] = W[c] * (1.0 + f)
    if not np.all(np.asarray(widen) == 1.0):
        top, bot = np.maximum(W['open'], W['close']), np.minimum(W['open'], W['close']); W['high'] = top + (W['high'] - top) * widen; W['low'] = bot - (bot - W['low']) * widen


def _act(W, mv, mn):
    W['volume'] = W['volume'] * mv; W['taker_buy_base'] = W['taker_buy_base'] * mv
    for c in ('n_agg', 'n_trades', 'large_count', 'same_size_max', 'same_size_prev', 'n_runs'): W[c] = W[c] * (mn if c in ('n_agg', 'n_trades', 'n_runs') else 1.0)
    W['large_qty'] = W['large_qty'] * mv


def _taker(W, share): W['taker_buy_base'] = W['volume'] * np.clip(share, 0.02, 0.98)


def fam_fake_breakout_v3(W, n, r):
    s = _u(r, .004, .012) * r.choice([-1, 1]); tri = _tri(n, .4); _price(W, s * tri, 1.2); pl = _plateau(n); _act(W, 1 + _u(r, 1.5, 3.0) * tri * pl, 1 + _u(r, .8, 2.0) * tri * pl)
    x = np.arange(n) / max(n - 1, 1); sg = np.sign(s); _taker(W, .5 + sg * np.where(x <= .4, _u(r, .15, .3), -_u(r, .1, .25)) * pl); W['q_p99'] = W['q_p99'] * (1 + _u(r, .3, .8) * tri); W['large_count'] = W['large_count'] * (1 + _u(r, 1, 3) * tri)


def fam_wash_cluster_v3(W, n, r):
    pl = _plateau(n); _price(W, 0.0, .8); _act(W, 1 + _u(r, 1, 3) * pl, 1 + _u(r, .5, 1.5) * pl); _taker(W, .5 + r.normal(0, .015, n)); na = W['n_agg']
    W['same_size_max'] = np.maximum(W['same_size_max'], na * _u(r, .15, .35) * pl); W['same_size_prev'] = np.maximum(W['same_size_prev'], na * _u(r, .2, .4) * pl); W['dup_share'] = np.maximum(W['dup_share'], _u(r, .5, .8) * pl)
    W['q_p99'] = W['q_p50'] * (1 + _u(r, .5, 2) * (1 - pl) + .3); W['q_p90'] = np.minimum(W['q_p90'], W['q_p99']); W['q_max'] = W['q_p99'] * _u(r, 1.2, 2.0)
    W['buy_run_max'] = np.minimum(W['buy_run_max'], r.integers(2, 5, n)); W['sell_run_max'] = np.minimum(W['sell_run_max'], r.integers(2, 5, n)); W['sec_active'] = W['sec_active'] * (1 - .35 * pl)


def fam_liquidity_vacuum_v3(W, n, r):
    pl = _plateau(n); _price(W, _u(r, -.003, .003) * _tri(n, .5), 1 + _u(r, 1, 3) * pl); _act(W, 1 - _u(r, .3, .6) * pl, 1 - _u(r, .2, .5) * pl)
    _taker(W, .5 + np.clip(r.normal(0, .15, n), -.35, .35)); W['q_p50'] = W['q_p50'] * (1 + _u(r, .2, .5) * pl)


def fam_volume_burst_v3(W, n, r):
    pl = _plateau(n); mv = 1 + _u(r, 3, 7) * pl; _act(W, mv, 1 + _u(r, 1, 3) * pl); _price(W, 0.0, 1 + _u(r, .3, .8) * pl); W['large_count'] = W['large_count'] * (1 + _u(r, 1, 3) * pl); W['q_max'] = W['q_max'] * (1 + _u(r, .5, 1.5) * pl)


def fam_one_sided_flow_v3(W, n, r):
    pl = _plateau(n); sg = r.choice([-1, 1]); _act(W, 1 + _u(r, .3, 1.0) * pl, 1 + _u(r, .2, .8) * pl); _taker(W, .5 + sg * _u(r, .3, .42) * pl); _price(W, sg * _u(r, .002, .006) * _tri(n, .8, .6))
    k = 'buy_run_max' if sg > 0 else 'sell_run_max'; W[k] = W[k] * (1 + _u(r, 2, 5) * pl)


def fam_pump_dump_v3(W, n, r):
    pl = _plateau(n); tri = _tri(n, .5); _price(W, _u(r, .01, .03) * tri, 1.3); _act(W, 1 + _u(r, 2, 5) * tri * pl, 1 + _u(r, 1, 3) * tri * pl)
    x = np.arange(n) / max(n - 1, 1); _taker(W, np.where(x <= .5, .5 + _u(r, .15, .25), .5 - _u(r, .15, .25)) * pl + .5 * (1 - pl)); W['large_count'] = W['large_count'] * (1 + _u(r, 2, 5) * tri)


def fam_iceberg_absorption(W, n, r):                                   # UNSEEN family: repeated identical-size slices at one price, flat price, one-sided but patient flow
    pl = _plateau(n); sg = r.choice([-1, 1]); _price(W, 0.0, .7); _act(W, 1 + _u(r, .4, 1.2) * pl, 1 + _u(r, .2, .8) * pl); _taker(W, .5 + sg * _u(r, .08, .14) * pl); na = W['n_agg']
    W['same_size_max'] = np.maximum(W['same_size_max'], na * _u(r, .3, .5) * pl); W['dup_share'] = np.maximum(W['dup_share'], _u(r, .6, .85) * pl); W['q_p90'] = W['q_p50'] * (1 + _u(r, .2, .8)); W['q_p99'] = W['q_p90'] * 1.1; W['q_max'] = W['q_p99'] * 1.1
    W['large_count'] = W['large_count'] * (1 + _u(r, 1, 2.5) * pl); k = 'buy_run_max' if sg > 0 else 'sell_run_max'; W[k] = W[k] * (1 + _u(r, 1.5, 3) * pl)


def fam_periodic_pinging(W, n, r):                                     # UNSEEN family: tiny trades fired in a few seconds of every minute (machine rhythm), normal volume
    pl = _plateau(n); _price(W, 0.0, .9); _act(W, 1 + _u(r, 0, .3) * pl, 1 + _u(r, .8, 2) * pl); na = W['n_agg']; W['sec_active'] = W['sec_active'] * (1 - pl) + _u(r, 5, 12) * pl
    W['sec_max_agg'] = np.maximum(W['sec_max_agg'], na * _u(r, .25, .4) * pl); W['n_runs'] = np.maximum(W['n_runs'], na * _u(r, .8, .95) * pl); W['buy_run_max'] = np.minimum(W['buy_run_max'], r.integers(2, 4, n)); W['sell_run_max'] = np.minimum(W['sell_run_max'], r.integers(2, 4, n))
    W['same_size_max'] = np.maximum(W['same_size_max'], na * _u(r, .15, .3) * pl); W['q_p50'] = W['q_p50'] * (1 - .5 * pl); W['q_p90'] = np.maximum(W['q_p90'] * (1 - .3 * pl), W['q_p50'])


FAMILIES = {'fake_breakout_v3': (fam_fake_breakout_v3, (8, 25), False), 'wash_cluster_v3': (fam_wash_cluster_v3, (10, 40), False), 'liquidity_vacuum_v3': (fam_liquidity_vacuum_v3, (8, 30), False),
            'volume_burst_v3': (fam_volume_burst_v3, (3, 12), False), 'one_sided_flow_v3': (fam_one_sided_flow_v3, (8, 30), False), 'pump_dump_v3': (fam_pump_dump_v3, (10, 30), False),
            'iceberg_absorption': (fam_iceberg_absorption, (10, 40), True), 'periodic_pinging': (fam_periodic_pinging, (10, 40), True)}


def _repair(W):
    """Keep every row physically consistent after an edit."""
    W['volume'] = np.maximum(W['volume'], 1e-6); W['taker_buy_base'] = np.clip(W['taker_buy_base'], 0.0, W['volume'])
    for c in INT_COLS: W[c] = np.maximum(np.rint(W[c]), 0).astype('int64')
    W['n_agg'] = np.maximum(W['n_agg'], 1); W['n_trades'] = np.maximum(W['n_trades'], W['n_agg']); W['large_count'] = np.minimum(W['large_count'], W['n_agg']); W['same_size_max'] = np.clip(W['same_size_max'], 1, W['n_agg'])
    W['same_size_prev'] = np.minimum(W['same_size_prev'], W['n_agg']); W['sec_active'] = np.clip(W['sec_active'], 1, 60); W['sec_max_agg'] = np.clip(W['sec_max_agg'], 1, W['n_agg'])
    W['n_runs'] = np.clip(W['n_runs'], 1, W['n_agg']); W['buy_run_max'] = np.clip(W['buy_run_max'], 0, W['n_agg']); W['sell_run_max'] = np.clip(W['sell_run_max'], 0, W['n_agg']); W['dup_share'] = np.clip(W['dup_share'], 0.0, 1.0)
    W['large_qty'] = np.minimum(W['large_qty'], W['volume']); W['q_p90'] = np.maximum(W['q_p90'], W['q_p50']); W['q_p99'] = np.maximum(W['q_p99'], W['q_p90']); W['q_max'] = np.maximum(W['q_max'], W['q_p99'])
    W['high'] = np.maximum(W['high'], np.maximum(W['open'], W['close'])); W['low'] = np.minimum(W['low'], np.minimum(W['open'], W['close']))


def generate(base, sym, which, n_events=60, spacing=90, warm=180):
    """base: genuine per-minute table of ONE symbol (sorted, contiguous). Returns (df_with_events, events_df, meta). Only the event minutes differ from base."""
    miss = [c for c in NEEDED if c not in base.columns]
    if miss: raise ValueError('missing columns: %s' % miss)
    if which not in SEEDS: raise ValueError('set must be one of %s' % list(SEEDS))
    r = np.random.default_rng([SEEDS[which], zlib.crc32(sym.encode()) % 100000]); df = base.sort_values('ts').reset_index(drop=True).copy(); N = len(df)
    for c in F_COLS: df[c] = df[c].astype('float64')
    names = list(FAMILIES); fams = [names[i % len(names)] for i in range(n_events)]; r.shuffle(fams); ev = []; pos = warm; maxdur = max(v[1][1] for v in FAMILIES.values())
    slack = (N - warm - maxdur - 10) - n_events * (spacing + maxdur)
    if slack < 0: raise ValueError('base table too short for %d events (%d minutes)' % (n_events, N))
    gaps = np.sort(r.integers(0, slack + 1, n_events)); prev = 0; df['l3_family'] = ''; df['l3_event'] = -1
    for k, fam in enumerate(fams):
        fn, (lo, hi), unseen = FAMILIES[fam]; dur = int(r.integers(lo, hi + 1)); pos += int(gaps[k] - prev) + (spacing + maxdur if k else 0); prev = int(gaps[k]); i0, i1 = pos, pos + dur
        W = {c: df.loc[i0:i1 - 1, c].to_numpy().copy() for c in F_COLS + INT_COLS}
        for c in INT_COLS: W[c] = W[c].astype('float64')
        fn(W, dur, r); _repair(W)
        for c in W: df.loc[i0:i1 - 1, c] = W[c]
        df.loc[i0:i1 - 1, 'l3_family'] = fam; df.loc[i0:i1 - 1, 'l3_event'] = k
        ev.append({'event_id': k, 'family': fam, 'unseen': unseen, 'start_idx': i0, 'end_idx': i1 - 1, 'start_ts': df.loc[i0, 'ts'], 'end_ts': df.loc[i1 - 1, 'ts'], 'minutes': dur})
    for c in INT_COLS: df[c] = df[c].astype('int64')
    meta = {'statement': 'SYNTHETIC, labelled by construction; generator mismatch to real markets unknown', 'set': which, 'seed': SEEDS[which], 'symbol': sym, 'generator_sha16': GENERATOR_SHA16, 'n_events': n_events, 'families': {k: {'unseen': v[2], 'duration': v[1]} for k, v in FAMILIES.items()}}
    return df, pd.DataFrame(ev), meta


def main():
    a = argparse.ArgumentParser(); a.add_argument('--set', required=True, choices=list(SEEDS)); a.add_argument('--symbol', required=True); a.add_argument('--base', required=True, help='parquet of ONE genuine, contiguous validation segment')
    a.add_argument('--out', required=True); a.add_argument('--n-events', type=int, default=60); a.add_argument('--open-final', action='store_true'); x = a.parse_args(); os.makedirs(x.out, exist_ok=True)
    lock = os.path.join(x.out, 'L3_final_opened_%s.lock' % x.symbol)
    if x.set == 'final':
        if not x.open_final: raise SystemExit('L3-final is single-use: add --open-final to confirm that this is the one opening')
        if os.path.exists(lock): raise SystemExit('L3-final for %s was already generated (%s). Not allowed again.' % (x.symbol, lock))
    df, ev, meta = generate(pd.read_parquet(x.base), x.symbol, x.set, x.n_events); tag = '%s_%s' % (x.symbol, x.set)
    df.to_parquet(os.path.join(x.out, 'l3_%s.parquet' % tag)); ev.to_csv(os.path.join(x.out, 'l3_%s_events.csv' % tag), index=False); json.dump(meta, open(os.path.join(x.out, 'l3_%s_meta.json' % tag), 'w'), indent=1, default=str)
    if x.set == 'final': json.dump({'opened_utc': pd.Timestamp.now('UTC').isoformat(), 'generator_sha16': GENERATOR_SHA16}, open(lock, 'w'))
    print('ok', tag, 'events', len(ev), 'generator_sha16', GENERATOR_SHA16)


if __name__ == '__main__': main()
