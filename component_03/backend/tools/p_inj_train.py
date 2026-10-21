"""p_inj_train.py - TRAINING injector on the per-minute archive table (protocol pack v1, P-DATA / P-FEAT3).

What it is: the mechanics of the old Layer-3 v3 injector (nb27, l3v3_lib.apply_event: 6 known families +
momentum_ignition_like / slow_accumulation_like held out) ported to the minute table written by
tools/p_data_archive.py (columns: ts, open, high, low, close, volume, taker_buy_base, n_trades, n_agg,
large_count, large_qty, q_p50, q_p90, q_p99, q_max, same_size_max, same_size_prev, dup_share,
buy_run_max, sell_run_max, n_runs, sec_active, sec_max_agg).

Design (same idea as nb27): events are injected on RAW minute quantities; features are recomputed afterwards
(base6() here, p_feat3.features_v3 for the 12 v3 features), so cross-feature coherence holds by construction.

Port of nb27 (unchanged recipes for volume/trades/large/return/buy-share/range multipliers):
  amplitude log-uniform 0.5..2.0, durations DUR, events >= GAP+TAIL apart, 15 tail bars labelled -1.
NEW in this file:
  * the v3 raw columns (size quantiles, same-size, dup_share, runs, per-second columns) are moved by the family
    'shape' rules in _shape() (tail / ssm / dup / kb / ks / nrm / burst / cov), because the nb27 recipes do not touch
    them. Same-size, n_runs and sec_max_agg scale WITH n_agg (so the p_feat3 ratios stay unchanged unless the shape
    rule changes them); run lengths and sec_active move only through the shape rules.
  * price path: returns are modified; later bars are multiplied by a constant level factor so that every later
    log-return and hl ratio is unchanged (no feature uses price levels).
  * large_trade_count = AGG-based large_count (decision for v3 candidates; limitation, handover 5.2).
  * events that would span a hole in the timestamps (missing minutes) are not placed.
VERIFIED 2026-10-19 against the uploaded files: base6() = c03_engine/features.py (log return, (buy-sell)/(v+1e-8),
  (high-low)/close, trades/(volume+1e-8), 15-bar volume_acceleration with FLOOR, large count); the invariants enforced
  here are the ones of p_feat3.py (same_size_prev = trades repeating the previous size, run/seconds/size-quantile columns).
Differences to tools/p_l3.py (generator of L3-val/L3-final, sha16 39b2cb4082122195): p_l3 edits event minutes with
  absolute targets (np.maximum(existing, n_agg*u), plateau/triangle profiles), parameter ranges drawn per event,
  its own 6 v3 families + iceberg_absorption / periodic_pinging, ~1 event per 90+ minutes, seeds 3101/3202/3303.
  THIS file: multiplicative recipes relative to the baseline (nb27: amplitude log-uniform 0.5..2.0, sigma from the past 120
  bars), nb27 profiles (u-shaped ramps, bumps), families 1-8 of nb27 (7,8 held out), return path rebuilt with a level factor,
  events >= 60 bars apart, seeds chosen by the caller (recommended TRAIN 2011/2016, never 3101/3202/3303).
  Never use p_l3 recipes here and never use this file to build L3-val / L3-final.
Held-out families 7 and 8 are only generated when asked for (fams=ALL); training uses OLD only.
"""
from __future__ import annotations
import numpy as np, pandas as pd

FAMS = {1: 'pump_and_dump_like', 2: 'fake_breakout_like', 3: 'abnormal_volume_burst', 4: 'trade_flow_imbalance',
        5: 'wash_like_aggregate_proxy', 6: 'liquidity_stress_proxy', 7: 'momentum_ignition_like', 8: 'slow_accumulation_like'}
OLD, NEW = [1, 2, 3, 4, 5, 6], [7, 8]
DUR = {1: (8, 30), 2: (6, 20), 3: (1, 8), 4: (10, 60), 5: (10, 60), 6: (5, 25), 7: (15, 45), 8: (30, 120)}
TAIL, GAP, LOOK = 15, 45, 120
FLOOR = {'BTCUSDT': 2.539920310163499, 'ETHUSDT': 27.765804660846563}     # = c03_engine/features.py
REQ = ['ts', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'n_trades', 'n_agg', 'large_count', 'large_qty',
       'q_p50', 'q_p90', 'q_p99', 'q_max', 'same_size_max', 'same_size_prev', 'dup_share', 'buy_run_max', 'sell_run_max',
       'n_runs', 'sec_active', 'sec_max_agg']
NUM = [c for c in REQ if c != 'ts']


# ------------------------------------------------------------------ base features (for tests and training tables)
def base6(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    c = df['close'].to_numpy(float); v = df['volume'].to_numpy(float); n = len(df)
    r = np.r_[0.0, np.diff(np.log(c))]
    va = np.zeros(n)
    if n > 15:
        w = np.lib.stride_tricks.sliding_window_view(v, 15)[: n - 15]
        va[15:] = (v[15:] - w.mean(axis=1)) / np.maximum(w.std(axis=1, ddof=1), FLOOR[symbol])
    return pd.DataFrame({'log_returns': r,
                         'trade_flow_imbalance': (2 * df['taker_buy_base'].to_numpy(float) - v) / (v + 1e-8),
                         'hl_spread_ratio': (df['high'].to_numpy(float) - df['low'].to_numpy(float)) / c,
                         'rush_order_density': df['n_trades'].to_numpy(float) / (v + 1e-8),
                         'volume_acceleration': va, 'large_trade_count': df['large_count'].to_numpy(float)}, index=df.index)


# ------------------------------------------------------------------ event shapes
def _side(pos, up, down):
    return np.where(pos, up, down)


def _shape(fam, D, a, sign, rng, bf, r_sl, sig):
    """multiplier / additive arrays over the event bars. nb27 part: fv ft fl dr bf hlm. NEW v3 part: tail ssm dup kb ks nrm burst cov."""
    k = np.arange(D); u = (k + 0.5) / D; ramp = np.minimum(1.0, np.minimum(u * 5, (1 - u) * 5)); amin = min(a, 1.8) / 1.8
    one, zero = np.ones(D), np.zeros(D)
    S = dict(fv=one.copy(), ft=one.copy(), fl=one.copy(), dr=zero.copy(), bf=bf.copy(), hlm=one.copy(),
             tail=one.copy(), ssm=one.copy(), dup=zero.copy(), kb=one.copy(), ks=one.copy(), nrm=one.copy(), burst=one.copy(), cov=one.copy())
    clipb = lambda x: np.clip(x, 0.02, 0.98)
    if fam == 1:
        up = u < 0.6; sh = np.where(up, u / 0.6, 1 - (u - 0.6) / 0.4); sg = np.where(up, 1.0, -1.0)
        S.update(dr=sg * np.where(up, 1.5, 2.0) * a * sig * (0.5 + sh), fv=1 + 3 * a * sh, ft=1 + 2 * a * sh, fl=1 + 3 * a * sh,
                 bf=clipb(bf + sg * 0.2 * amin * sh), hlm=1 + a * sh, tail=1 + 0.8 * a * sh, ssm=1 + 0.5 * a * sh,
                 kb=_side(sg > 0, 1 + a * sh, 1 / (1 + 0.5 * a * sh)), ks=_side(sg > 0, 1 / (1 + 0.5 * a * sh), 1 + a * sh),
                 nrm=1 / (1 + 0.3 * a * sh), burst=1 + 1.5 * a * sh, cov=1 + 0.2 * a * sh)
    elif fam == 2:
        n1 = max(1, int(0.4 * D)); n2 = max(1, D - n1); mu1 = 2.0 * a * sig; mu = np.where(k < n1, mu1, -mu1 * n1 / n2)
        bump = np.exp(-((u - 0.35) / 0.2) ** 2); ph = k < n1
        S.update(dr=mu, fv=1 + 2.5 * a * bump, ft=1 + 2.0 * a * bump, fl=1 + 2.5 * a * bump, bf=clipb(bf + np.where(ph, 0.2, -0.2) * amin),
                 hlm=1 + a * bump, tail=1 + 0.4 * a * bump, kb=_side(ph, 1 + 0.5 * a * bump, 1 / (1 + 0.3 * a * bump)),
                 ks=_side(ph, 1 / (1 + 0.3 * a * bump), 1 + 0.5 * a * bump), burst=1 + a * bump)
    elif fam == 3:
        bump = np.sin(np.pi * u)
        S.update(fv=1 + 5 * a * bump, ft=1 + 3 * a * bump, fl=1 + 5 * a * bump, dr=r_sl * 0.5 * a * bump,
                 bf=clipb(bf + rng.uniform(-0.05, 0.05, D) * bump), hlm=1 + 0.5 * a * bump,
                 tail=1 + 1.0 * a * bump, ssm=1 + 0.3 * a * bump, burst=1 + 2 * a * bump, cov=1 + 0.3 * a * bump)
    elif fam == 4:
        S.update(bf=clipb(bf + sign * min(0.25 * a, 0.45) * ramp), fv=1 + 0.3 * a * ramp, ft=1 + 0.2 * a * ramp, dr=sign * 0.3 * a * sig * ramp,
                 kb=1 + 1.5 * a * ramp if sign > 0 else 1 / (1 + 0.5 * a * ramp), ks=1 / (1 + 0.5 * a * ramp) if sign > 0 else 1 + 1.5 * a * ramp,
                 nrm=1 / (1 + 0.8 * a * ramp))
    elif fam == 5:
        S.update(fv=1 + 1.5 * a * ramp, ft=1 + 2.5 * a * ramp, fl=1 + 2.0 * a * ramp, hlm=1 - 0.4 * amin * ramp, dr=r_sl * (-0.5 * amin * ramp),
                 bf=0.5 + (bf - 0.5) * (1 - 0.8 * ramp), ssm=1 + 3 * a * ramp, dup=0.25 * amin * ramp, tail=np.maximum(0.5, 1 - 0.2 * amin * ramp),
                 nrm=1 + 0.8 * a * ramp, kb=1 / (1 + 0.8 * a * ramp), ks=1 / (1 + 0.8 * a * ramp), cov=1 + 0.3 * a * ramp)
    elif fam == 6:
        alt = np.where(k % 2 == 0, 1.0, -1.0)
        S.update(hlm=1 + 3 * a * ramp, fv=1 + a * ramp, ft=1 + a * ramp, fl=1 + 1.5 * a * ramp, dr=alt * 1.5 * a * sig * ramp,
                 bf=clipb(bf + rng.uniform(-0.1, 0.1, D) * ramp), cov=1 / (1 + 0.8 * a * ramp), burst=1 + 2 * a * ramp, tail=1 + 0.3 * a * ramp,
                 nrm=1 + 0.5 * a * ramp, kb=1 / (1 + 0.3 * a * ramp), ks=1 / (1 + 0.3 * a * ramp))
    elif fam == 7:
        n1 = max(2, int(0.8 * D)); r1 = np.where(k < n1, (np.minimum(k, n1 - 1) / n1) ** 1.5, 1.0); rev = k >= n1; di = np.where(rev, -sign, sign)
        S.update(dr=np.where(rev, -sign * 3.0 * a * sig, sign * 1.0 * a * sig * r1), fv=1 + 3 * a * r1, ft=1 + 2 * a * r1, fl=1 + 3 * a * r1,
                 bf=clipb(bf + np.where(rev, -sign * 0.2, sign * 0.25) * min(a, 1.6) / 1.6 * np.maximum(r1, 0.2)), hlm=1 + a * r1,
                 tail=1 + 0.6 * a * r1, kb=_side(di > 0, 1 + a * r1, 1 / (1 + 0.5 * a * r1)), ks=_side(di > 0, 1 / (1 + 0.5 * a * r1), 1 + a * r1),
                 burst=1 + a * r1)
    elif fam == 8:
        S.update(bf=clipb(bf + sign * 0.12 * min(a, 1.8) * ramp), fv=1 + 0.4 * a * ramp, ft=1 + 0.3 * a * ramp, fl=1 + 0.5 * a * ramp,
                 dr=sign * 0.3 * a * sig * ramp, tail=1 + 0.1 * a * ramp,
                 kb=1 + 0.5 * a * ramp if sign > 0 else 1 / (1 + 0.3 * a * ramp), ks=1 / (1 + 0.3 * a * ramp) if sign > 0 else 1 + 0.5 * a * ramp,
                 nrm=1 / (1 + 0.3 * a * ramp))
    return S


def apply_event(W, base, fam, t0, D, a, sign, rng):
    """modify the working arrays W in [t0, t0+D). Baselines (return scale) come from the ORIGINAL arrays `base`."""
    sl = slice(t0, t0 + D); r0 = base['r'][t0 - LOOK:t0]
    sig = 1.4826 * np.median(np.abs(r0 - np.median(r0))) + 1e-9
    vol0 = W['volume'][sl]; bf0 = np.where(vol0 > 0, W['taker_buy_base'][sl] / np.maximum(vol0, 1e-12), 0.5)
    S = _shape(fam, D, a, sign, rng, bf0, W['r'][sl], sig)
    v = np.maximum(vol0 * S['fv'], 0.0); tr0 = W['n_trades'][sl]; ag0 = W['n_agg'][sl]
    tr = np.where(v > 0, np.maximum(1.0, np.round(tr0 * S['ft'])), 0.0)
    ag = np.where(tr > 0, np.minimum(tr, np.maximum(1.0, np.round(ag0 * S['ft']))), 0.0)
    lg = np.minimum(np.round(np.maximum(W['large_count'][sl] * S['fl'], 0)), ag)
    lq = np.minimum(W['large_qty'][sl] * S['fl'], v)
    gm = np.where(tr0 > 0, S['fv'] / S['ft'], 1.0)                       # mean trade-size ratio
    q50 = W['q_p50'][sl] * gm ** 0.5; q90 = np.maximum(W['q_p90'][sl] * gm ** 0.8, q50)
    q99 = np.maximum(W['q_p99'][sl] * gm * S['tail'] ** 0.5, q90); qmx = np.maximum(W['q_max'][sl] * gm * S['tail'], q99)
    has = ag > 0; one = np.where(has, 1.0, 0.0)
    ss = np.clip(np.round(W['same_size_max'][sl] * S['ft'] * S['ssm']), one, ag); sp = np.clip(np.round(W['same_size_prev'][sl] * S['ft'] * S['ssm']), 0, ag)
    dup0 = W['dup_share'][sl]; dup = np.clip(dup0 + (1 - dup0) * S['dup'], 0, 1)
    bb = np.clip(np.round(W['buy_run_max'][sl] * S['kb']), one, ag); sb = np.clip(np.round(W['sell_run_max'][sl] * S['ks']), one, ag)
    sb = np.where(bb + sb > ag, np.maximum(ag - bb, 0), sb)
    nr = np.clip(np.round(W['n_runs'][sl] * S['ft'] * S['nrm']), one, ag)
    sa = np.where(has, np.clip(np.round(W['sec_active'][sl] * S['cov']), 1, 60), 0.0)
    sm = np.round(W['sec_max_agg'][sl] * S['ft'] * S['burst']); sm = np.where(has, np.minimum(np.maximum(sm, np.ceil(ag / np.maximum(sa, 1))), ag), 0.0)
    r = W['r'][sl] + S['dr']; hl = np.maximum(W['hl'][sl] * S['hlm'], np.abs(r) * 1.1)
    for key, val in dict(volume=v, taker_buy_base=np.clip(S['bf'], 0, 1) * v, n_trades=tr, n_agg=ag, large_count=lg, large_qty=lq,
                         q_p50=q50, q_p90=q90, q_p99=q99, q_max=qmx, same_size_max=ss, same_size_prev=sp, dup_share=dup,
                         buy_run_max=bb, sell_run_max=sb, n_runs=nr, sec_active=sa, sec_max_agg=sm, r=r, hl=hl).items():
        W[key][sl] = val


# ------------------------------------------------------------------ placement, injection
def place_events(lo, hi, fams, rng, brk_cum, max_tries=20000):
    """(t0, D, fam) inside [lo, hi); >= GAP+TAIL apart; t0 >= lo+LOOK; no timestamp hole in [t0-LOOK, t0+D+TAIL]."""
    taken, ev = [], []
    for fam in fams:
        d0, d1 = DUR[fam]; D = int(rng.integers(d0, d1 + 1)); ok = False
        for _ in range(max_tries):
            t0 = int(rng.integers(lo + LOOK, hi - D - TAIL))
            if brk_cum[t0 + D + TAIL] - brk_cum[t0 - LOOK] != 0:
                continue
            if all(t0 + D + TAIL + GAP <= s or e + TAIL + GAP <= t0 for s, e in taken):
                taken.append((t0, t0 + D)); ev.append((t0, D, fam)); ok = True; break
        if not ok:
            break
    return sorted(ev)


def amp(rng): return float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))


def _brk_cum(ts):
    m = np.asarray(ts).astype('datetime64[m]').astype('int64')
    return np.cumsum(np.r_[0, (np.diff(m) != 1).astype(int)])


def inject(df, symbol, events, seed):
    """events: DataFrame t0, dur, family, amp (+ optional sign). Returns (table, events with sign). Same inputs -> same output."""
    missing = [c for c in REQ if c not in df.columns]
    assert not missing, f'minute table lacks columns {missing}'
    n = len(df); rng = np.random.default_rng(seed)
    W = {c: df[c].to_numpy(float).copy() for c in NUM}
    c0 = W['close']; W['r'] = np.r_[0.0, np.diff(np.log(c0))]; W['hl'] = (W['high'] - W['low']) / c0
    base = {k: v.copy() for k, v in W.items()}
    lab = np.zeros(n, np.int16); eid = np.full(n, -1, np.int32); ev = events.reset_index(drop=True).copy()
    drawn = rng.choice([-1.0, 1.0], size=len(ev))          # always drawn, so a replay with stored signs uses the same rng state
    if 'sign' not in ev.columns:
        ev['sign'] = drawn
    for i, e in enumerate(ev.itertuples()):
        t0, D, fam = int(e.t0), int(e.dur), int(e.family)
        apply_event(W, base, fam, t0, D, float(e.amp), float(e.sign), rng)
        lab[t0:t0 + D] = fam; eid[t0:t0 + D] = i; lab[t0 + D:t0 + D + TAIL] = -1
    out = df.copy()
    for c in ['volume', 'taker_buy_base', 'n_trades', 'n_agg', 'large_count', 'large_qty', 'q_p50', 'q_p90', 'q_p99', 'q_max', 'same_size_max',
              'same_size_prev', 'dup_share', 'buy_run_max', 'sell_run_max', 'n_runs', 'sec_active', 'sec_max_agg']:
        out[c] = W[c]
    # price path: constant level factor after every event keeps all other returns / hl ratios unchanged
    m = np.exp(np.cumsum(W['r'] - base['r'])); mp = np.r_[1.0, m[:-1]]
    close = df['close'].to_numpy(float) * m; opn = df['open'].to_numpy(float) * mp
    mid = (df['high'].to_numpy(float) + df['low'].to_numpy(float)) / 2 * m; ins = lab > 0
    hi = np.where(ins, mid + W['hl'] * close / 2, df['high'].to_numpy(float) * m); lo = np.where(ins, mid - W['hl'] * close / 2, df['low'].to_numpy(float) * m)
    out['close'] = close; out['open'] = opn; out['high'] = np.maximum(hi, np.maximum(opn, close)); out['low'] = np.minimum(lo, np.minimum(opn, close))
    out['label'] = lab; out['event_id'] = eid
    return out, ev


def build_replica(df, symbol, lo, hi, fams, seed):
    rng = np.random.default_rng(seed); bc = _brk_cum(df['ts'])
    order = list(fams); rng.shuffle(order); pl = place_events(lo, hi, order, rng, bc)
    ev = pd.DataFrame([{'event': i, 't0': t0, 'dur': D, 'family': f, 'name': FAMS[f], 'amp': amp(rng)} for i, (t0, D, f) in enumerate(pl)])
    out, ev = inject(df, symbol, ev, seed + 1)
    return out, ev


def capacity(lo, hi, fams):
    md = np.mean([np.mean(DUR[f]) for f in fams]); return max(1, int(0.5 * (hi - lo - LOOK) / (md + TAIL + GAP)))


def iter_set(df, symbol, lo, hi, total, fams, seed):
    """pooled replicas (at most ~50% of bars inside events per replica), yielded one by one (memory). Equal family mix."""
    check_fams(fams)
    cap = capacity(lo, hi, fams); nrep = int(np.ceil(total / cap)); per = int(np.ceil(total / nrep))
    for r in range(nrep):
        fl = [fams[i % len(fams)] for i in range(per)]
        f, ev = build_replica(df, symbol, lo, hi, fl, seed * 1000 + r + 7)
        f['replica'] = r; f['bar'] = np.arange(len(f)); f['in_seg'] = (f['bar'] >= lo) & (f['bar'] < hi); ev['replica'] = r
        yield f, ev


def build_set(df, symbol, lo, hi, total, fams, seed):
    fr, evs = zip(*iter_set(df, symbol, lo, hi, total, fams, seed))
    return pd.concat(fr), pd.concat(evs, ignore_index=True)


def check_fams(fams):
    assert set(fams) <= set(FAMS), fams
