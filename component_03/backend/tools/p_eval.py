"""p_eval.py - shared evaluation module for the C03 improvement round (protocol pack v1).

Rules implemented (handover section 3):
  * episodes: alarm bars merged when the time gap between consecutive alarm bars <= `gap` minutes
    (gap measured on timestamps, so missing minutes are handled). ASSUMPTION to cross-check vs nb29b.
  * genuine_eps_per_day: episodes / (number of genuine bars / 1440).
  * t1_threshold: q-quantile of the series' OWN scores over the previous `window` bars,
    strictly before t (shift 1). NaN until MIN_PERIODS (1440) valid values exist.
  * t1c_alarms: injected scores compared with the threshold computed on the GENUINE series
    at the same bars (series must be aligned, same length). Alarm = score > threshold.
  * recall = share of events with >= 1 alarm inside [start, end] (inclusive bar indices);
    macro over families; mean of BTC/ETH via mean_over_symbols.
  * paired_bootstrap: 95% event-level bootstrap, resampling events WITHIN family (stratified),
    same resample for both arms (paired); reproducible by seed.
  * lofo: generic leave-one-family-out driver.
Pure numpy/pandas. No data is read here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WINDOW = 10080          # 7 days of 1-min bars
MIN_PERIODS = 1440      # 1 day
BARS_PER_DAY = 1440


def _to_minutes(ts) -> np.ndarray:
    a = np.asarray(ts)
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[m]").astype("int64")
    return a.astype("int64")  # already integer minutes


def episodes(alarm_bool, ts, gap: int = 5):
    """Merge alarm bars into episodes. Returns int array shape (k,2): [start_idx, end_idx] inclusive."""
    alarm = np.asarray(alarm_bool, dtype=bool)
    idx = np.flatnonzero(alarm)
    if idx.size == 0:
        return np.zeros((0, 2), dtype=int)
    m = _to_minutes(ts)[idx]
    breaks = np.flatnonzero(np.diff(m) > gap)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return np.column_stack([starts, ends]).astype(int)


def genuine_eps_per_day(alarm_bool, ts, gap: int = 5, valid=None) -> float:
    """Episodes per day on genuine bars. Same rule as b2_lib.eps_per_day (nb29b): alarms are restricted to
    `valid` bars (warm-up masked) and days = valid.sum()/1440. valid=None -> all bars valid."""
    a = np.asarray(alarm_bool, dtype=bool)
    v = np.ones(len(a), bool) if valid is None else np.asarray(valid, bool)
    if v.sum() == 0:
        return float("nan")
    return len(episodes(a & v, ts, gap)) / (v.sum() / BARS_PER_DAY)


QGRID = [0.99, 0.993, 0.995, 0.997, 0.998, 0.999, 0.9993, 0.9995, 0.9997, 0.9998, 0.9999, 0.99995]


def t1_q(scores, ts, valid, budget: float, qgrid=QGRID, window: int = WINDOW) -> float:
    """Smallest q on the grid whose T1 alarms give <= budget genuine episodes/day (as b2_lib.t1_q)."""
    for q in qgrid:
        if genuine_eps_per_day(t1_alarms(scores, q, window), ts, valid=valid) <= budget:
            return q
    return qgrid[-1]


def events_from_t0dur(ev: pd.DataFrame) -> pd.DataFrame:
    """Old event tables (t0, dur, family) -> start/end inclusive (end = t0+dur-1, as b2_lib.recall_by_family)."""
    out = ev.copy()
    out["start"] = ev["t0"].to_numpy(int)
    out["end"] = ev["t0"].to_numpy(int) + ev["dur"].to_numpy(int) - 1
    return out


def t1_threshold(scores, q: float, window: int = WINDOW, min_periods: int = MIN_PERIODS) -> np.ndarray:
    """Past-only rolling quantile: value at t uses scores[t-window : t) only."""
    s = pd.Series(np.asarray(scores, dtype=float))
    return s.rolling(window, min_periods=min_periods).quantile(q).shift(1).to_numpy()


def t1_alarms(scores, q: float, window: int = WINDOW) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    thr = t1_threshold(s, q, window)
    return np.where(np.isnan(thr) | np.isnan(s), False, s > thr)


def t1c_alarms(scores_injected, scores_genuine, q: float, window: int = WINDOW) -> np.ndarray:
    """Threshold from the genuine series at the same bars, applied to the injected series."""
    si = np.asarray(scores_injected, dtype=float)
    sg = np.asarray(scores_genuine, dtype=float)
    if si.shape != sg.shape:
        raise ValueError("injected and genuine series must be aligned (same length)")
    thr = t1_threshold(sg, q, window)
    return np.where(np.isnan(thr) | np.isnan(si), False, si > thr)


def event_hits(alarms, events: pd.DataFrame) -> np.ndarray:
    """bool per event: any alarm in [start, end] (inclusive). events needs columns start, end."""
    c = np.r_[0, np.cumsum(np.asarray(alarms, dtype=bool))]
    s = events["start"].to_numpy(int)
    e = events["end"].to_numpy(int)
    return (c[e + 1] - c[s]) > 0


def macro_recall(hits, families, idx=None) -> float:
    """Macro over families of the share of hit events (optionally on a resample idx)."""
    hits = np.asarray(hits, bool)
    fam = np.asarray(families)
    if idx is not None:
        hits, fam = hits[idx], fam[idx]
    vals = [hits[fam == f].mean() for f in np.unique(fam)]
    return float(np.mean(vals)) if vals else float("nan")


def recall_by_family(alarms, events: pd.DataFrame) -> dict:
    """{'per_family': {fam: recall}, 'macro': float, 'n_events': int}"""
    hits = event_hits(alarms, events)
    fam = events["family"].to_numpy()
    per = {str(f): float(hits[fam == f].mean()) for f in np.unique(fam)}
    return {"per_family": per, "macro": float(np.mean(list(per.values()))) if per else float("nan"),
            "n_events": int(len(events))}


def mean_over_symbols(values: dict) -> float:
    return float(np.mean(list(values.values())))


def paired_bootstrap(diff_fn, families, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """diff_fn(idx)->float, idx = resampled event indices (same idx for both arms => paired).
    Resampling is stratified by family. Returns point, mean, lo, hi (95%), n."""
    fam = np.asarray(families)
    groups = [np.flatnonzero(fam == f) for f in np.unique(fam)]
    rng = np.random.default_rng(seed)
    point = float(diff_fn(np.arange(len(fam))))
    out = np.empty(n)
    for b in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        out[b] = diff_fn(idx)
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {"point": point, "mean": float(out.mean()), "lo": float(lo), "hi": float(hi), "n": n, "seed": seed}


def diff_macro_recall(hits_a, hits_b, families):
    """Returns diff_fn for paired_bootstrap: macro recall(A) - macro recall(B)."""
    ha, hb = np.asarray(hits_a, bool), np.asarray(hits_b, bool)
    return lambda idx: macro_recall(ha, families, idx) - macro_recall(hb, families, idx)


def lofo(families, run_fn) -> dict:
    """Leave-one-family-out. run_fn(held_out_family) -> bool hits over ALL events (model fitted without
    that family); recall of the held-out family is read from its own events only.
    For label-free models run_fn can ignore its argument."""
    fam = np.asarray(families)
    per = {}
    for f in np.unique(fam):
        hits = np.asarray(run_fn(f), bool)
        per[str(f)] = float(hits[fam == f].mean())
    return {"per_family": per, "macro": float(np.mean(list(per.values())))}
