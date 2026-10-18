"""Historical context (novelty 4): the validated retrieval of protocol H1 v4 / notebook 53, as an additional output of the frozen engine.

Query = the last 60 minutes of this market, written as the 24-number descriptor "c2" chosen by the pre-registered selection on CAL-3:
  12 score summaries (fused, AE, XGBoost, Transformer x mean / max / last over the 60 minutes)
  + 6 mean signed z of the six Layer-1 features + 6 last signed z (N1 normalisation, the normalisation the engine already uses)
standardised with the per-symbol mean/std of genuine DEV windows (stored in the index), then Euclidean distance.
Only windows that ended at least 60 (outcome) + 60 (window) minutes before the query window began can be returned, and no two returned windows are closer than 120 minutes.
Result = historical CONTEXT (what the market did in the hour after similar windows). Notebook 53: 5 of 6 hypotheses held; H-H5 (using neighbours to raise the probability of an Alert) FAILED, so no Alert probability is derived from them.

This module only reads index files; it never changes a model, a threshold or a score. The index files are data (not committed): history_index_v2_<SYMBOL>.npz.
"""
import json
from pathlib import Path
import numpy as np

W, OUTW, GAP, K = 60, 60, 120, 5
DEV_LO, DEV_HI = 90720, 129600
DEVN = DEV_HI - DEV_LO                    # the genuine timeline of the index = DEV part (38,880 bars) followed by Aug-Oct 2024; timeline index of 2024-08-01T00:00 is DEVN
AUG0_UTC = np.datetime64('2024-08-01T00:00:00', 'ns').astype('int64')
NAMES = [f'{s}_{t}' for s in ('fused', 'AE', 'XGBoost', 'Transformer') for t in ('mean', 'max', 'last')] + [f'z_mean_{i}' for i in range(6)] + [f'z_last_{i}' for i in range(6)]


def select(dist, times, k=K, gap=GAP):
    """same as notebook 53: indices of up to k nearest candidates, nearest first, every pair at least `gap` bars apart"""
    order = np.argsort(dist, kind='stable'); picked = []
    for i in order:
        if all(abs(times[i] - times[j]) >= gap for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return picked


def retrieve(A, At, q, q_end, k=K, gap=GAP):
    """notebook 53 `retrieve`: A standardised archive (n, d) sorted by end time At; q query descriptor; past-only: archive end + OUTW < query start (q_end - W + 1)"""
    n = int(np.searchsorted(At, q_end - W + 1 - OUTW, side='left'))
    if n == 0:
        return np.array([], int), np.array([])
    d = np.sqrt(((A[:n] - q) ** 2).sum(axis=1)); m = min(n, 600)
    part = np.argpartition(d, m - 1)[:m] if n > m else np.arange(n)
    sel = select(d[part], At[part], k, gap)
    if len(sel) < k and n > m:
        sel = select(d, At[:n], k, gap)
        return np.array(sel, int), d[sel]
    ii = part[sel]
    return np.array(ii, int), d[ii]


def descriptor_raw(F4, z):
    """F4: 4 arrays (fused, AE, XGBoost, Transformer; transformed scores) of the last 60 minutes, oldest first; z: (60, 6) signed z of the features. Returns the 24 raw numbers (None when any is not finite)."""
    F4 = [np.asarray(x, 'float64') for x in F4]; z = np.asarray(z, 'float64')
    if any(len(x) != W for x in F4) or z.shape != (W, 6):
        return None
    d = np.concatenate([np.array([x.mean(), x.max(), x[-1]]) for x in F4] + [z.mean(axis=0), z[-1]])
    return d if np.isfinite(d).all() else None


class HistoryIndex:
    def __init__(self, folder=None):
        self.sym, self.version, self.spec, self.error, self.tr = {}, {}, {}, None, {}
        if folder is None:
            return
        f = Path(folder)
        try:                                   # optional: fused score + Alert line of every July minute (notebook 60), used only to DRAW a matched hour
            if (f / 'history_traces_v1.npz').exists():
                z = np.load(f / 'history_traces_v1.npz')
                self.tr = {s: (z[f'{s}_ts'], z[f'{s}_F'], z[f'{s}_thr']) for s in ('BTCUSDT', 'ETHUSDT') if f'{s}_F' in z.files}
        except Exception:
            self.tr = {}
        try:
            j = json.loads((f / 'history_index_v2.json').read_text())
            self.spec, self.version = j['spec'], j.get('version_sha256', {})
            for s in ('BTCUSDT', 'ETHUSDT'):
                z = np.load(f / f'history_index_v2_{s}.npz')
                self.sym[s] = {k: z[k] for k in z.files}
            assert self.spec.get('candidate') == 'c2', f"index candidate is {self.spec.get('candidate')}, this module implements c2"
        except Exception as e:
            self.sym, self.error = {}, f'{type(e).__name__}: {e}'

    @property
    def ok(self):
        return bool(self.sym)

    def trace(self, sym, end_ts_ns, back=W, fwd=OUTW):
        """fused score and Alert line of the `back` minutes ending at `end_ts_ns` plus the `fwd` minutes after it (July minutes, from notebook 60); None when not available"""
        if sym not in self.tr:
            return None
        ts, F, thr = self.tr[sym]
        p = int(np.searchsorted(ts, end_ts_ns))
        a, b = p - back + 1, p + fwd + 1
        if p >= len(ts) or ts[p] != end_ts_ns or a < 0 or b > len(ts) or ts[b - 1] - ts[a] != (b - 1 - a) * 60_000_000_000:
            return None
        f, t = F[a:b], thr[a:b]
        return (f.copy(), t.copy()) if np.isfinite(f).all() else None

    def standardise(self, sym, raw24):
        """raw 24 numbers -> the query vector in the index space. The index stores the 32 extended columns standardised (mean, std over genuine DEV windows) and then selects `cols` and multiplies by `weights`."""
        a = self.sym[sym]; cols = a['cols']; return (np.asarray(raw24) - a['mean'][cols]) / a['std'][cols] * a['weights']

    def timeline_t(self, q_ts_ns):
        """timeline index of a bar of the replay (the replay starts at 2024-08-01T00:00 UTC = timeline index DEVN, one bar per minute)"""
        return int(DEVN + (int(q_ts_ns) - AUG0_UTC) // 60_000_000_000)

    def find(self, sym, raw24, q_ts_ns, k=K):
        """k most similar earlier windows -> list of dicts (nearest first)"""
        a = self.sym[sym]; q = self.standardise(sym, raw24); qe = self.timeline_t(q_ts_ns)
        ii, dist = retrieve(a['Z'].astype('float64'), a['t'], q, qe, k)
        med = float(np.median(np.sqrt(((a['Z'][: max(1, int(np.searchsorted(a['t'], qe - W + 1 - OUTW)))][::50].astype('float64') - q) ** 2).sum(axis=1)))) if len(ii) else None
        out = []
        for i, d in zip(ii, dist):
            out.append({'end_utc': str(np.datetime64(int(a['ts'][i]), 'ns'))[:19] + 'Z', 'end_ts_ns': int(a['ts'][i]), 'distance': float(d), 'typical_distance': med,
                        'peak_fused_next_hour': None if not np.isfinite(a['PK'][i]) else float(a['PK'][i]), 'minutes_above_alert_next_hour': None if not np.isfinite(a['MA'][i]) else int(a['MA'][i])})
        return out
