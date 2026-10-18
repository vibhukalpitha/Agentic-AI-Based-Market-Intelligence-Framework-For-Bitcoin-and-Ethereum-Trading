"""Assessment reliability of every minute (novelty 3): three evidence checks, a data-quality gate, an evidence level and a conformal set.

This is an ADDITIONAL OUTPUT layered on the frozen engine scores. It never changes a model, a feature, the normalisation, the fusion stacker or a threshold.
The rules are the ones pre-registered in protocol C1 v3 and evaluated in notebook 49 (copied here, not re-tuned):

  rare        fused score > trailing 7-day 99.9th percentile of the market's own fused scores (strictly before this minute)
  persistent  fused score above the Alert line in >= 3 of the last 5 minutes (including this one)
  no conflict NOT (exactly one of XGBoost / Transformer above its own alert line while the other is at or below its trailing 7-day 90th percentile)
  gate        >= 1,440 minutes of history AND no rejected input minute (state NO_DATA) in the last 15 minutes
  level       ALERT minutes only:  STRONG = gate and 3 checks | MODERATE = gate and exactly 2 | LIMITED = every other ALERT minute
  conformal   (variant R, eps 0.10)  normal is in the set iff score <= trailing 7-day 90th percentile; abnormal is in the set iff its conformal p-value > 0.10
              (calibration table from CAL-3). Sets: normal | abnormal | cannot_tell | neither. They describe the SCORE, not whether manipulation happened.

Differences from the notebook, stated honestly: (1) in notebook 49 the data-quality gate looked at volume <= 0, trades_count <= 0 or a non-finite feature in the last 15 bars;
in the engine such a minute never reaches the models (it is rejected as NO_DATA), so the gate counts rejected minutes. (2) The notebook used pandas rolling quantiles on saved score files;
this module keeps the same 10,080-minute windows in memory and uses numpy's linear quantile (identical definition); tests/test_reliability.py proves bar-by-bar equality with a pandas re-implementation.
"""
from collections import deque
import json
from pathlib import Path
import numpy as np

EPS = 0.10
Q_RARE = 0.999
Q_LOW = 0.90
PERSIST_K, PERSIST_NEED = 5, 3
GATE_GOOD_MINUTES = 15
LEVEL_NAMES = {3: 'STRONG', 2: 'MODERATE', 1: 'LIMITED', 0: None}
SET_NAMES = {1: 'normal', 2: 'abnormal', 3: 'cannot_tell', 0: 'neither'}
RULES_VERSION = 'protocol C1 v3 (notebook 49)'


# ------------------------------------------------------------------ pure functions (same definitions as notebook 49; unit-tested against hand calculations)
def p_abnormal(x, cal_abn_sorted):
    """conformal p-value of the abnormal class: (1 + #{calibration abnormal scores <= x}) / (n + 1)"""
    cal = np.asarray(cal_abn_sorted)
    return float((1 + np.searchsorted(cal, x, side='right')) / (len(cal) + 1))


def p_normal(x, cal_norm_sorted):
    """conformal p-value of the normal class (variant S): (1 + #{calibration normal scores >= x}) / (n + 1)"""
    cal = np.asarray(cal_norm_sorted)
    return float((1 + (len(cal) - np.searchsorted(cal, x, side='left'))) / (len(cal) + 1))


def level_code(alert, gate, rare, persistent, no_conflict):
    """3 STRONG, 2 MODERATE, 1 LIMITED, 0 not an ALERT minute (or the checks are not available yet)"""
    if not alert:
        return 0
    n = int(bool(rare)) + int(bool(persistent)) + int(bool(no_conflict))
    if gate and n == 3:
        return 3
    if gate and n == 2:
        return 2
    return 1


def _q(buf, q):
    a = np.asarray(buf, 'float64')
    a = a[~np.isnan(a)]
    return a, (float(np.quantile(a, q)) if len(a) else None)


class Assets:
    """conformal calibration table of CAL-3 (sorted abnormal / normal fused scores per symbol). Optional: without it the conformal set is reported as not available."""

    def __init__(self, folder=None):
        self.folder, self.abn, self.norm, self.info, self.error = None, {}, {}, {}, None
        if folder is None:
            return
        f = Path(folder)
        try:
            z = np.load(f / 'conformal_tables_v1.npz')
            for k in z.files:
                s, kind = k.rsplit('_', 1)
                (self.abn if kind == 'abn' else self.norm)[s] = np.sort(np.asarray(z[k], 'float64'))
            d = f / 'declared_conformal_v1.json'
            self.info = json.loads(d.read_text()) if d.exists() else {}
            self.folder = str(f)
        except Exception as e:                                  # the layer still works without conformal sets
            self.error = f'{type(e).__name__}: {e}'


class ReliabilityTracker:
    def __init__(self, models, trf, alert_q, model_q, win=10080, minp=1440, assets=None, sym=None):
        self.models, self.trf, self.qa, self.mq, self.WIN, self.MINP = list(models), trf, float(alert_q), model_q, int(win), int(minp)
        self.sym, self.assets = sym, assets
        self.fh = deque(maxlen=self.WIN)
        self.mh = {m: deque(maxlen=self.WIN) for m in self.models}
        self.alert5 = deque(maxlen=PERSIST_K)
        self.good_run = 0

    # ------------------------------------------------------------------ start-up (the engine primes with a batch; the tracker copies its memory)
    def prime(self, fused_hist, model_hist, good_run=GATE_GOOD_MINUTES):
        """fused_hist: past fused scores (NaN allowed); model_hist: {model: past TRANSFORMED scores}; both oldest first, the last entry is the last primed minute"""
        self.fh.clear()
        self.fh.extend(np.asarray(fused_hist, 'float64')[-self.WIN:])
        for m in self.models:
            self.mh[m].clear()
            self.mh[m].extend(np.asarray(model_hist[m], 'float64')[-self.WIN:])
        self.alert5.clear()
        a = np.asarray(self.fh, 'float64')
        for t in range(max(0, len(a) - (PERSIST_K - 1)), len(a)):         # alert flags of the last 4 primed minutes, each against the history before it
            h = a[:t]
            h = h[~np.isnan(h)]
            self.alert5.append(bool(len(h) >= self.MINP and not np.isnan(a[t]) and a[t] > np.quantile(h, self.qa)))
        self.good_run = good_run

    def note_bad(self):
        """an input minute was rejected (NO_DATA): the engine memory is untouched, the gate closes for the next 15 minutes"""
        self.good_run = 0

    # ------------------------------------------------------------------ one scored minute
    def update(self, fz, scores, state):
        """fz: fused score of this minute (NaN while warming up); scores: {model: RAW score}; state: NORMAL / WATCH / ALERT from the engine"""
        hist, thr_rare = _q(self.fh, Q_RARE)
        n_hist = len(hist)
        ok_hist = n_hist >= self.MINP
        have = (fz == fz) and ok_hist
        alert = state == 'ALERT'
        self.alert5.append(bool(alert))
        persist_n = int(sum(self.alert5))
        persistent = persist_n >= PERSIST_NEED
        self.good_run += 1
        gate = bool(ok_hist and self.good_run >= GATE_GOOD_MINUTES)
        out = {'rules': RULES_VERSION, 'level': None, 'level_code': 0, 'checks': {'rare': None, 'persistent': persistent, 'no_conflict': None}, 'checks_met': None,
               'gate': {'ok': gate, 'history_minutes': n_hist, 'minutes_since_rejected_input': self.good_run},
               'conformal': {'available': False, 'set': None, 'set_code': None, 'p_abnormal': None}, 'detail': {'persistent_count': persist_n, 'rare_line': thr_rare, 'normal_line': None, 'models': {}}}
        if have:
            rare = bool(fz > thr_rare)
            normal_line = float(np.quantile(hist, 1.0 - EPS))
            out['detail']['normal_line'] = normal_line
            det, ab, lo = {}, {}, {}
            for m in self.models:
                a, qmod = _q(self.mh[m], self.mq[m])
                _, q90 = _q(self.mh[m], Q_LOW)
                v = float(self.trf[m](np.float64(scores[m]))) if scores[m] == scores[m] else None
                if len(a) >= self.MINP and v is not None:
                    ab[m], lo[m] = bool(qmod < v), bool(v <= q90)
                    det[m] = {'above_own_alert_line': ab[m], 'at_or_below_own_90th': lo[m]}
            xg, tr = (self.models[1], self.models[2]) if len(self.models) >= 3 else (None, None)
            no_conflict = None
            if xg in ab and tr in ab:
                no_conflict = not ((ab[xg] and lo[tr]) or (ab[tr] and lo[xg]))
            out['detail']['models'] = {m: det.get(m) for m in self.models}
            out['checks'].update(rare=rare, no_conflict=no_conflict)
            if no_conflict is not None:
                out['checks_met'] = int(rare) + int(persistent) + int(no_conflict)
                lc = level_code(alert, gate, rare, persistent, no_conflict)
                out['level_code'], out['level'] = lc, LEVEL_NAMES[lc]
            tab = self.assets.abn.get(self.sym) if self.assets else None
            if tab is not None and len(tab):
                pa = p_abnormal(fz, tab)
                code = int(fz <= normal_line) + 2 * int(pa > EPS)
                out['conformal'] = {'available': True, 'set': SET_NAMES[code], 'set_code': code, 'p_abnormal': pa}
        # the histories advance AFTER the minute was judged (every quantile above is "strictly before this minute", as in the notebook)
        self.fh.append(float(fz) if fz == fz else np.nan)
        for m in self.models:
            self.mh[m].append(float(self.trf[m](np.float64(scores[m]))) if scores[m] == scores[m] else np.nan)
        return out
