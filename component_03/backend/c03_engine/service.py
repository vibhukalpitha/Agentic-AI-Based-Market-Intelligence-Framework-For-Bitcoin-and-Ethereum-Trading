"""Per-symbol service state around the scoring core: statuses, bars, windows (15 min / 1 h), alert episodes, state changes. No I/O here."""
from collections import deque
import datetime as dt
import numpy as np
from .core import SymbolEngine
from . import explain
from .reliability import ReliabilityTracker, Assets
import os, threading
from collections import OrderedDict
from . import context
from .validate import check_features, check_warmup, problem, DataError

_ASSETS = {}


def default_assets():
    """optional N3/N4 assets folder (conformal tables, history index): env C03_ASSETS. Missing folder or files -> the reliability layer still runs, conformal sets are reported as not available."""
    f = os.environ.get('C03_ASSETS')
    if f not in _ASSETS: _ASSETS[f] = Assets(f)
    return _ASSETS[f]


LIMIT_REF = '/api/c03/model-info#limitations'
HORIZONS = {'15m': ('15 min', 15), '1h': ('1 hour', 60)}


def iso(ts): return ts if ts.endswith('Z') else ts.replace(' ', 'T')[:19] + 'Z'


def minute_of(ts): return int(dt.datetime.fromisoformat(iso(ts).replace('Z', '+00:00')).timestamp() // 60)


class SymbolService:
    def __init__(self, export, sym, max_bars=20000, assets=None):
        self.ex, self.sym = export, sym; self.eng = SymbolEngine(export, sym); self.assets = assets if assets is not None else default_assets(); self.rel = ReliabilityTracker(self.eng.MODELS, self.eng.c.TRF, self.eng.qa, self.eng.mq, self.eng.WIN, self.eng.MINP, self.assets, sym); mh = export.decl['mh']; self.mh_thr = mh['thresholds'][sym]; self.mh_pts = {h: np.asarray(mh['score100_percentile_points_genuine_dev_windows'][sym][v[0]]) for h, v in HORIZONS.items()}
        self.bars = deque(maxlen=max_bars); self.windows = {h: deque(maxlen=5000) for h in HORIZONS}; self.cur = {h: None for h in HORIZONS}; self.episodes = []; self.open_ep = None; self.state = 'NORMAL'; self.last = None
        self.last_m = None; self.n_bad = 0; self.consec_bad = 0
        self.lock = threading.RLock(); self.hidx = context.default_history(); self.dwin = deque(maxlen=max_bars); self.xai = OrderedDict()      # dwin: the scored minutes held (up to max_bars) for the history query (the last 60 up to the asked minute are used); xai: detail of flagged bars

    def prime(self, F, ts):
        check_warmup(F)                         # clear DataError (too short / columns missing / bad rows) instead of an assertion or a silent NaN
        fz = self.eng.prime(F, ts); self.last_ts = ts[-1]; self.rel.prime(fz, {m: list(self.eng.hist[m]) for m in self.eng.MODELS})
        try: self.last_m = minute_of(ts[-1])
        except Exception: self.last_m = None

    def _tier(self, h, peak):
        t = self.mh_thr[HORIZONS[h][0]]; return 'ALERT' if peak > t['Alert'] else ('WATCH' if peak > t['Watch'] else 'NORMAL')

    def _unus100(self, h, peak): return float(np.searchsorted(self.mh_pts[h], peak) / max(len(self.mh_pts[h]), 1) * 100.0)

    def on_bar(self, feat, ts, ohlc=None):
        with self.lock: return self._on_bar(feat, ts, ohlc)

    def _on_bar(self, feat, ts, ohlc=None):
        """Process one closed bar. Returns (status, events). A bar that fails the input checks is not given to the models: it is reported as state NO_DATA."""
        problems = check_features(feat)
        try: m = minute_of(ts)
        except Exception: m = None; problems = problems + [problem('bad_timestamp', str(ts)[:40])]
        if m is not None and self.last_m is not None and m <= self.last_m: problems = problems + [problem('duplicate_or_out_of_order', f'{iso(ts)} is not after {dt.datetime.fromtimestamp(self.last_m * 60, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}')]
        if problems: return self.reject(ts, problems, feat, ohlc)
        warnings = [] if self.last_m is None or m - self.last_m == 1 else [f'{m - self.last_m - 1} minute(s) are missing before this bar; indicators that look back 3 or 15 minutes mix across the gap']
        self.last_m = m; self.consec_bad = 0
        r = self.eng.step(feat, ts); events = []; state = r['state']; fz = r['fusion']; reasons, caveats = ([], [])
        sc_ = r['scores']; self.dwin.append((m, fz, sc_['ae'], sc_['xgboost'], sc_['transformer'], np.array(self.eng.zbuf['N1'][-1], 'float64')))
        if state != 'NORMAL':
            try:
                self.xai[iso(ts)] = context.xai_detail(self.eng)
                while len(self.xai) > 300: self.xai.popitem(last=False)
            except Exception: pass
        if state != 'NORMAL': reasons, caveats = explain.reasons(self.eng, r)
        rel = self.rel.update(fz, {m: r['scores'][k] for m, k in zip(self.eng.MODELS, ['ae', 'xgboost', 'transformer'])}, state)
        f = {k: float(v) for k, v in zip(self.ex.B.FEATS, feat)}
        bar = {'symbol': self.sym, 'bar_open_utc': iso(ts), 'features': f, 'scores': r['scores'], 'fused_score': fz, 'unusualness': r['unusualness'], 'state': state, 'confidence': r['confidence'], 'warmup': r['warmup'], 'close': None if ohlc is None else ohlc.get('close')}
        self.bars.append(bar)
        # windows
        for h, (_, n) in HORIZONS.items():
            start = m // n * n; cw = self.cur[h]
            if cw is not None and cw['start'] != start:
                w = self._close_window(h); events.append({'type': 'window_closed', 'data': w}) if w['tier'] != 'NORMAL' else None; cw = None
            if cw is None: cw = self.cur[h] = {'start': start, 'peak': -np.inf, 'flagged': 0, 'bars': 0}
            if fz == fz: cw['peak'] = max(cw['peak'], fz)
            cw['flagged'] += int(state != 'NORMAL'); cw['bars'] += 1
        # episodes (consecutive ALERT bars within 5 minutes)
        if state == 'ALERT':
            if self.open_ep is None or m - self.open_ep['_last'] > 5:
                if self.open_ep is not None: self._close_ep()
                self.open_ep = {'id': f'{self.sym}-{m}', 'symbol': self.sym, 'start_utc': iso(ts), 'end_utc': None, 'peak_unusualness': 0.0, 'state_max': 'ALERT', 'confidence': r['confidence'], 'models_above': set(), 'status': 'OPEN', '_last': m}; self.episodes.append(self.open_ep)
            e = self.open_ep; e['_last'] = m; e['end_utc'] = iso(ts); e['peak_unusualness'] = max(e['peak_unusualness'], r['unusualness'] or 0.0); e['models_above'] |= set(r['models_above'] or [])
            order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}; e['confidence'] = max([e['confidence'], r['confidence']], key=lambda x: order.get(x, -1))
        elif self.open_ep is not None and m - self.open_ep['_last'] > 5: self._close_ep()
        status = self._status(bar, r, reasons, caveats); status['data_quality'] = {'ok': True, 'warnings': warnings, 'bad_minutes_total': self.n_bad}; status['reliability'] = rel; bar['reliability'] = {k: rel[k] for k in ('level', 'level_code', 'checks_met')} | {'conformal_set': rel['conformal']['set']}
        bar['x'] = {'thr_alert': r['thr_alert'], 'thr_watch': r['thr_watch'], 'z': r['z'], 'contrib': r['contrib'], 'plevel': r['plevel'], 'models_above': r['models_above'], 'reasons': reasons, 'rel': rel}      # v0.2.1 additive: per-bar detail, returned by GET /bars only with detail=true
        if state != self.state: events.append({'type': 'state_change', 'data': {'symbol': self.sym, 'from': self.state, 'to': state, 'at_utc': iso(ts), 'status': status}}); self.state = state
        events.insert(0, {'type': 'bar_scored', 'data': status}); self.last = status; return status, events

    # ------------------------------------------------------------------ queries used by /explain, /similar, /reliability
    def explain(self, at=None):
        """detail of the latest bar (computed now) or of an earlier FLAGGED bar (kept for the last 300 flagged bars); None when not available"""
        with self.lock:
            if not self.bars or self.bars[-1]['state'] == 'NO_DATA': return None
            last = self.bars[-1]; at = at or last['bar_open_utc']
            if at == last['bar_open_utc']: det = context.xai_detail(self.eng); bar = last
            else:
                det = self.xai.get(at); bar = next((b for b in reversed(self.bars) if b['bar_open_utc'] == at), None)
                if det is None or bar is None: return None
            return {'symbol': self.sym, 'bar_open_utc': bar['bar_open_utc'], 'state': bar['state'], 'detail': det,
                    'reasons': (self.last or {}).get('reasons', []) if bar is last else [],
                    'validation': {'xgboost_groups': 'supported on held-out test events (notebook 59, protocol E4 v1)', 'single_features': 'indicative only', 'transformer': 'group level only (notebook 55)', 'ae_errors': 'secondary, lower confidence (notebook 45)'},
                    'interventional_available': 'shap_int' in det}

    def similar(self, k=5, at=None):
        """the k most similar past hours (protocol H1 v4): context only, never an alert probability (H-H5 failed). `at` (optional, UTC ISO minute) asks about a minute the engine still holds instead of the newest one."""
        with self.lock:
            base = {'symbol': self.sym, 'note': 'Historical context: what the market did in the hour after similar hours. It does not change any score or alert.'}
            if not self.hidx.ok: return base | {'status': 'unavailable', 'reason': 'the history index is not loaded (set C03_ASSETS)', 'matches': []}
            full = list(self.dwin)
            if at is not None:
                try: m_at = minute_of(at)
                except Exception: return base | {'status': 'unavailable', 'reason': f'the time {str(at)[:40]!r} is not a valid UTC time', 'matches': []}
                pos = next((n for n, x in enumerate(full) if x[0] == m_at), None)
                if pos is None: return base | {'status': 'unavailable', 'reason': 'the engine holds no scored minute for that time', 'matches': []}
                w = full[max(0, pos - 59):pos + 1]
            else:
                w = full[-60:]
            if len(w) < 60: return base | {'status': 'collecting', 'have': len(w), 'need': 60, 'matches': []}
            mins = [x[0] for x in w]
            if mins[-1] - mins[0] != 59 or any(b - a != 1 for a, b in zip(mins, mins[1:])): return base | {'status': 'collecting', 'reason': 'a minute is missing in the hour before; the query needs 60 consecutive scored minutes', 'have': len(w), 'need': 60, 'matches': []}
            from .history import descriptor_raw
            d = descriptor_raw([[x[i] for x in w] for i in (1, 2, 3, 4)], np.array([x[5] for x in w]))
            if d is None: return base | {'status': 'unavailable', 'reason': 'a score of the last hour is not finite', 'matches': []}
            res = self.hidx.find(self.sym, d, mins[-1] * 60 * 10 ** 9, k); out = []
            for r in res:
                tr = self.hidx.trace(self.sym, r['end_ts_ns']); out.append(r | {'trace': None if tr is None else {'fused': tr[0].tolist(), 'alert_line': tr[1].tolist(), 'minutes_before': 60 - 1, 'minutes_after': 60}})
            qend = dt.datetime.fromtimestamp(mins[-1] * 60, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            return base | {'status': 'ok', 'query_end_utc': qend, 'now_fused': [x[1] for x in w], 'matches': out}

    def reliability_view(self, limit=120):
        """current reliability object plus the level / checks / conformal set of the last `limit` bars"""
        with self.lock:
            cur = (self.last or {}).get('reliability'); ser = [{'bar_open_utc': b['bar_open_utc'], **b['reliability']} for b in list(self.bars)[-limit:] if b.get('reliability')]
            return {'symbol': self.sym, 'current': cur, 'series': ser, 'note': 'Checks: rare, persistent, no conflict between XGBoost and Transformer. Conformal set: normal / abnormal / cannot_tell. Levels describe how much to trust an ALERT; they are shown as display labels.'}

    def reject(self, ts, problems, feat=None, ohlc=None):
        """report a minute that cannot be scored: state NO_DATA, the reasons in plain words. The engine memory, windows and episodes are left untouched."""
        try: bt = iso(ts)
        except Exception: bt = str(ts)
        self.n_bad += 1; self.consec_bad += 1; self.rel.note_bad()
        try: f = {k: float(v) for k, v in zip(self.ex.B.FEATS, feat)}
        except Exception: f = None
        why = '; '.join(p['text'] for p in problems)
        status = {'symbol': self.sym, 'bar_open_utc': bt, 'state': 'NO_DATA', 'unusualness': None, 'integrity_score': None, 'confidence': None, 'fused_score': None,
                  'components': {'ae': {'z': None, 'contribution': None}, 'xgboost': {'z': None, 'contribution': None}, 'transformer': {'z': None, 'contribution': None}, 'tgnn': None},
                  'agreement': None, 'horizons': {}, 'reasons': [], 'caveats': ['This minute was not scored: ' + why], 'warmup': False, 'thresholds': None, 'limitations_ref': LIMIT_REF,
                  'data_quality': {'ok': False, 'problems': problems, 'consecutive_bad': self.consec_bad, 'bad_minutes_total': self.n_bad}}
        self.bars.append({'symbol': self.sym, 'bar_open_utc': bt, 'features': f, 'scores': None, 'fused_score': None, 'unusualness': None, 'state': 'NO_DATA', 'confidence': None, 'warmup': False, 'close': None if ohlc is None else ohlc.get('close'), 'data_quality': status['data_quality']})
        self.last = status
        return status, [{'type': 'bad_input', 'data': status}]

    def _close_ep(self):
        self.open_ep['status'] = 'CLOSED'; self.open_ep = None

    def _close_window(self, h):
        cw = self.cur[h]; n = HORIZONS[h][1]; peak = cw['peak'] if np.isfinite(cw['peak']) else None
        w = {'symbol': self.sym, 'horizon': h, 'window_start_utc': dt.datetime.fromtimestamp(cw['start'] * 60, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'window_end_utc': dt.datetime.fromtimestamp((cw['start'] + n) * 60, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
             'peak_score': peak, 'peak_unusualness': None if peak is None else self._unus100(h, peak), 'tier': 'NORMAL' if peak is None else self._tier(h, peak), 'flagged_minutes': cw['flagged']}
        self.windows[h].append(w); return w

    def _status(self, bar, r, reasons, caveats):
        comp = {}
        for k in ['ae', 'xgboost', 'transformer']: comp[k] = {'z': r['z'][k], 'contribution': None if r['contrib'] is None else r['contrib'][k]}
        comp['tgnn'] = None; hz = {}
        for h, (_, n) in HORIZONS.items():
            cw = self.cur[h]; pk = cw['peak'] if cw and np.isfinite(cw['peak']) else None
            hz[h] = {'window_start_utc': dt.datetime.fromtimestamp(cw['start'] * 60, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'peak_unusualness': None if pk is None else self._unus100(h, pk), 'tier': 'NORMAL' if pk is None else self._tier(h, pk)}
        u = r['unusualness']
        return {'symbol': self.sym, 'bar_open_utc': bar['bar_open_utc'], 'state': r['state'], 'unusualness': u, 'integrity_score': None if u is None else 100.0 - u, 'confidence': r['confidence'], 'fused_score': r['fusion'], 'components': comp, 'agreement': r['agreement'], 'horizons': hz,
                'reasons': reasons, 'caveats': caveats, 'warmup': r['warmup'], 'thresholds': {'alert': r['thr_alert'], 'watch': r['thr_watch']}, 'limitations_ref': LIMIT_REF}

    # ------------------------------------------------------------------ queries
    def episodes_view(self, since=None, limit=100):
        out = [{k: (sorted(v) if k == 'models_above' else v) for k, v in e.items() if not k.startswith('_')} for e in self.episodes]
        if since: out = [e for e in out if e['start_utc'] >= since]
        return list(reversed(out))[:limit]

    def day_view(self, date):
        ws = {h: [w for w in self.windows[h] if w['window_start_utc'].startswith(date)] for h in HORIZONS}
        worst = lambda lst: [{'start_utc': w['window_start_utc'], 'peak_unusualness': w['peak_unusualness']} for w in sorted([w for w in lst if w['peak_unusualness'] is not None], key=lambda w: -w['peak_unusualness'])[:3]]
        return {'symbol': self.sym, 'date': date, 'hourly_watch_windows': sum(w['tier'] != 'NORMAL' for w in ws['1h']), 'quarter_hour_alert_windows': sum(w['tier'] == 'ALERT' for w in ws['15m']), 'worst_hours': worst(ws['1h']), 'worst_quarter_hours': worst(ws['15m']), 'note': 'counts, not a daily score'}
