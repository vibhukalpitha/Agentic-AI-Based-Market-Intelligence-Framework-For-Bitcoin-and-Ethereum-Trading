"""live_api.py - LIVE source for the dashboard: reads the running engine API (/api/c03/..., engine contract v0.2) and builds the SAME per-symbol objects
that live_engine.LiveSymbol fills in replay mode, so every page (Overview, Evidence, Uncertainty, Similar Past) is drawn by the same code.
Nothing is computed here: every number (scores, thresholds, state, checks, reasons) comes from the engine. This module only reshapes it.
Needs the API of the engine to return per-bar detail (GET /bars?detail=true, engine v0.2.1). Without it the page falls back to what the plain v0.2 API gives
and says so (thresholds are then unknown for past minutes).
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests

SYMBOLS = ['BTCUSDT', 'ETHUSDT']
STATE_NUM = {'NORMAL': 0, 'WATCH': 1, 'ALERT': 2, 'NO_DATA': 0}
BITS = {'ae': 1, 'xgboost': 2, 'transformer': 4}
MAX_BARS = 2880
_C = {s: {'bars': {}, 'order': []} for s in SYMBOLS}       # server-side memory of the page: bars already downloaded
_X = {}                                                     # (symbol, ts) -> /explain detail or None
_SIM = {}                                                   # symbol -> (last_bar_ts, /similar answer)


class ApiDown(Exception):
    pass


def api_base():
    return os.environ.get('C03_API', 'http://127.0.0.1:8000').rstrip('/')


def get(path, timeout=8, **params):
    try:
        r = requests.get(api_base() + '/api/c03/' + path, params=params, timeout=timeout)
    except Exception as e:                                  # connection refused, timeout, DNS ...
        raise ApiDown(f'{type(e).__name__}: {e}')
    if r.status_code == 503:
        raise ApiDown('the engine is still warming up (no scored minute yet)')
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def health():
    return get('health', timeout=4)


def _iso_to_ts(t):
    return t.replace('T', ' ').replace('Z', '+00:00')       # same text form the replay uses


def poll(sym):
    """download the minutes the page does not have yet; returns the list of bars (oldest first)"""
    c = _C[sym]
    if not c['order']:
        new = get('bars', symbol=sym, limit=MAX_BARS, detail='true') or []
    else:
        new = get('bars', symbol=sym, limit=MAX_BARS, detail='true', **{'from': c['order'][-1]}) or []
        if new and new[0]['bar_open_utc'] < c['order'][-1]:                       # the server restarted and started a new history
            c['bars'], c['order'] = {}, []
    for b in new:
        t = b['bar_open_utc']
        if t not in c['bars']:
            c['order'].append(t)
        c['bars'][t] = b
    if len(c['order']) > MAX_BARS:
        for t in c['order'][:-MAX_BARS]:
            c['bars'].pop(t, None)
        c['order'] = c['order'][-MAX_BARS:]
    return [c['bars'][t] for t in c['order']]


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and x != x) else round(float(x), n)


class ApiHist:
    """what render_hist expects from the history index, answered by the engine API (/similar)"""

    def __init__(self, sym, ans):
        self.ok = bool(ans) and ans.get('status') == 'ok'
        self.status, self.ans, self.version = (ans or {}).get('status'), ans, {}
        self._tr = {m['end_ts_ns']: m.get('trace') for m in (ans or {}).get('matches', [])}

    def find(self, sym, d, q_ts_ns, k=5):
        return (self.ans or {}).get('matches', [])

    def trace(self, sym, end_ts_ns):
        t = self._tr.get(end_ts_ns)
        return None if not t else (np.asarray(t['fused'], 'float64'), np.asarray(t['alert_line'], 'float64'))


def similar(sym, ts_last):
    c = _SIM.get(sym)
    if c is None or c[0] != ts_last:
        _SIM[sym] = (ts_last, get('similar', symbol=sym, k=5))
    return _SIM[sym][1]


def explain_at(sym, ts):
    key = (sym, ts)
    if key not in _X:
        a = get('explain', symbol=sym, at=ts)
        _X[key] = None if not a else a.get('detail')
        if len(_X) > 3000:
            _X.pop(next(iter(_X)))
    return _X[key]


def _grid(bars):
    """one entry per UTC minute: a minute that never arrived (feed gap, server down) becomes a NO DATA minute with the reason, so index = minute offset like in the replay"""
    if not bars:
        return bars
    idx = {b['bar_open_utc']: b for b in bars}
    t0, t1 = pd.Timestamp(bars[0]['bar_open_utc']), pd.Timestamp(bars[-1]['bar_open_utc'])
    out = []
    for t in pd.date_range(t0, t1, freq='min'):
        k = t.strftime('%Y-%m-%dT%H:%M:%SZ')
        out.append(idx.get(k) or {'bar_open_utc': k, 'state': 'NO_DATA', 'features': None, 'close': None, 'warmup': False,
                                  'data_quality': {'ok': False, 'problems': [{'text': 'No minute arrived from the exchange feed for this time (data gap). The engine never scores an incomplete minute.'}]}})
    return out


def build(sym, bars, wins, cfg, cal=None, assets=None):
    """one LiveSymbol-compatible object from the downloaded minutes"""
    bars = _grid(bars)
    n = len(bars)
    nan = lambda: np.full(n, np.nan)
    ts = [_iso_to_ts(b['bar_open_utc']) for b in bars]
    dt = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_localize(None).to_numpy()
    has_x = n > 0 and all(('x' in b) for b in bars if b['state'] != 'NO_DATA')
    j = SimpleNamespace(symbol=sym, sym=sym.replace('USDT', '/USDT'), n=n, ts=ts, dt=dt, t0=int(pd.Timestamp(ts[0]).value // 60_000_000_000) if n else 0, done=n,
                        fusion=nan(), thr_alert=nan(), thr_watch=nan(), ae=nan(), xgb=nan(), tr=nan(), c_ae=nan(), c_xgb=nan(), c_tr=nan(), c_shared=nan(), plev=nan(), integ=nan(), unus_engine=nan(),
                        close=nan(), z1=np.full((n, 6), np.nan), zm=np.full((n, 3), np.nan), state=np.zeros(n, int), agree=np.full(n, -1), conf=[None] * n,
                        rel_ab=np.full((n, 3), -1), rel_lo=np.full((n, 3), -1), rel_level=np.zeros(n, int), rel_met=np.full(n, -1), rel_gate=np.zeros(n, int), rel_rare=np.full(n, -1), rel_pers=np.full(n, -1),
                        rel_nc=np.full(n, -1), rel_set=np.full(n, -1), rel_pabn=nan(), rel_rare_line=nan(), rel_norm_line=nan(), rel_hist=np.zeros(n, int),
                        xai={}, reasons={}, bad={}, wmap={'15m': {}, '1h': {}}, cal=cal, _ev={}, _evid={}, hidx=None, api=True, has_detail=has_x,
                        features=next((list(b['features']) for b in reversed(bars) if b.get('features')), []), bundle_version='live engine API (engine v0.2.1 per-bar detail)' if has_x else 'live engine API (plain v0.2)',
                        svc=SimpleNamespace(assets=assets), bandA=cfg['bandA'], bandW=cfg['bandW'], conf_labels=cfg.get('conf_labels'), thr_windows=cfg.get('thr_windows'),
                        alert_budget_per_day=cfg.get('alert_budget_per_day', 2))
    first = n
    rows = []
    for k, b in enumerate(bars):
        j.close[k] = np.nan if b.get('close') is None else b['close']
        f = b.get('features') or {}
        rows.append({'ts': ts[k], **{q: (np.nan if f.get(q) is None else f[q]) for q in j.features}})
        if b['state'] == 'NO_DATA':
            j.bad[k] = [p['text'] for p in (b.get('data_quality') or {}).get('problems', [])] or ['This minute failed the input checks.']
            continue
        if not b.get('warmup') and first == n:
            first = k
        fz, un = b.get('fused_score'), b.get('unusualness')
        j.fusion[k] = np.nan if fz is None else fz
        j.unus_engine[k] = np.nan if un is None else un
        j.integ[k] = np.nan if un is None else 100.0 - un
        j.state[k] = STATE_NUM[b['state']]
        j.conf[k] = b.get('confidence')
        sc = b.get('scores') or {}
        j.ae[k], j.xgb[k], j.tr[k] = (np.nan if sc.get(m) is None else sc[m] for m in ('ae', 'xgboost', 'transformer'))
        x = b.get('x')
        if x:
            j.thr_alert[k], j.thr_watch[k] = x['thr_alert'], x['thr_watch']
            j.plev[k] = np.nan if x.get('plevel') is None else x['plevel']
            if x.get('z'):
                j.zm[k] = [x['z']['ae'], x['z']['xgboost'], x['z']['transformer']]
            if x.get('contrib'):
                ct = x['contrib']
                j.c_ae[k], j.c_xgb[k], j.c_tr[k], j.c_shared[k] = ct['ae'], ct['xgboost'], ct['transformer'], ct['shared']
            j.agree[k] = -1 if x.get('models_above') is None else sum(BITS[m] for m in x['models_above'])
            if x.get('reasons'):
                j.reasons[k] = [{'type': q['type'], 'text': q['text'], 'weight': _r(q['weight'], 3), 'feature': q.get('feature')} for q in x['reasons']]
            rel = x.get('rel')
            if rel:
                c = rel['checks']
                j.rel_level[k], j.rel_met[k], j.rel_gate[k] = rel['level_code'], -1 if rel['checks_met'] is None else rel['checks_met'], int(rel['gate']['ok'])
                j.rel_rare[k], j.rel_pers[k], j.rel_nc[k] = [-1 if c[q] is None else int(c[q]) for q in ('rare', 'persistent', 'no_conflict')]
                j.rel_set[k] = rel['conformal']['set_code'] if rel['conformal']['available'] else -1
                j.rel_pabn[k] = np.nan if rel['conformal']['p_abnormal'] is None else rel['conformal']['p_abnormal']
                d = rel['detail']
                for q_, m_ in enumerate(cfg['models']):
                    md_ = (d.get('models') or {}).get(m_)
                    if md_:
                        j.rel_ab[k, q_], j.rel_lo[k, q_] = int(md_['above_own_alert_line']), int(md_['at_or_below_own_90th'])
                j.rel_rare_line[k] = np.nan if d.get('rare_line') is None else d['rare_line']
                j.rel_norm_line[k] = np.nan if d.get('normal_line') is None else d['normal_line']
                j.rel_hist[k] = rel['gate']['history_minutes']
        else:                                                                       # plain v0.2 API: only what the bar carries
            r = b.get('reliability') or {}
            j.rel_level[k], j.rel_met[k] = r.get('level_code', 0), -1 if r.get('checks_met') is None else r['checks_met']
    j.first = min(first, max(0, n - 1))
    j.df = pd.DataFrame(rows)
    # flagged-minute episodes (gap <= 5): same rule as the stored-bundle loader
    eps, o = [], None
    for k in range(j.first, n):
        if j.state[k] >= 1:
            if o and k - o[1] <= 5:
                o[1] = k
            else:
                o = [k, k]
                eps.append(o)
    j.eps = [tuple(e) for e in eps]
    j.alert_eps = [e for e in j.eps if (j.state[e[0]:e[1] + 1] == 2).any()]
    for h in ('15m', '1h'):
        for w in wins.get(h) or []:
            m = int(pd.Timestamp(w['window_start_utc']).value // 60_000_000_000)
            j.wmap[h][m] = {'start': w['window_start_utc'], 'end': w['window_end_utc'], 'peak_unus': _r(w['peak_unusualness'], 2), 'tier': w['tier'], 'flagged_min': w['flagged_minutes']}
    # detail of flagged minutes (SHAP / interventional Shapley / Transformer gradients): fetched once per minute, the engine keeps the last 300 flagged minutes
    fl = [k for k in range(j.first, n) if j.state[k] >= 1][-300:]
    todo = [k for k in fl if (sym, bars[k]['bar_open_utc']) not in _X][-40:]
    for k in todo:
        explain_at(sym, bars[k]['bar_open_utc'])
    for k in fl:
        d = _X.get((sym, bars[k]['bar_open_utc']))
        if d:
            j.xai[k] = d
    return j
