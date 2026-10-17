"""
TradeSense - Component 03 IntegritySense : Streamlit dashboard (Rashmi)

Run (from this folder):   streamlit run app.py

Everything is drawn natively with Streamlit + Plotly.
LIVE MODEL MODE (default when ../backend/c03_engine and ../backend/export exist): every replayed minute is passed through the
TRAINED models (Autoencoder + XGBoost + Transformer -> fusion -> state) by the same engine as the live API. Nothing is read from result files.
STORED-RESULTS MODE (fallback): reads data/BTCUSDT_dash.json and data/ETHUSDT_dash.json made by backend/tools/make_dashboard_bundle.py.

For the 4-tab platform (PP1):   from app import render ; render()   inside a tab.
It only DISPLAYS engine output. It never changes a model, feature, threshold or score.
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'data'
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
sys.path.insert(0, str(HERE))

COL = {0: '#10B981', 1: '#F59E0B', 2: '#EF4444'}
NAME = ['NORMAL', 'WATCH', 'ALERT']
LVC = {'STRONG': '#10B981', 'MODERATE': '#F59E0B', 'LIMITED': '#EF4444'}
CONFC = {'HIGH': '#EF4444', 'MEDIUM': '#FB923C', 'LOW': '#F59E0B'}
FPLAIN = {'log_returns': 'Price movement', 'trade_flow_imbalance': 'Buy/sell pressure', 'hl_spread_ratio': 'Price range',
          'rush_order_density': 'Rush of small trades', 'volume_acceleration': 'Volume surge', 'large_trade_count': 'Large trades'}
DET = [dict(c='c_ae', bit=1, name='Unfamiliar-behaviour check', tech='Autoencoder'),
       dict(c='c_xgb', bit=2, name='Known-pattern check', tech='XGBoost'),
       dict(c='c_tr', bit=4, name='Multi-minute pattern check', tech='Transformer')]
ACC, TXT, DIM, GRID = '#8B93FF', '#E8ECF5', '#8E98B0', '#1B2640'


# ------------------------------------------------------------------ data
def fname(f):
    return FPLAIN.get(f, str(f).replace('_', ' '))


def fx(v, d=2):
    return '–' if v is None or not np.isfinite(v) else f'{v:.{d}f}'


def _f(lst):
    return np.array(lst, dtype=float)


@st.cache_resource(show_spinner='Loading real engine results ...')
def load_bundle(path, mtime):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'unus_engine' not in d or 'ts' not in d:
        raise ValueError(f'{path} is not a dashboard bundle')
    j = SimpleNamespace(**{k: d.get(k) for k in d})
    j.symbol = d['symbol']
    j.sym = j.symbol.replace('USDT', '/USDT')
    j.n = len(d['ts'])
    j.first = int(d.get('bundle_from_bar') or 0)
    j.dt = pd.to_datetime(pd.Series(d['ts']), utc=True).dt.tz_localize(None).to_numpy()
    j.t0 = int(pd.Timestamp(d['ts'][0]).value // 60_000_000_000)
    for k in ['close', 'fusion', 'thr_alert', 'thr_watch', 'ae', 'xgb', 'tr', 'c_ae', 'c_xgb', 'c_tr', 'plev']:
        setattr(j, k, _f(d[k]))
    j.state = np.array(d['state'], dtype=int)
    j.agree = np.array([-1 if v is None else int(v) for v in d['agree']])
    j.integ = 100.0 - _f(d['unus_engine'])
    j.conf = d['conf']
    j.xai = {int(k): v for k, v in (d.get('xai') or {}).items()}
    j.reasons = {int(k): v for k, v in (d.get('reasons') or {}).items()}
    j.features = d.get('feature_names') or list(FPLAIN)
    ok = ~np.isnan(j.integ)
    st_ = j.state
    a = j.integ[(st_ == 2) & ok & (np.arange(j.n) >= j.first)]
    w = j.integ[(st_ >= 1) & ok & (np.arange(j.n) >= j.first)]
    j.bandA = float(a.max()) if len(a) else 0.0
    j.bandW = max(float(w.max()) if len(w) else 0.0, j.bandA)
    eps, o = [], None
    for k in range(j.first, j.n):
        if st_[k] >= 1:
            if o and k - o[1] <= 5:
                o[1] = k
            else:
                o = [k, k]
                eps.append(o)
    j.eps = [tuple(e) for e in eps]
    j.wmap = {}
    for h in ('15m', '1h'):
        j.wmap[h] = {int(pd.Timestamp(w_['start']).value // 60_000_000_000): w_ for w_ in ((d.get('windows') or {}).get(h) or [])}
    j.cal = d.get('calibration')
    j._ev, j._evid = {}, {}
    j.alert_eps = [e for e in j.eps if (st_[e[0]:e[1] + 1] == 2).any()]
    return j


@st.cache_resource(show_spinner=False)
def load_export(root, backend):
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from c03_engine.store import Export
    return Export(root, verify=True)


def load_calibration():
    p = DATA_DIR / 'calibration_cal3.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def engine_paths():
    """(backend, export) when the trained-model engine can be used here, else None"""
    if st.session_state.get('_data_dir') or __import__('os').environ.get('C03_MODE') == 'stored':
        return None
    import live_engine as LE
    found, why = LE.find_backend(HERE)
    return found


def live_mode():
    return engine_paths() is not None


def bundles():
    paths = engine_paths()
    if paths is not None:
        if '_LB' not in st.session_state:
            import live_engine as LE
            ex = load_export(str(paths[1]), str(paths[0]))
            cal = load_calibration()
            with st.spinner('Loading the trained models and filling their rolling windows (about 10 s) ...'):
                st.session_state['_LB'] = {s: LE.make_live(ex, s, cal.get(s)) for s in SYMBOLS}
                st.session_state['_EX'] = ex
        return st.session_state['_LB'], []
    out = {}
    miss = []
    for s in SYMBOLS:
        p = Path(st.session_state.get('_data_dir', DATA_DIR)) / f'{s}_dash.json'
        if not p.exists():
            miss.append(str(p))
        else:
            out[s] = load_bundle(str(p), p.stat().st_mtime)
    return out, miss


# ------------------------------------------------------------------ pattern resemblance (protocol G1 v1)
@st.cache_resource
def load_resem(path, mtime):
    m = json.loads(Path(path).read_text(encoding='utf-8'))
    m['_mean'], m['_scale'] = np.array(m['scaler_mean']), np.array(m['scaler_scale'])
    m['_coef'], m['_int'] = np.array(m['coef']), np.array(m['intercept'])
    return m


def resem_model():
    p = Path(st.session_state.get('_data_dir', DATA_DIR)) / 'declared_pattern_resemblance_v1.json'
    return load_resem(str(p), p.stat().st_mtime) if p.exists() else None


def resem_of(v):
    """Past-only: uses only the episode bars seen so far (v.s..v.e). Returns None / dict(status=...)."""
    m = resem_model()
    if m is None or not m.get('show_closest'):
        return None
    j = v.j
    key = ('resem', v.s, v.e)
    if key in j._ev:
        return j._ev[key]
    zs = [j.xai[k]['z'] for k in range(v.s, v.e + 1) if k in j.xai and j.xai[k].get('z')]
    out = {'status': 'short'}
    if v.len >= 3 and len(zs) >= 3:
        zs = np.array(zs, float)
        sig = np.concatenate([np.abs(zs).max(0), zs.mean(0), [np.log(v.len)]])
        lo = m['_coef'] @ ((sig - m['_mean']) / m['_scale']) + m['_int']
        p = np.exp(lo - lo.max())
        p /= p.sum()
        o = np.argsort(-p)
        nm = m['plain_names']
        cl = m['classes']
        out = {'status': 'ok', 'first': nm[str(cl[o[0]])], 'shortlist2': nm[str(cl[o[1]])]}
    j._ev[key] = out
    return out


FAM_DESC = {
    'Pump-and-dump-like surge': 'A sharp price push with a surge in volume and large trades, as in a pump-and-dump set-up.',
    'Fake-breakout-like move': 'A sudden price push with a wide range and heavy activity, like a breakout that may not hold.',
    'Abnormal volume burst': 'A sudden burst of volume with large trades and a rush of small trades.',
    'One-sided trade flow': 'Buyers (or sellers) dominate the executed volume while the price moves.',
    'Wash-like trading (proxy)': 'Many small trades per unit of volume with a wide range and uneven flow, as in artificial-activity (wash-like) trading.',
    'Liquidity-stress-like (proxy)': 'A very wide price range with price movement and large trades, as when liquidity thins out.'}


def resem_card(v):
    m = resem_model()
    j, k = v.j, v.eng
    ev = evid(v)
    x = j.xai.get(v.xb) if v.xb is not None else None
    with st.container(border=True):
        st.markdown('**Detected behaviour & evidence** ' + (badge('INDICATIVE', ACC) if m and m.get('show_closest') else badge('COMING SOON', DIM)), unsafe_allow_html=True)
        L, R = st.columns([1.15, 1.35])
        with L:
            if not m or not m.get('show_closest'):
                st.caption('Pattern resemblance is not available yet.')
            elif v.mx < 2:
                st.caption('Shown only for ALERT episodes. This one is WATCH only, so it is not rated.')
            else:
                r = resem_of(v)
                if r['status'] == 'ok':
                    st.markdown(f"<div style='font-size:12px;letter-spacing:1px;color:{DIM}'>THIS BEHAVIOUR RESEMBLES</div>"
                                f"<div style='font-size:30px;font-weight:700;color:#FB923C;line-height:1.2'>{r['first']}</div>", unsafe_allow_html=True)
                    st.markdown(FAM_DESC.get(r['first'], ''))
                    st.caption('Closest patterns (shortlist): ' + r['first'] + ' · ' + r['shortlist2'])
                elif v.full - v.s + 1 < 3:
                    st.caption('This episode lasted under 3 minutes, which is too short to compare with known patterns.')
                else:
                    st.caption('Still too short to compare with known patterns. It needs at least 3 minutes; keep playing.')
            t1, t2, t3 = st.columns(3)
            t1.markdown(f"<div style='font-size:11px;letter-spacing:1px;color:{DIM}'>ALERT LEVEL</div><div style='font-size:18px;font-weight:700;color:{COL[v.mx]}'>{lvl_txt(v)}</div>", unsafe_allow_html=True)
            cf = ev.conf if v.mx == 2 and ev.conf else '–'
            t2.markdown(f"<div style='font-size:11px;letter-spacing:1px;color:{DIM}'>ENGINE CONFIDENCE</div><div style='font-size:18px;font-weight:700;color:{CONFC.get(cf, DIM)}'>{cf}</div>", unsafe_allow_html=True)
            lv = ev.level or 'NOT RATED'
            t3.markdown(f"<div style='font-size:11px;letter-spacing:1px;color:{DIM}'>EVIDENCE</div><div style='font-size:18px;font-weight:700'>{lv}</div>", unsafe_allow_html=True)
            rj = (m or {}).get('results', {}).get('JITTERED')
            acc = (f"In synthetic tests (imperfect episode edges) the first guess was right {rj['top1'] * 100:.0f}% of the time and the right pattern was inside the shortlist {rj['top2'] * 100:.0f}% of the time. "
                   if rj else '')
            st.caption(acc + 'It is a resemblance, not proof of manipulation, and says nothing about price direction. A genuinely new pattern can look like an older one.')
        with R:
            st.markdown('**What drove it** ' + badge('INDICATIVE', ACC), unsafe_allow_html=True)
            if x and x.get('shap'):
                st.plotly_chart(hbar_fig(x['shap'], [fname(f) for f in j.features], True), width='stretch', config=dict(displayModeBar=False), key='x_drv')
            else:
                st.caption('Detailed signal data was not captured for this episode.')


# ------------------------------------------------------------------ provenance (everything here is read from the files, nothing typed in)
@st.cache_resource
def file_sha(path, mtime):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def render_provenance(B):
    d = Path(st.session_state.get('_data_dir', DATA_DIR))
    with st.expander('Where do these numbers come from? (data & model provenance)'):
        if live_mode():
            ex = st.session_state.get('_EX')
            st.markdown('**Trained models run live in this session.** Each replayed minute (six recorded Layer-1 features) is passed through the frozen engine: '
                        'Autoencoder (3 seeds) + XGBoost (3 seeds) + Transformer (3 seeds) -> fusion v0 -> rolling 7-day thresholds -> state, confidence, reasons and SHAP. '
                        'No result file is read. Only the recorded input bars and the declared model files are used. A live Binance feed would use the same engine; this page replays recorded bars.')
            if ex is not None:
                man = ex.manifest
                st.markdown(f"Engine export: {len(man.get('files', {}))} files, every sha256 checked against the manifest at start-up (library sha {man.get('library_sha16', 'n/a')}).")
            rows = []
            for sym, jj in B.items():
                pr = jj.parity()
                rows.append({'market': sym, 'minutes scored by the models so far': jj.done - jj.first, 'flagged episodes': len(jj.eps), 'ALERT episodes': len(jj.alert_eps),
                             'fused score vs notebook reference (max abs diff)': None if not pr or pr['fusion_max_abs_diff'] is None else round(pr['fusion_max_abs_diff'], 5),
                             'state differs from notebook (minutes)': None if not pr else pr['state_differs']})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            st.caption('The last two columns compare this live run with the reference scores the research notebooks stored for the same minutes. Small differences come from the rolling 7-day window still filling at the start of the recording.')
        else:
            st.markdown('Every number, chart and label on this page is read from the files below. Nothing is typed in or mocked. '
                        'The numbers were produced offline by running the **frozen engine** over recorded Binance 1-minute bars and saving the result; this page **replays** that result minute by minute. '
                        'It is not a live feed. (Place the backend folder next to this one to switch to live trained-model mode.)')
            rows = []
            for sym, jj in B.items():
                p = d / f'{sym}_dash.json'
                sc = jj.n - jj.first
                rows.append({'file': p.name, 'sha256 (first 16)': file_sha(str(p), p.stat().st_mtime), 'size MB': round(p.stat().st_size / 1e6, 2),
                             'bundle version': jj.bundle_version, 'bars in file': jj.n, 'bars scored': sc, 'from': jj.ts[jj.first][:16], 'to': jj.ts[jj.n - 1][:16],
                             'flagged episodes': len(jj.eps), 'ALERT episodes': len(jj.alert_eps), 'bars with detail': len(jj.xai)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        m = resem_model()
        if m:
            h = m.get('hypotheses', {})
            st.markdown(f"**Pattern-resemblance layer:** protocol G1 v1 (sha256 {m.get('protocol_g1_sha16')}), declared {m.get('declared_utc')}. "
                        f"Pre-registered checks: " + ' · '.join(f"{k.replace('_', '-')} {'held' if v else 'FAILED'}" for k, v in h.items()) +
                        f". Closest-pattern output shown: {'yes' if m.get('show_closest') else 'no'}; no-close-match output shown: {'yes' if m.get('show_no_match') else 'no (failed its test, so it is not shown)'}.")
        else:
            st.markdown('**Pattern-resemblance layer:** file not found, so that output is not shown.')
        jj = next(iter(B.values()))
        cal = jj.cal
        if cal:
            st.markdown(f"**Calibration (uncertainty tab):** {cal['n_normal']:,} normal and {cal['n_abnormal']:,} abnormal minutes from the calibration split, read from the bundle.")
        st.markdown('**Method constants written in the protocols (not data):** evidence rule = rarer than 1 in 1,000, at least 3 of the last 5 minutes ALERT, at least 2 of 3 checks above their own line; '
                    'an episode merges flagged minutes with gaps of at most 5 minutes; a pattern label needs at least 3 minutes. Detector names are plain-language labels for the autoencoder, XGBoost and Transformer.')
        st.caption('Check yourself: run `python verify_dynamic.py` in this folder. It checks the engine against the notebook reference scores and proves that the page follows the models.')


# ------------------------------------------------------------------ episode logic
def epview(j, ep, at):
    s, e_full = ep
    e = min(e_full, at)
    key = (s, e)
    if key in j._ev:
        return j._ev[key]
    pk, xb, mx = s, None, 0
    for k in range(s, e + 1):
        mx = max(mx, int(j.state[k]))
        if j.fusion[k] > j.fusion[pk]:
            pk = k
        if k in j.xai and (xb is None or j.fusion[k] > j.fusion[xb]):
            xb = k
    v = SimpleNamespace(j=j, s=s, e=e, full=e_full, pk=pk, xb=xb, eng=(xb if xb is not None else pk), mx=mx, len=e - s + 1)
    j._ev[key] = v
    return v


def vis_eps(j, at):
    return [epview(j, ep, at) for ep in j.eps if ep[0] <= at]


def evid(v):
    j, k = v.j, v.eng
    key = (v.s, v.e)
    if key in j._evid:
        return j._evid[key]
    pl = None if np.isnan(j.plev[k]) else float(j.plev[k])
    ag = None if j.agree[k] < 0 else int(j.agree[k])
    cnt = int(sum(1 for q in range(max(0, k - 4), k + 1) if j.state[q] == 2))
    nA = None if ag is None else bin(ag).count('1')
    rare, pers, agr = pl is not None and pl <= 0.001, cnt >= 3, nA is not None and nA >= 2
    met = sum([rare, pers, agr])
    r = SimpleNamespace(k=k, pl=pl, ag=ag, nA=nA, cnt=cnt, rare=rare, pers=pers, agr=agr, met=met, conf=j.conf[k],
                        level=None if v.mx < 2 else ('STRONG' if met == 3 else 'MODERATE' if met == 2 else 'LIMITED'))
    j._evid[key] = r
    return r


def lead_det(j, k):
    best, bv = None, -1e9
    for d in DET:
        x = getattr(j, d['c'])[k]
        if np.isfinite(x) and x > bv:
            best, bv = d, x
    return best


def is_market_wide(v, at, B):
    o = next((x for x in B.values() if x is not v.j), None)
    if o is None:
        return False
    lo, hi = max(0, v.s - 10), min(v.e + 10, at, o.n - 1)
    return bool((o.state[lo:hi + 1] >= 1).any())


def lvl_txt(v):
    return 'ALERT' if v.mx == 2 else 'WATCH'


# ------------------------------------------------------------------ state helpers
def ss():
    return st.session_state


def set_i(k, pause=True, toast=None):
    j = CUR()
    ss().i = int(max(j.first, min(j.n - 1, k)))
    ss().ver = ss().get('ver', 0) + 1
    if pause:
        ss().playing = False
    if toast:
        ss().toast = toast


def CUR():
    return ss()._B[ss().sym]


def hhmm(j, k):
    return j.ts[k][11:16]


def tstr(j, k):
    return j.ts[k][:16] + ' UTC'


def jump_alert(d):
    j, i = CUR(), ss().i
    E = j.alert_eps
    t = next((e for e in E if e[0] - 30 > i), None) if d > 0 else next((e for e in reversed(E) if e[0] - 30 < i), None)
    if t is None and d > 0 and getattr(j, 'done', j.n) < j.n:
        ss().goal = 'next'            # the models have not scored that far yet: run them forward (handled in render, with a progress bar)
        return
    if t:
        ss().sel = None
        set_i(max(j.first, t[0] - 30), toast=f"{'Next' if d > 0 else 'Previous'} alert episode starts at {hhmm(j, t[0])} UTC ({j.ts[t[0]][5:10]})")
    else:
        ss().toast = 'No further alert in that direction'


def jump_aug():
    j = CUR()
    e = next((e for e in j.alert_eps if j.ts[e[0]][:10] == '2024-08-05'), None)
    if e:
        ss().sel = None
        set_i(max(j.first, e[0] - 30), toast=f'Aug 5 stress day: first alert at {hhmm(j, e[0])} UTC')
    elif getattr(j, 'done', j.n) < j.n:
        ss().goal = 'aug5'


def on_seek():
    ss().i = int(ss()[f'seek{ss().ver}'])
    ss().sel = None
    ss().playing = False


def toggle_play():
    j = CUR()
    if ss().playing:
        ss().playing = False
    else:
        if ss().i >= j.n - 1:
            ss().i = j.first
            ss().ver += 1
        ss().playing = True


def open_episode(sym, s, k):
    ss().sym_sel = sym.replace('USDT', '/USDT')
    ss().sym = sym
    ss().sel = s
    set_i(k)


# ------------------------------------------------------------------ plotly helpers
def base_fig(h, left=56):
    f = go.Figure()
    f.update_layout(template='plotly_dark', height=h, margin=dict(l=left, r=10, t=8, b=26), paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', showlegend=False, font=dict(size=11, color=DIM), hovermode='x unified',
                    xaxis=dict(gridcolor=GRID, showline=False), yaxis=dict(gridcolor=GRID, zeroline=False))
    return f


def gp(s):
    return (max(0.0, min(100.0, s)) / 100.0) ** 0.35


def gauge_fig(score, j, color, h=230):
    pa, pw = gp(j.bandA) * 100, gp(j.bandW) * 100
    val = gp(100 if score is None or np.isnan(score) else score) * 100
    f = go.Figure(go.Indicator(mode='gauge', value=val, gauge=dict(
        shape='angular', axis=dict(range=[0, 100], tickvals=[0, gp(1) * 100, gp(5) * 100, gp(25) * 100, gp(50) * 100, 100],
                                   ticktext=['0', '1', '5', '25', '50', '100'], tickcolor=DIM, tickfont=dict(size=10, color=DIM)),
        bar=dict(color='rgba(0,0,0,0)', thickness=0.0),
        steps=[dict(range=[0, pa], color=COL[2]), dict(range=[pa, pw], color=COL[1]), dict(range=[pw, 100], color=COL[0])],
        threshold=dict(line=dict(color='#FFFFFF', width=5), thickness=0.95, value=val), borderwidth=0, bgcolor='rgba(0,0,0,0)')))
    f.update_layout(height=h, margin=dict(l=24, r=24, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', font=dict(color=DIM))
    return f


def ring_fig(score, color, h=130):
    p = gp(100 if score is None or np.isnan(score) else score)
    f = go.Figure(go.Pie(values=[p, 1 - p], hole=0.78, marker=dict(colors=[color, GRID]), textinfo='none', sort=False,
                         direction='clockwise', hoverinfo='skip'))
    f.add_annotation(text=f"<b>{'–' if score is None or np.isnan(score) else (f'{score:.1f}' if score < 10 else f'{score:.0f}')}</b><br><span style='font-size:10px;color:{DIM}'>/ 100</span>",
                     showarrow=False, font=dict(size=22, color=TXT))
    f.update_layout(height=h, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return f


def hbar_fig(vals, names, signed=False, color='#6366F1'):
    vals = np.array(vals, dtype=float)
    idx = np.argsort(-np.abs(vals))[::-1]
    cols = [('#8B5CF6' if vals[q] >= 0 else '#38BDF8') if signed else color for q in idx]
    f = go.Figure(go.Bar(x=vals[idx], y=[names[q] for q in idx], orientation='h', marker_color=cols,
                         text=[(f'{vals[q]:+.2f}' if signed else f'{vals[q]:.2f}') for q in idx], textposition='outside',
                         cliponaxis=False, hoverinfo='skip'))
    f.update_layout(template='plotly_dark', height=34 * len(vals) + 40, margin=dict(l=4, r=40, t=6, b=6), paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(visible=False), yaxis=dict(gridcolor=GRID), font=dict(size=12, color=TXT))
    return f


def pill(state, txt=None):
    c = COL[state]
    return f'<span style="border:1px solid {c};color:{c};background:{c}22;border-radius:99px;padding:3px 12px;font-size:12.5px;font-weight:600">● {txt or NAME[state]}</span>'


def badge(txt, c):
    return f'<span style="border:1px solid {c}88;color:{c};background:{c}1A;border-radius:6px;padding:2px 8px;font-size:11px;font-weight:700;letter-spacing:.5px">{txt}</span>'


def label(t):
    st.markdown(f'<div style="font-size:11px;letter-spacing:1px;color:{DIM};text-transform:uppercase;font-weight:600">{t}</div>', unsafe_allow_html=True)


def score_hint(j, sc, st_):
    """one plain sentence that explains why a given Integrity Score is (or is not) flagged"""
    if sc is None or np.isnan(sc):
        return ''
    if st_ == 2:
        return f'ALERT zone: the score is down at {sc:.1f}. ALERT starts at roughly {j.bandA:.1f} or lower.'
    if st_ == 1:
        return f'WATCH zone: the score is down at {sc:.1f}. WATCH starts at roughly {j.bandW:.1f} or lower.'
    if sc >= 50:
        return 'Ordinary minute: more usual than most of the last 7 days.'
    share = float(np.nanmean(j.integ[j.first:] <= sc))
    return (f'This minute is more unusual than {100 - sc:.0f}% of the last 7 days, but {share * 100:.0f}% of the minutes in this recording are at least that unusual, so it is still normal. '
            f'WATCH starts only at roughly {j.bandW:.1f} or lower, ALERT at {j.bandA:.1f} or lower.')


def story(j, k, st_):
    f = j.fusion[k]
    if st_ == 0:
        return f'At **{hhmm(j, k)} UTC**, no WATCH or ALERT. The fused score ({fx(f)}) is below the watch line ({fx(j.thr_watch[k])}). ' + score_hint(j, j.integ[k], 0)
    lv, thr = ('ALERT', j.thr_alert[k]) if st_ == 2 else ('WATCH', j.thr_watch[k])
    rs = j.reasons.get(k) or []
    F = next((x for x in rs if x['type'] == 'feature'), None)
    why = f" Strongest signal: **{fname(F['feature'])}**." if F else ''
    if not F:
        d = lead_det(j, k)
        why = f' Lead signal: **{d["name"]}**.' if d else ''
    return f'At **{hhmm(j, k)} UTC** the fused score ({fx(f)}) is above the **{lv}** line ({fx(thr)}).{why} An attention signal, not proof of wrongdoing.'


# ------------------------------------------------------------------ OVERVIEW
def buckets(j, at, tfv, zm):
    zm = max(zm, tfv * 60)
    a = max(j.first, at - zm + 1)
    out = []
    for b in range(a // tfv, at // tfv + 1):
        s, e = max(a, b * tfv), min(at, b * tfv + tfv - 1)
        if e < s:
            continue
        seg = j.integ[s:e + 1]
        seg = seg[~np.isnan(seg)]
        out.append((s, e, j.close[e], float(seg.min()) if len(seg) else None, int(j.state[s:e + 1].max())))
    return out


def shade_runs(f, j, B):
    run = None
    for (s, e, c, g, st_) in B + [(0, 0, 0, 0, 0)]:
        if st_ > 0 and run is None:
            run = [s, e, st_]
        elif st_ > 0 and run[2] == st_:
            run[1] = e
        else:
            if run:
                f.add_vrect(x0=j.dt[run[0]], x1=j.dt[run[1]] + np.timedelta64(1, 'm') * 1, fillcolor=COL[run[2]], opacity=0.22, line_width=0)
            run = [s, e, st_] if st_ > 0 else None


def ov_charts(j, i):
    c1, c2 = st.columns([1.2, 1])
    with c1:
        tf = st.segmented_control('Timeframe', ['1 min', '15 min', '1 hour'], default='1 min', key='tf')
    with c2:
        zm = st.segmented_control('Zoom', ['3h', '12h', '24h', '3d'], default='24h', key='zm')
    tfv = {'1 min': 1, '15 min': 15, '1 hour': 60}[tf or '1 min']
    zv = {'3h': 180, '12h': 720, '24h': 1440, '3d': 4320}[zm or '24h']
    B = buckets(j, i, tfv, zv)
    x = [j.dt[b[1]] for b in B]
    f = base_fig(240, 64)
    shade_runs(f, j, B)
    f.add_trace(go.Scatter(x=x, y=[b[2] for b in B], mode='lines', line=dict(color=ACC, width=1.8), name='Price', hovertemplate='$%{y:,.0f}'))
    f.add_vline(x=x[-1], line=dict(color='#FFFFFF', dash='dot', width=1))
    f.update_yaxes(tickprefix='$', tickformat=',.0f')
    st.markdown('<span style="font-size:11px;color:#8E98B0;letter-spacing:1px">PRICE (USD) · AMBER = WATCH · RED = ALERT</span>', unsafe_allow_html=True)
    st.plotly_chart(f, width='stretch', config=dict(displayModeBar=False), key='ov_price')
    g = base_fig(150, 64)
    shade_runs(g, j, B)
    g.add_trace(go.Scatter(x=x, y=[100 if b[3] is None else b[3] for b in B], mode='lines', line=dict(color=COL[0], width=1.6),
                           fill='tozeroy', fillcolor='rgba(16,185,129,.12)', name='Integrity Score', hovertemplate='%{y:.1f}'))
    g.update_yaxes(range=[0, 100])
    st.markdown('<span style="font-size:11px;color:#8E98B0;letter-spacing:1px">INTEGRITY SCORE (100 = COMPLETELY ORDINARY)</span>', unsafe_allow_html=True)
    st.plotly_chart(g, width='stretch', config=dict(displayModeBar=False), key='ov_score')


def day_info(j, i):
    d = j.ts[i][:10]
    a = i
    while a > j.first and j.ts[a - 1][:10] == d:
        a -= 1
    z = i
    while z < j.n - 1 and j.ts[z + 1][:10] == d:
        z += 1
    return d, a, z


def day_summary(j, i, B):
    d, a, z = day_info(j, i)
    done = i >= z
    vis = [v for v in vis_eps(j, i) if v.e >= a and j.ts[v.s][:10] == d]
    al = [v for v in vis if v.mx == 2]
    wo = [v for v in vis if v.mx == 1]
    seg = slice(a, i + 1)
    cl = j.close[seg]
    hi, lo = float(cl.max()), float(cl.min())
    integ = j.integ[seg]
    mk = a + int(np.nanargmin(integ)) if np.isfinite(integ).any() else a
    mn = float(np.nanmin(integ)) if np.isfinite(integ).any() else None
    nb = i - a + 1
    fl = int((j.state[seg] >= 1).sum())
    calm = (1 - fl / nb) * 100
    ch = (j.close[i] / j.close[a] - 1) * 100
    t0 = 'Today' if done else 'So far today'
    big = abs(ch) >= 3 or (hi - lo) / j.close[a] >= 0.05
    lowsc = mn is not None and mn < 20
    busy = not al and not wo and nb >= 30 and (big or lowsc)
    if al:
        head, hc = f"{t0}: {len(al)} unusual stretch{'es' if len(al) > 1 else ''} spotted", COL[2]
    elif wo:
        head, hc = f'{t0}: slightly elevated activity', COL[1]
    elif busy:
        head, hc = f'{t0}: busy market, no alert raised', ACC
    else:
        head, hc = f'{t0}: calm market', COL[0]
    rows = []
    if al:
        f1 = al[0]
        F = next((x for x in (j.reasons.get(f1.eng) or []) if x['type'] == 'feature'), None)
        rows.append(f"The first strong unusual moment began at **{hhmm(j, f1.s)} UTC**" + (f", mainly because of **{fname(F['feature']).lower()}**." if F else '.'))
        rows.append(f'The rest of the time (**{calm:.0f}%**) looked normal.')
        mw = sum(1 for v in al if is_market_wide(v, i, B))
        if mw:
            rows.append(f"{'All of them' if mw == len(al) and mw > 1 else str(mw) + ' of them'} also showed up on the other coin, so it looks market-wide.")
    elif wo:
        rows.append(f"There {'were ' + str(len(wo)) + ' mildly unusual stretches' if len(wo) > 1 else 'was 1 mildly unusual stretch'}, none strong enough for an ALERT.")
    elif busy:
        rows.append(f"Trading was quite unusual compared with the last 7 days (score as low as **{mn:.1f}** of 100), but it never reached the alert line." if lowsc
                    else 'Trading stayed below the alert line.')
    else:
        rows.append('Trading behaved normally.')
    if nb >= 30:
        rows.append(f"Price {'rose' if ch >= 0 else 'fell'} **{abs(ch):.1f}%** since the day's start." + (' The system watches how trading behaves, not price direction alone.' if busy and big else ''))
    with st.container(border=True):
        st.markdown('#### Summary of the day')
        st.caption(f"{d} UTC · {j.sym}" + ('' if done else f' · up to {hhmm(j, i)} UTC'))
        st.markdown(f'<div style="font-size:17px;font-weight:600"><span style="color:{hc}">●</span> {head}</div>', unsafe_allow_html=True)
        st.markdown('\n'.join('- ' + r for r in rows))
        c = st.columns(4)
        c[0].metric('Strong unusual stretches', len(al), f'first at {hhmm(j, al[0].s)} UTC' if al else 'none so far', delta_color='off')
        c[1].metric('Mild stretches', len(wo))
        c[2].metric('Most unusual minute', f'{hhmm(j, mk)} UTC', f'score {mn:.1f} of 100' if mn is not None else '', delta_color='off')
        c[3].metric('Price range', f'\\${lo:,.0f} – \\${hi:,.0f}')
        st.caption('An attention signal about market behaviour. Not a forecast, and not proof of wrongdoing.')


def render_overview(B, j, i):
    st_ = int(j.state[i])
    sc = j.integ[i]
    c0, c1, c2 = st.columns([1, 2.1, 1])
    with c0:
        with st.container(border=True):
            st.markdown('#### Integrity Score')
            st.plotly_chart(gauge_fig(sc, j, COL[st_]), width='stretch', config=dict(displayModeBar=False), key='gauge')
            sctxt = '–' if np.isnan(sc) else (f'{sc:.1f}' if sc < 10 else f'{sc:.0f}')
            st.markdown(f'<div style="text-align:center;margin-top:-60px"><span style="font-size:42px;font-weight:700">{sctxt}</span> <span style="color:{DIM}">/ 100</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;margin:10px 0 14px">{pill(st_, NAME[st_] + " · " + ["Typical market behaviour", "Elevated activity", "Unusual behaviour"][st_])}</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            a.metric('Unusualness', 'n/a' if np.isnan(sc) else f'{100 - sc:.2f}%')
            pl = j.plev[i]
            b.metric('Rarity (p-level)', 'n/a' if np.isnan(pl) else ('<0.001' if pl < 0.001 else f'{pl:.3f}'))
            a.metric('Fused score', fx(j.fusion[i]))
            b.metric('Alert threshold', fx(j.thr_alert[i]))
            st.markdown(f'<div style="font-size:12.5px;color:{TXT};line-height:1.5;margin-bottom:6px">{score_hint(j, sc, st_)}</div>', unsafe_allow_html=True)
            st.caption(f'Zones on the gauge: ALERT ≤ {j.bandA:.1f} · WATCH ≤ {j.bandW:.1f} · NORMAL above. Score = 100 − unusualness percentile versus the last 7 days (the gauge is stretched near 0).')
    with c1:
        with st.container(border=True):
            st.markdown('#### Price & Detected Windows')
            b0 = j.close[max(j.first, i - 15)]
            dl = (j.close[i] / b0 - 1) * 100
            st.markdown(f'<span style="font-size:32px;font-weight:700">${j.close[i]:,.2f}</span> &nbsp; <span style="color:{"#10B981" if dl >= 0 else "#EF4444"};font-weight:600">{dl:+.2f}% / 15m</span>', unsafe_allow_html=True)
            st.info(story(j, i, st_))
            ov_charts(j, i)
    with c2:
        with st.container(border=True):
            st.markdown('#### Replay Summary')
            c = [int((j.state[j.first:i + 1] == q).sum()) for q in range(3)]
            tot = max(1, sum(c))
            vis = vis_eps(j, i)
            nal = sum(1 for v in vis if v.mx == 2)
            st.caption(f'{(i - j.first + 1) / 1440:.1f} days replayed')
            f = go.Figure(go.Pie(values=[max(x, 0.012 * tot) if q and x else x for q, x in enumerate(c)], hole=0.75, sort=False,
                                 marker=dict(colors=[COL[0], COL[1], COL[2]]), textinfo='none', hoverinfo='skip'))
            f.add_annotation(text=f'<b>{nal}</b><br><span style="font-size:9px;color:{DIM}">ALERT EPISODES</span>', showarrow=False, font=dict(size=30, color=TXT))
            f.update_layout(height=190, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(f, width='stretch', config=dict(displayModeBar=False), key='donut')
            for q in range(3):
                st.markdown(f'<div style="display:flex;justify-content:space-between"><span><span style="color:{COL[q]}">■</span> {NAME[q]}</span><b>{c[q] / tot * 100:.1f}%</b></div>', unsafe_allow_html=True)
            per = {}
            for v in vis:
                if v.mx == 2:
                    per[j.ts[v.s][:10]] = per.get(j.ts[v.s][:10], 0) + 1
            days = []
            for k in range(j.first, i + 1, 1440):
                days.append(j.ts[k][:10])
            if j.ts[i][:10] not in days:
                days.append(j.ts[i][:10])
            avg = nal / max(1, (i - j.first + 1) / 1440)
            bud = j.alert_budget_per_day or 2
            st.divider()
            st.markdown(f'<div style="font-size:12px;color:{DIM}">Daily alert rate (budget {bud:.1f}) &nbsp; <b style="color:{"#F59E0B" if avg > bud else "#10B981"}">avg {avg:.2f}/d</b></div>', unsafe_allow_html=True)
            for d in days[-8:]:
                n = per.get(d, 0)
                st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:12px;color:{"#F59E0B" if n >= 5 else DIM}"><span>{d}{" (BUSY)" if n >= 5 else ""}</span><span>{"▮" * min(n, 12)} {n}</span></div>', unsafe_allow_html=True)
    day_summary(j, i, B)
    with st.container(border=True):
        h1, h2 = st.columns([3, 1.4])
        h1.markdown('#### Recent Flagged Episodes')
        h1.caption('Consecutive flagged minutes (gap ≤ 5 min) merged · press Inspect to open an episode')
        lvl = h2.segmented_control('Show', ['ALERT only', 'WATCH + ALERT'], default='WATCH + ALERT', key='feedlvl')
        mn_lvl = 2 if lvl == 'ALERT only' else 1
        rows = []
        for jj in B.values():
            for v in vis_eps(jj, i):
                if v.mx >= mn_lvl:
                    rows.append(v)
        rows.sort(key=lambda v: -v.s)
        if not rows:
            st.info('No flagged episode yet in this replay. Press Play or use Next Alert.')
        for n, v in enumerate(rows[:12]):
            jj = v.j
            d = lead_det(jj, v.eng)
            pk = 100 - (jj.integ[v.eng] if np.isfinite(jj.integ[v.eng]) else 100)
            mw = is_market_wide(v, i, B)
            cf = jj.conf[v.eng]
            a, b, c, e = st.columns([1.5, 5, 1.1, 0.9])
            a.markdown(f'`{jj.ts[v.s][:16]}` **{jj.sym}**')
            b.markdown(f"**{v.len} min** · Lead: {d['name'] if d else '–'} · Peak unusualness {pk:.1f}%" + ('  ·  :blue[MARKET-WIDE]' if mw else ''))
            c.markdown(badge(lvl_txt(v), COL[v.mx]) + (f' <span style="color:{CONFC.get(cf, DIM)};font-size:11px">CONF {cf}</span>' if v.mx == 2 and cf else ''), unsafe_allow_html=True)
            e.button('Inspect', key=f'insp{n}_{jj.symbol}_{v.s}', on_click=open_episode, args=(jj.symbol, v.s, v.pk))


# ------------------------------------------------------------------ episode window figures
def ep_window(v, i):
    j = v.j
    a, b = max(j.first, v.s - 60), min(i, v.e + 30)
    x = j.dt[a:b + 1]
    shade = COL[v.mx]
    f = base_fig(180, 64)
    f.add_vrect(x0=j.dt[v.s], x1=j.dt[v.e] + np.timedelta64(1, 'm'), fillcolor=shade, opacity=0.22, line_width=0)
    f.add_trace(go.Scatter(x=x, y=j.close[a:b + 1], mode='lines', line=dict(color=TXT, width=1.6), hovertemplate='$%{y:,.0f}'))
    f.update_yaxes(tickprefix='$', tickformat=',.0f')
    g = base_fig(170, 64)
    g.add_vrect(x0=j.dt[v.s], x1=j.dt[v.e] + np.timedelta64(1, 'm'), fillcolor=shade, opacity=0.22, line_width=0)
    g.add_trace(go.Scatter(x=x, y=j.thr_alert[a:b + 1], mode='lines', line=dict(color=COL[2], dash='dash', width=1.2), name='ALERT line'))
    g.add_trace(go.Scatter(x=x, y=j.thr_watch[a:b + 1], mode='lines', line=dict(color=COL[1], dash='dash', width=1.2), name='WATCH line'))
    g.add_trace(go.Scatter(x=x, y=j.fusion[a:b + 1], mode='lines', line=dict(color=ACC, width=1.8), name='Fused score'))
    g.add_trace(go.Scatter(x=[j.dt[v.pk]], y=[j.fusion[v.pk]], mode='markers', marker=dict(color='#fff', size=8), name='Peak'))
    return f, g


def pick_episode(vis):
    """shared episode picker, shown once above the tabs"""
    if not vis:
        st.info('No episode has been flagged yet in the replay. Press Play or use Next Alert.')
        return None, None
    j = CUR()
    latest = vis[-1]
    opts = list(reversed(vis))[:40]
    labels = {}
    for e in opts:
        labels[e.s] = f"{j.ts[e.s][5:16]} · ALERT · {evid(e).level.title()} evidence" if e.mx == 2 else f"{j.ts[e.s][5:16]} · WATCH only (not rated)"
    cur_s = ss().sel if ss().sel in labels else latest.s
    idx = [e.s for e in opts].index(cur_s)

    pk = f'ep_pick_{j.symbol}_{latest.s}_{ss().sel}'

    def _chg():
        ss().sel = ss()[pk]

    s = st.selectbox('Episode to inspect', [e.s for e in opts], index=idx, format_func=lambda s_: labels[s_], key=pk, on_change=_chg)
    v = next(e for e in opts if e.s == s)
    if j.state[ss().i] >= 1 and v is latest:
        st.caption(f'{lvl_txt(v)} episode in progress · started {tstr(j, v.s)}')
    elif v is latest:
        st.caption(f'No alert right now. Showing the latest episode: {tstr(j, v.s)} ({ss().i - v.e} min ago)')
    else:
        st.caption(f'Showing the selected episode: {tstr(j, v.s)}')
    return v, latest


# ------------------------------------------------------------------ EXPLAINABLE AI
def render_xai(v):
    if v is None:
        st.info('Nothing has been flagged yet. Press Play, or use Next Alert at the top.')
        return
    j, k = v.j, v.eng
    i = ss().i
    x = j.xai.get(v.xb) if v.xb is not None else None
    sc = j.integ[k]
    F = j.features
    names = [fname(f) for f in F]
    rs = j.reasons.get(k) or []
    feats = [r for r in rs if r['type'] == 'feature']
    mods = [r for r in rs if r['type'] == 'model']
    top = F[int(np.argmax(x['shap']))] if x and x.get('shap') else (feats[0]['feature'] if feats else None)
    ev = evid(v)
    nA = ev.nA
    if top:
        take = f"The strongest push toward unusual behaviour came from **{fname(top).lower()}**" + (
            '. No single check crossed its own line; the combined score did' if nA == 0 else (f', flagged by **{nA} of 3** checks' if nA is not None else '')) + '.'
    else:
        take = f'The fused score rose above its {lvl_txt(v).lower()} line.'
    h1, h2, h3 = st.columns([1.1, 1.1, 1.1])
    with h1.container(border=True):
        a, b = st.columns([1, 1.3])
        a.plotly_chart(ring_fig(sc, COL[v.mx]), width='stretch', config=dict(displayModeBar=False), key='x_ring')
        with b:
            label('Episode state (at its peak)')
            st.markdown(pill(v.mx, lvl_txt(v)), unsafe_allow_html=True)
            st.caption(f'{j.sym}\n\n{hhmm(j, v.s)} – {hhmm(j, v.e)} UTC ({v.len} min)\n\nThe state of this episode at its strongest minute, not of the current minute (see Overview).')
    with h2.container(border=True):
        st.markdown('**Key takeaway**')
        st.markdown(take)
    with h3.container(border=True):
        st.markdown('**Top supporting signals** ' + badge('INDICATIVE', ACC), unsafe_allow_html=True)
        sig = (feats if feats else mods)[:3]
        if not sig:
            st.caption('No signal detail for this episode.')
        smx = max([s_.get('weight') or 0 for s_ in sig] + [1e-9])
        for s_ in sig:
            nm = fname(s_['feature']) if s_['type'] == 'feature' else (s_['text'][:44] + '…' if len(s_['text']) > 46 else s_['text'])
            w = s_.get('weight') or 0
            st.markdown(f'{nm} &nbsp; **{round(w * 100)}%**')
            st.progress(min(1.0, max(0.04, w / smx)))
    resem_card(v)
    st.markdown('### 1 · Price & unusual score')
    c1, c2 = st.columns([2, 1])
    with c1.container(border=True):
        f, g = ep_window(v, i)
        st.caption('PRICE (USD) · shaded = this episode')
        st.plotly_chart(f, width='stretch', config=dict(displayModeBar=False), key='x_price')
        st.caption('FUSED SCORE · dashed red = ALERT line · dashed amber = WATCH line')
        st.plotly_chart(g, width='stretch', config=dict(displayModeBar=False), key='x_fused')
    with c2.container(border=True):
        st.markdown('**Which checks noticed it**')
        dmax = max([abs(getattr(j, d['c'])[k]) if np.isfinite(getattr(j, d['c'])[k]) else 0 for d in DET] + [1e-9])
        for d in DET:
            on = ev.ag is not None and (ev.ag & d['bit']) > 0
            sh = getattr(j, d['c'])[k]
            with st.container(border=True):
                st.markdown(f"**{d['name']}** &nbsp; " + badge('N/A' if ev.ag is None else 'ABOVE ITS LINE' if on else 'BELOW ITS LINE', COL[2] if on else DIM), unsafe_allow_html=True)
                st.caption(f"Contribution to the fused score: {fx(sh)}")
                st.progress(float(min(1.0, max(0.02, abs(sh) / dmax))) if np.isfinite(sh) else 0.02)
        with st.container(border=True):
            st.markdown('**TGNN check** ' + badge('COMING SOON', DIM), unsafe_allow_html=True)
    st.markdown('### 2 · Which signals mattered')
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown('**Known-pattern check · signal contributions** ' + badge('INDICATIVE', ACC), unsafe_allow_html=True)
        if x and x.get('shap'):
            st.caption('Right = pushed the score up · Left = pulled it down')
            st.plotly_chart(hbar_fig(x['shap'], names, True), width='stretch', config=dict(displayModeBar=False), key='x_shap')
            q = int(np.argmax(x['shap']))
            st.markdown(f'Biggest push: **{names[q]}** ({x["shap"][q]:+.2f}).')
        else:
            st.caption('Detailed signal data was not captured for this episode.')
    if x and x.get('ae_err'):
        with c2.container(border=True):
            st.markdown('**Unfamiliar-behaviour check · unusual signals** ' + badge('SECONDARY', DIM), unsafe_allow_html=True)
            st.caption("Longer bar = less familiar versus this market's own recent past")
            st.plotly_chart(hbar_fig(x['ae_err'], names, False), width='stretch', config=dict(displayModeBar=False), key='x_ae')
    if x and x.get('tr_time') and x.get('tr_feat'):
        st.markdown('### 3 · Unusual behaviour over time')
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown('**Which recent minutes the multi-minute check relied on** ' + badge('INDICATIVE', ACC), unsafe_allow_html=True)
            t = np.array(x['tr_time'])
            ff = go.Figure(go.Bar(x=list(range(-len(t) + 1, 1)), y=t * 100, marker_color='#6366F1', hovertemplate='%{x} min: %{y:.1f}%<extra></extra>'))
            ff.update_layout(template='plotly_dark', height=210, margin=dict(l=40, r=8, t=6, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             xaxis=dict(title='minutes before the flagged minute', gridcolor=GRID), yaxis=dict(title='%', gridcolor=GRID))
            st.plotly_chart(ff, width='stretch', config=dict(displayModeBar=False), key='x_trt')
        with c2.container(border=True):
            st.markdown('**Signals it relied on** ' + badge('INDICATIVE', ACC), unsafe_allow_html=True)
            st.plotly_chart(hbar_fig(x['tr_feat'], names, False), width='stretch', config=dict(displayModeBar=False), key='x_trf')
    st.caption('An alert is an attention signal, not proof of wrongdoing or a trading recommendation.')


# ------------------------------------------------------------------ UNCERTAINTY
def win_row(v, hk, n, lab):
    j = v.j
    m0 = j.t0
    mk = m0 + v.eng
    s0 = (mk // n) * n
    now = m0 + ss().i
    w = j.wmap[hk].get(s0)
    if s0 + n > now + 1:
        cl = pd.Timestamp((s0 + n) * 60, unit='s').strftime('%H:%M')
        st.markdown(f'**{lab}** · still open · verdict at {cl} UTC')
        return
    if not w:
        st.markdown(f'**{lab}** · n/a')
        return
    lv = {'ALERT': 2, 'WATCH': 1}.get(w['tier'], 0)
    st.markdown(f"**{lab}** &nbsp; {badge(w['tier'], COL[lv])}<br><span style='color:{DIM};font-size:12px'>{w['start'][11:16]}–{w['end'][11:16]} UTC · peak unusualness {w['peak_unus']:.1f}% · {w['flagged_min']} flagged min</span>", unsafe_allow_html=True)


def render_unc(v, vis):
    if v is None:
        st.info('Nothing has been flagged yet. Press Play, or use Next Alert at the top.')
        return
    j, k = v.j, v.eng
    e = evid(v)
    x = j.xai.get(v.xb) if v.xb is not None else None
    sc = j.integ[k]
    F = j.features
    lvl = e.level
    lc = LVC[lvl] if lvl else DIM
    one = ('Below the alert line. Evidence levels are rated for ALERT episodes only.' if not lvl else
           'Several independent checks agree with each other.' if lvl == 'STRONG' else
           'Some checks agree, others do not.' if lvl == 'MODERATE' else 'Mostly a single signal. Treat it as a hint.')
    top = F[int(np.argmax(x['shap']))] if x and x.get('shap') else None
    thr = j.thr_alert[v.pk] if v.mx == 2 else j.thr_watch[v.pk]
    found = (f'Unusual **{fname(top).lower()}** for {v.len} min at {hhmm(j, v.s)} UTC.' if top else f"{v.len} flagged minute{'s' if v.len > 1 else ''} starting {hhmm(j, v.s)} UTC.")
    why = f'The fused score reached **{fx(j.fusion[v.pk])}**, above the {lvl_txt(v).lower()} line of {fx(thr)}.'
    means = ('Stay aware. Nothing is raised as an alert.' if not lvl else 'Worth a close look. It is not proof of wrongdoing.' if lvl == 'STRONG'
             else 'Treat it as a warning and keep watching.' if lvl == 'MODERATE' else 'Treat it as a hint, not a conclusion.')
    a, b = st.columns([2, 1])
    with a.container(border=True):
        label('Assessment reliability')
        st.markdown(f'<div style="margin:6px 0"><span style="font-size:34px;font-weight:800;color:{lc}">{lvl or "NOT RATED"}</span> &nbsp;'
                    + (badge('CONF ' + e.conf, CONFC.get(e.conf, DIM)) if lvl and e.conf else '') + '</div>', unsafe_allow_html=True)
        st.write(one)
    with b.container(border=True):
        r1, r2 = st.columns([1, 1.2])
        r1.plotly_chart(ring_fig(sc, COL[v.mx]), width='stretch', config=dict(displayModeBar=False), key='u_ring')
        with r2:
            label('Episode state (at its peak)')
            st.markdown(pill(v.mx, lvl_txt(v)), unsafe_allow_html=True)
            st.caption(f'{j.sym} · {hhmm(j, v.s)} UTC')
    c = st.columns(3)
    for col, (t, body) in zip(c, [('What the system found', found), ('Why it was flagged', why), ('What this means for you', means)]):
        with col.container(border=True):
            st.markdown(f'**{t}**')
            st.markdown(body)
    st.markdown('### Evidence consistency')
    last5 = ''.join('●' if j.state[q] == 2 else '○' for q in range(k - 4, k + 1))
    ck = lambda b_: '❔' if b_ is None else '✅' if b_ else '❌'
    c = st.columns(3)
    with c[0].container(border=True):
        st.markdown(f'**{ck(None if e.pl is None else e.rare)} Rare minute**')
        st.caption('n/a' if e.pl is None else f"p-level {e.pl:.4f} (needs ≤ 0.001)")
    with c[1].container(border=True):
        st.markdown(f'**{ck(e.pers)} Persistent signal**')
        st.caption(f'{e.cnt} of the last 5 minutes were ALERT (needs 3)')
        st.markdown(f'`{last5}`')
    with c[2].container(border=True):
        st.markdown(f'**{ck(None if e.nA is None else e.agr)} Checks agree**')
        st.caption('n/a' if e.nA is None else f'{e.nA} of 3 checks above their own line (needs 2)')
        st.markdown('`' + ''.join('●' if e.ag is not None and (e.ag & d['bit']) else '○' for d in DET) + '`')
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown('**How rare is this minute**')
        if e.pl:
            N = round(1 / e.pl)
            st.progress(max(0.02, min(1.0, np.log10(1 / e.pl) / 4)))
            st.caption('1 in 10 · 1 in 100 · 1 in 1,000 · 1 in 10,000')
            st.markdown(f'About **1 in {N:,}** minutes reach this level (p = {e.pl:.4f}).')
        else:
            st.caption('Rarity is not available for this minute.')
    with c2.container(border=True):
        st.markdown('**Closed time-window verdicts**')
        win_row(v, '15m', 15, '15-minute window')
        win_row(v, '1h', 60, '1-hour window')
    with st.expander('How was this assessed? (technical details)'):
        cl = j.conf_labels or {}
        st.markdown('**Rule:** three checks are made at the strongest minute of the episode. **Rare** = the score is rarer than 1 in 1,000 minutes of this market\'s last 7 days. '
                    '**Persistent** = at least 3 of the last 5 minutes were ALERT. **Checks agree** = at least 2 of the 3 detectors are above their own line. '
                    '3 met = Strong, 2 = Moderate, otherwise Limited. Only ALERT episodes are rated.')
        st.markdown(f"**Engine confidence:** HIGH = {cl.get('High')}. MEDIUM = {cl.get('Medium')}. LOW = {cl.get('Low')}.")
        st.markdown('**Detectors:** ' + ' · '.join(f"{d['name']} = {d['tech']}" for d in DET) + '.')
    cal = j.cal
    if cal:
        st.markdown('### Uncertainty visualisation')
        c1, c2 = st.columns([2, 1])
        with c1.container(border=True):
            st.markdown('**Fused-score distribution · calibration minutes vs this episode**')
            E = np.array(cal['edges'])
            ff = go.Figure()
            for key, nm, col, fc in [('normal', f"normal minutes ({cal['n_normal']:,})", '#38BDF8', 'rgba(56,189,248,.25)'),
                                     ('abnormal', f"abnormal minutes ({cal['n_abnormal']:,})", '#A78BFA', 'rgba(167,139,250,.25)')]:
                ff.add_trace(go.Scatter(x=np.repeat(E, 2)[1:-1], y=np.repeat(cal[key], 2), mode='lines', line=dict(color=col, width=1.4),
                                        fill='tozeroy', fillcolor=fc, name=nm))
            xv = float(max(E[0], min(E[-1], j.fusion[v.pk])))
            ff.add_vline(x=xv, line=dict(color=COL[2], dash='dash', width=2), annotation_text=f'This episode: {j.fusion[v.pk]:.2f}', annotation_font_color=COL[2])
            ff.update_layout(template='plotly_dark', height=270, margin=dict(l=40, r=10, t=20, b=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             legend=dict(orientation='h', y=-0.2), xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, title='density'))
            st.plotly_chart(ff, width='stretch', config=dict(displayModeBar=False), key='u_cal')
        with c2.container(border=True):
            st.markdown('**Reliability levels**')
            for l_, t_ in [('STRONG', 'All three checks met'), ('MODERATE', 'Two of three checks met'), ('LIMITED', 'Fewer than two checks met')]:
                on = lvl == l_
                st.markdown(f'<div style="border:1px solid {LVC[l_] if on else GRID};background:{LVC[l_] + "18" if on else "transparent"};border-radius:8px;padding:8px 10px;margin-bottom:6px">'
                            f'<b style="color:{LVC[l_]}">{l_.title()} evidence</b><br><span style="color:{DIM};font-size:12px">{t_}</span></div>', unsafe_allow_html=True)
            cnt = {'STRONG': 0, 'MODERATE': 0, 'LIMITED': 0}
            for q in vis:
                if q.mx == 2:
                    cnt[evid(q).level] += 1
            st.caption(f"All alerts so far ({sum(cnt.values())}): Strong {cnt['STRONG']} · Moderate {cnt['MODERATE']} · Limited {cnt['LIMITED']}")
    with st.expander('Technical notes (calibration and diagnostics)'):
        tw = j.thr_windows or {}
        g = lambda a_, b_: fx((tw.get(a_) or {}).get(b_))
        st.markdown(f"- Rarity p-level: `{e.pl}`\n- Detectors above line (bitmask): `{'n/a' if e.ag is None else format(e.ag, '03b') + ' (AE·XGB·TR)'}`\n"
                    f"- Engine confidence: `{e.conf}`\n- Alert / watch line at peak: `{fx(j.thr_alert[v.pk], 3)} / {fx(j.thr_watch[v.pk], 3)}`\n"
                    f"- 15-min lines (alert / watch): `{g('15 min', 'Alert')} / {g('15 min', 'Watch')}`\n- 1-hour lines (alert / watch): `{g('1 hour', 'Alert')} / {g('1 hour', 'Watch')}`\n"
                    f"- Calibration source: `{cal['source'] if cal else 'n/a'}`\n- Calibration table id: `{cal['npz_sha16'] if cal else 'n/a'}`")


# ------------------------------------------------------------------ SIMILAR PAST
HW, HSEP = 60, 120


def hist_desc(sym, n, key):
    """per-minute descriptor of the previous 60 minutes of fused/AE/XGB/Transformer scores (mean, max, last, std); one cached copy per symbol and processed length"""
    c = ss().setdefault('_hd', {})
    if sym in c and c[sym][0] == key:
        return c[sym][1]
    j = ss()._B[sym]
    cols = []
    for f in ['fusion', 'ae', 'xgb', 'tr']:
        s_ = pd.Series(getattr(j, f))
        r = s_.rolling(HW, min_periods=HW)
        cols += [r.mean().to_numpy(), r.max().to_numpy(), s_.to_numpy(), r.std(ddof=0).to_numpy()]
    D = np.vstack(cols).T
    c[sym] = (key, D)
    return D


def hist_find(j, k):
    D = hist_desc(j.symbol, j.n, (j.first, getattr(j, 'done', j.n)))
    first = j.first + HW - 1
    q = D[k]
    if np.isnan(q).any() or k - HSEP < first:
        return None
    C = D[first:k - HSEP + 1]
    ok = ~np.isnan(C).any(axis=1)
    idx = np.arange(first, k - HSEP + 1)[ok]
    C = C[ok]
    if len(C) < 10:
        return None
    sd = C.std(axis=0)
    sd[sd == 0] = 1
    dist = np.sqrt((((C - q) / sd) ** 2).sum(axis=1))
    order = np.argsort(dist)
    pick = []
    for o in order:
        if all(abs(idx[o] - p[0]) >= HW for p in pick):
            pick.append((int(idx[o]), float(dist[o])))
        if len(pick) == 5:
            break
    return pick


def spark(vals, thr, split=None, h=130):
    f = base_fig(h, 36)
    f.update_layout(hovermode=False, margin=dict(l=36, r=6, t=6, b=22))
    f.add_trace(go.Scatter(y=vals, mode='lines', line=dict(color=ACC, width=1.6)))
    if thr is not None and np.isfinite(thr):
        f.add_hline(y=thr, line=dict(color=COL[2], dash='dash', width=1))
    if split is not None:
        f.add_vline(x=split, line=dict(color='#2C3858', width=1))
        f.add_annotation(x=split, y=0, yref='paper', text='then | next hour →', showarrow=False, font=dict(size=9, color=DIM), xanchor='left', yanchor='top')
    f.update_xaxes(visible=False)
    return f


def render_hist(j, i):
    st.info('**Context, not a forecast.** Similar past patterns do not guarantee the same future. Matches are searched only among earlier minutes of this recording (at least 2 hours before the current minute), never looking ahead.')
    res = hist_find(j, i)
    with st.container(border=True):
        st.markdown('**Current 60-minute profile**')
        st.caption('Fused score · dashed = alert line')
        st.plotly_chart(spark(j.fusion[i - HW + 1:i + 1], j.thr_alert[i], None, 150), width='stretch', config=dict(displayModeBar=False), key='h_now')
    if not res:
        st.info('Not enough earlier history yet to find similar moments. Move the replay later (a few hours after the start).')
        return
    rows = []
    for e, d in res:
        seg = j.state[e + 1:e + 61]
        pk = np.nanmax(100 - j.integ[e + 1:e + 61]) if len(seg) else float('nan')
        rows.append(dict(e=e, d=d, mx=int(seg.max()) if len(seg) else 0, fl=int((seg >= 1).sum()), pk=pk))
    fol = sum(1 for r in rows if r['mx'] == 2)
    st.markdown('### Top 5 most similar earlier hours')
    st.caption(f'{fol} of 5 were followed by an ALERT within the next hour')
    cols = st.columns(3) + st.columns(2)
    for n, (r, c) in enumerate(zip(rows, cols)):
        with c.container(border=True):
            st.markdown(f"**#{n + 1} closest · {j.ts[r['e']][5:16]} UTC** <span style='color:{DIM};font-size:12px'>distance {r['d']:.2f}</span>", unsafe_allow_html=True)
            seg = j.fusion[r['e'] - HW + 1:r['e'] + 61]
            st.plotly_chart(spark(seg, j.thr_alert[r['e']], HW - 1), width='stretch', config=dict(displayModeBar=False), key=f'h_{n}')
            s0 = int(j.state[r['e']])
            st.markdown(f"At that time: <b style='color:{COL[s0]}'>{NAME[s0]}</b><br>In the next hour: highest level <b style='color:{COL[r['mx']]}'>{NAME[r['mx']]}</b> · {r['fl']} flagged min · "
                        f"peak unusualness {('–' if np.isnan(r['pk']) else format(r['pk'], '.1f') + '%')}", unsafe_allow_html=True)


# ------------------------------------------------------------------ page
CSS = """
<style>
.block-container{padding-top:1.2rem;max-width:1500px}
[data-testid="stMetricValue"]{font-size:1.35rem}
[data-testid="stMetricLabel"]{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}
</style>
"""


def init_state(B):
    s = ss()
    if 'sym' not in s:
        s.sym = SYMBOLS[0]
        s.sym_sel = 'BTC/USDT'
        j = B[s.sym]
        s.ver, s.playing, s.sel, s.toast = 0, False, None, None
        if live_mode():
            # run the models forward until the first alert episode of at least 3 minutes has closed, so every tab has something real to show
            import live_engine as LE
            ok = lambda jj: any(e[1] - e[0] + 1 >= 3 and jj.done - 1 - e[1] > 5 for e in jj.alert_eps)
            with st.spinner('Running the trained models over the first hours of the recording ...'):
                LE.advance_until(B, s.sym, ok, j.n - 1)
            e = next((e for e in j.alert_eps if e[1] - e[0] + 1 >= 3), None)
            s.i = e[1] if e is not None else j.first
        else:
            # open on the Aug 5 stress-day alert (enough history behind it for every tab); else on the first alert
            e = next((e for e in j.alert_eps if j.ts[e[0]][:10] == '2024-08-05'), None) or (j.alert_eps[0] if j.alert_eps else None)
            s.i = e[1] if e is not None else j.first


def run_goal(B):
    """long forward jumps in live mode: run the trained models forward with a visible progress bar, then do the jump"""
    s = ss()
    g = s.get('goal')
    s.goal = None
    if not g or not live_mode():
        return
    import live_engine as LE
    j = B[s.sym]
    i = s.i
    bar = st.progress(0.0, text='Running the trained models ...')
    prog = lambda f, t: bar.progress(f, text=t)
    if g == 'next':
        LE.advance_until(B, s.sym, lambda jj: any(e[0] - 30 > i for e in jj.alert_eps), j.n - 1, prog)
        bar.empty()
        jump_alert(1)
    elif g == 'aug5':
        LE.advance_until(B, s.sym, lambda jj: any(jj.ts[e[0]][:10] == '2024-08-05' for e in jj.alert_eps) or jj.ts[jj.done - 1][:10] > '2024-08-05', j.n - 1, prog)
        bar.empty()
        jump_aug()


def ensure_scored(B, upto):
    """live mode: make sure the trained models have scored every minute up to `upto` (both markets)"""
    if not live_mode() or all(getattr(j, 'done', j.n) > upto for j in B.values()):
        return
    import live_engine as LE
    if upto - min(j.done for j in B.values()) > 16:
        bar = st.progress(0.0, text='Running the trained models ...')
        LE.advance_all(B, upto, lambda f, t: bar.progress(f, text=t))
        bar.empty()
    else:
        LE.advance_all(B, upto)


def render_data_checks():
    """try-it panel: what the system does with missing, garbage or impossible input. Checks only; nothing here is scored or stored."""
    paths = engine_paths()
    with st.expander('What happens with missing or garbage data? (try it)'):
        if paths is None:
            st.caption('This panel needs the engine folder (backend) next to the dashboard.')
            return
        if str(paths[0]) not in sys.path:
            sys.path.insert(0, str(paths[0]))
        from c03_engine import validate as V
        st.markdown('Every minute is checked **before** the trained models see it. A minute that fails is shown as **NO DATA** with the reason, '
                    'it never enters the models\' memory, and it is never shown as NORMAL. The next valid minute is scored as if it had never arrived.')
        st.markdown('##### 1 · Try one minute')
        j = CUR()
        k = max(j.first, min(ss().i, j.n - 1))
        real = [float(j.df.loc[k, f]) for f in V.FEATS] if hasattr(j, 'df') else None
        presets = {'A normal minute (copied from the recording)': real, 'All six values are 0': [0.0] * 6, 'An empty value (NaN)': [float('nan')] + (real or [0.0] * 6)[1:],
                   'Infinity': [(real or [0.0] * 6)[0], float('inf')] + (real or [0.0] * 6)[2:], 'A negative number of large trades': (real or [0.0] * 6)[:5] + [-5.0],
                   'Price range of 90 % in one minute': (real or [0.0] * 6)[:2] + [0.9] + (real or [0.0] * 6)[3:], 'An absurd volume spike (1e12)': (real or [0.0] * 6)[:4] + [1e12, (real or [0.0] * 6)[5]],
                   'Text instead of numbers': ['abc'] * 6, 'Only three numbers': [1.0, 2.0, 3.0], 'I will type my own numbers': None}
        pick = st.selectbox('Input', list(presets), key='dq_pick')
        vec = presets[pick]
        if vec is None:
            raw = st.text_input('Six numbers separated by commas: ' + ', '.join(V.FEATS), value='0.0008, 0.12, 0.0033, 12, 0.4, 3', key='dq_text')
            vec = [x.strip() for x in raw.split(',')] if raw.strip() else []
        else:
            st.caption('Values: ' + ', '.join(str(x) for x in vec))
        probs = V.check_features(vec)
        if probs:
            st.error('**Rejected: shown as NO DATA.** The models do not see this minute.')
            for p_ in probs:
                st.markdown(f"- {p_['text']}" + (f" ({p_['detail']})" if p_.get('detail') else ''))
        else:
            st.success('**Accepted.** This minute would be scored by the trained models.')
        st.markdown('##### 2 · Try a file')
        up = st.file_uploader('Drop any file (a CSV of minutes, or anything else, even a picture)', key='dq_file')
        if up is not None:
            rep = V.inspect_csv(up.name, up.getvalue())
            (st.success if rep['ok'] else st.error)(('Usable file' if rep['ok'] else 'This file cannot be used') + (f" · detected format: {rep['kind'].replace('_', ' ')} · {rep['rows']:,} rows" if rep['kind'] else ''))
            for f_ in rep['findings']:
                {'error': st.error, 'warn': st.warning, 'info': st.info}[f_['level']](f_['text'])
            st.caption('The file is only checked here; nothing is scored or stored.')
        st.caption('Limits are physical sanity limits (at least 5 times the largest value in the Aug 2024 data), never used to tune a model.')


def render(data_dir=None):
    """Entry point: draws the whole component. Use inside a tab of the 4-component platform."""
    t_run = time.time()
    if data_dir:
        ss()._data_dir = str(data_dir)
    B, miss = bundles()
    if miss:
        st.error('Bundle not found: ' + ', '.join(miss) + '. Run backend/tools/make_dashboard_bundle.py and save the two files in the data folder.')
        return
    ss()._B = B
    init_state(B)
    run_goal(B)
    ensure_scored(B, ss().i)
    st.markdown(CSS, unsafe_allow_html=True)
    s = ss()
    top = st.columns([3.2, 1.6, 2.2])
    top[0].markdown('## 🛡️ TradeSense · IntegritySense')
    top[0].caption('Market Integrity Intelligence Engine · 1-minute Binance Spot bars · Monitor-only · Component 03')
    with top[1]:
        pick = st.segmented_control('Market', ['BTC/USDT', 'ETH/USDT'], key='sym_sel')
        if pick and pick.replace('/', '') != s.sym:
            s.sym, s.sel, s.playing = pick.replace('/', ''), None, False
    j = CUR()
    s.i = max(j.first, min(j.n - 1, s.i))
    mode_txt = 'TRAINED MODELS RUN LIVE ON EACH MINUTE' if live_mode() else 'STORED ENGINE RESULTS (REPLAY)'
    top[0].markdown(badge('● ' + mode_txt, '#10B981' if live_mode() else '#F59E0B') + f'<span style="color:{DIM};font-size:12px"> &nbsp; recorded data {j.ts[0][:10]} → {j.ts[j.n - 1][:10]}</span>', unsafe_allow_html=True)
    top[2].markdown(f'<div style="height:6px"></div>'
                    f'<div style="text-align:right;font-size:18px;font-weight:700;font-family:monospace">{tstr(j, s.i)}</div>', unsafe_allow_html=True)
    # controls
    c = st.columns([1, 1, 1, 0.9, 2.3, 1.1, 1.1, 1.7])
    c[0].button('◀ 1 bar', width='stretch', on_click=lambda: set_i(s.i - 1))
    c[1].button('❚❚ Pause' if s.playing else '▶ Play', type='primary', width='stretch', on_click=toggle_play)
    c[2].button('1 bar ▶', width='stretch', on_click=lambda: set_i(s.i + 1))
    speed = c[3].selectbox('Speed (bars/s)', [1, 2, 5, 10], index=1, key='speed', label_visibility='collapsed')
    autop = c[4].checkbox('Auto-pause at WATCH/ALERT', value=True, key='autop')
    c[5].button('◀ Prev Alert', width='stretch', on_click=lambda: jump_alert(-1))
    c[6].button('Next Alert ▶', width='stretch', on_click=lambda: jump_alert(1))
    c[7].button('⚡ Jump to Aug 5 (stress day)', width='stretch', on_click=jump_aug)
    st.slider('Replay position', j.first, j.n - 1, int(s.i), key=f'seek{s.ver}', on_change=on_seek, label_visibility='collapsed')
    nbad = sum(1 for k in getattr(j, 'bad', {}) if k <= s.i)
    st.caption(f'{s.i - j.first + 1 - nbad} / {j.n - j.first} bars scored · {tstr(j, s.i)}' + (f' · {nbad} minute(s) skipped: invalid data' if nbad else ' · data checks: every minute so far passed'))
    if s.i in getattr(j, 'bad', {}):
        st.warning('**No score for this minute (NO DATA).** ' + ' '.join(j.bad[s.i]) + ' The models did not see it, and it is not shown as NORMAL. The next valid minute is scored as if this one never arrived.')
    if s.toast:
        st.toast(s.toast)
        s.toast = None
    i = s.i
    vis = vis_eps(j, i)
    v, latest = pick_episode(vis) if True else (None, None)
    t1, t2, t3, t4 = st.tabs(['Overview', 'Explainable AI & Evidence', 'Uncertainty & Assessment Reliability', 'Similar past · Historical Context'])
    with t1:
        render_overview(B, j, i)
    with t2:
        render_xai(v)
    with t3:
        render_unc(v, vis)
    with t4:
        render_hist(j, i)
    render_data_checks()
    render_provenance(B)
    # replay engine: advance, then rerun
    if s.playing:
        period = 1.0 / speed if speed <= 2 else 0.5
        adv = 1 if speed <= 2 else int(speed * period)
        time.sleep(max(0.05, period - (time.time() - t_run)))
        nxt = min(j.n - 1, s.i + adv)
        ensure_scored(B, nxt)
        stop = nxt >= j.n - 1
        if autop:
            for k in range(s.i + 1, nxt + 1):
                if j.state[k] >= 1 and j.state[k - 1] == 0:
                    nxt, stop, s.sel = k, True, None
                    s.toast = f'Paused: {NAME[int(j.state[k])]} started at {hhmm(j, k)} UTC'
                    break
        s.i = nxt
        s.ver += 1
        if stop:
            s.playing = False
        st.rerun()


if __name__ == '__main__':
    st.set_page_config(page_title='C03 IntegritySense', page_icon='🛡️', layout='wide')
    render()
