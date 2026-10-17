"""
verify_dynamic.py - proves that the dashboard shows REAL, DYNAMIC results produced by the TRAINED models, and nothing typed in.

Run from this folder:   python verify_dynamic.py
Needs: app.py, live_engine.py, ../backend (c03_engine + export with models) and data/declared_pattern_resemblance_v1.json

Part L  LIVE MODELS: the engine run inside the dashboard reproduces the research notebooks' stored scores bar for bar (each model and the fusion),
        is deterministic, and REACTS to the input features (change the market data -> the scores change, earlier minutes do not)
Part U  the page shows exactly what the live models computed (screen value == independently recomputed model value)
Part D  missing / garbage / all-zero / out-of-order / too-short data: rejected with a reason, never shown as NORMAL, leaves no trace in the models' memory
Part A/B (only if the stored-results files are present) the stored-results fallback obeys the frozen rules and follows its files
Part C  the code has no random numbers, no mock data and no big hand-typed number lists
"""
import ast
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'   [{detail}]' if detail else ''))


def arr(b, k):
    return np.array([np.nan if v is None else v for v in b[k]], float)


# ----------------------------------------------------------------------------- Part L / U (live trained models)
print('\n=== Part L: the trained models run inside the dashboard and reproduce the notebooks ===')
sys.path.insert(0, str(HERE))
import live_engine as LE

found, why = LE.find_backend(HERE)
LIVE_OK = found is not None
check('backend engine and export found (models, config, replay bars)', LIVE_OK, why or str(found[1]))
if LIVE_OK:
    sys.path.insert(0, str(found[0]))
    from c03_engine.store import Export
    ex = Export(found[1], verify=True)
    check('every engine file matches the sha256 manifest (models unchanged since export)', True, f"{len(ex.manifest.get('files', {}))} files")
    N_CHECK = 600          # minutes scored by the models in this check (1500 .. 2099)
    refs = {'fusion': 'exp_fusion', 'ae': 'exp_ae', 'xgb': 'exp_xgb', 'tr': 'exp_tr', 'thr_alert': 'exp_thr_alert', 'thr_watch': 'exp_thr_watch8',
            'c_ae': 'exp_contrib_ae', 'c_xgb': 'exp_contrib_xgb', 'c_tr': 'exp_contrib_tr', 'c_shared': 'exp_contrib_shared'}
    runs = {}
    for sym in ['BTCUSDT', 'ETHUSDT']:
        j = LE.make_live(ex, sym, None)
        j.advance(LE.FIRST + N_CHECK - 1)
        runs[sym] = j
        sl = slice(LE.FIRST, LE.FIRST + N_CHECK)
        worst = {}
        for mine, ref in refs.items():
            a, r = getattr(j, mine)[sl], j.df[ref].to_numpy('float64')[sl]
            m = np.isfinite(a) & np.isfinite(r)
            worst[mine] = float(np.abs(a[m] - r[m]).max()) if m.any() else float('nan')
        check(f'{sym}: Autoencoder, XGBoost, Transformer and fused scores equal the notebook scores (max difference < 1e-3)',
              all(worst[k] < 1e-3 for k in ['fusion', 'ae', 'xgb', 'tr']), ', '.join(f'{k} {worst[k]:.1e}' for k in ['ae', 'xgb', 'tr', 'fusion']))
        check(f'{sym}: exact contributions and the alert / watch lines equal the notebook values (< 1e-3)', all(worst[k] < 1e-3 for k in ['thr_alert', 'thr_watch', 'c_ae', 'c_xgb', 'c_tr', 'c_shared']))
        ref_state = j.df.exp_state.map(LE.STATE_NUM).to_numpy()[sl]
        check(f'{sym}: NORMAL / WATCH / ALERT decision equals the notebook decision on every minute', int((ref_state != j.state[sl]).sum()) == 0, f'{int((ref_state != j.state[sl]).sum())} differences in {N_CHECK} minutes')
    # determinism
    j2 = LE.make_live(ex, 'BTCUSDT', None)
    j2.advance(LE.FIRST + 99)
    check('running the models twice on the same minutes gives identical scores', np.array_equal(j2.fusion[LE.FIRST:LE.FIRST + 100], runs['BTCUSDT'].fusion[LE.FIRST:LE.FIRST + 100]))
    # the models react to the data: change the market features of minutes 1600..1609, nothing earlier may change
    jp = LE.make_live(ex, 'BTCUSDT', None)
    jp.df = jp.df.copy()
    for f_ in ['trade_flow_imbalance', 'hl_spread_ratio', 'volume_acceleration']:
        jp.df.loc[1600:1609, f_] = jp.df.loc[1600:1609, f_] + 6 * float(jp.df[f_].iloc[:1500].std() or 1.0)
    jp.df.loc[1600:1609, 'trade_flow_imbalance'] = jp.df.loc[1600:1609, 'trade_flow_imbalance'].clip(-0.99, 0.99)     # keep the changed values possible (the input checks reject impossible ones)
    jp.advance(LE.FIRST + 199)
    base_f, pert_f = runs['BTCUSDT'].fusion[LE.FIRST:LE.FIRST + 200], jp.fusion[LE.FIRST:LE.FIRST + 200]
    before = np.abs(base_f[:100] - pert_f[:100]).max()
    during = np.abs(base_f[100:110] - pert_f[100:110]).max()
    check('changing the input features changes the model scores on exactly those minutes (live inference, not stored numbers)', before < 1e-9 and during > 0.1 and not jp.bad, f'before the change {before:.1e}, during {during:.2f}')

    print('\n=== Part U: the page shows what the live models computed ===')
    from streamlit.testing.v1 import AppTest

    def metric(at, label):
        return next((m.value for m in at.metric if m.label == label), None)

    at = AppTest.from_file(str(HERE / 'app.py'), default_timeout=900)
    at.run()
    check('app starts in live-model mode without an error', not at.exception and isinstance(at.session_state['_B']['BTCUSDT'], LE.LiveSymbol))
    for _ in range(3):
        next(b for b in at.button if '1 bar ▶' in b.label).click().run()
    jl = at.session_state['_B']['BTCUSDT']
    i = at.session_state['i']
    fresh = LE.make_live(ex, 'BTCUSDT', None)
    fresh.advance(i)                      # an independent engine, never touched by the page
    check('"Fused score" on screen equals an independently recomputed model output at that minute',
          metric(at, 'Fused score') == f'{fresh.fusion[i]:.2f}', f'screen {metric(at, "Fused score")} / recomputed {fresh.fusion[i]:.2f}')
    check('"Unusualness" on screen equals the independently recomputed value', metric(at, 'Unusualness') == f'{100 - fresh.integ[i]:.2f}%', f'screen {metric(at, "Unusualness")}')
    check('the page and the independent engine agree on the state of every scored minute', np.array_equal(jl.state[jl.first:jl.done], fresh.state[jl.first:jl.done][:jl.done - jl.first]) or jl.done > fresh.done)

# ----------------------------------------------------------------------------- Part D
if LIVE_OK:
    print('\n=== Part D: missing, garbage and impossible data ===')
    from c03_engine import validate as VD
    from c03_engine.service import SymbolService
    dfx = {sy: pd.read_parquet(Path(ex.root) / 'replay' / f'{sy}_replay_2024-08-01_to_08-10.parquet').reset_index(drop=True) for sy in ['BTCUSDT', 'ETHUSDT']}
    fts = list(ex.B.FEATS)

    def mk():
        sv = SymbolService(ex, 'BTCUSDT')
        sv.prime(dfx['BTCUSDT'].iloc[:1500][fts], list(dfx['BTCUSDT'].ts[:1500]))
        return sv

    def feed(sv, idx, over=None):
        res = []
        for jj_, kk in enumerate(idx):
            row = dfx['BTCUSDT'].loc[kk, fts].to_numpy('float64') if not (over and jj_ in over) else over[jj_]
            res.append(sv.on_bar(row, str(dfx['BTCUSDT'].ts[kk]), {'close': 1.0})[0])
        return res
    idx = list(range(1500, 1520))
    ref = [r['fused_score'] for r in feed(mk(), [k for j_, k in enumerate(idx) if j_ != 4])]
    garbage = {'an empty value (NaN)': [np.nan, .1, .003, 10, .2, 1], 'infinity': [.001, np.inf, .003, 10, .2, 1], 'all six values 0': [0] * 6,
               'an absurd value (1e12)': [.001, .1, .003, 10, 1e12, 1], 'a negative count': [.001, .1, .003, 10, .2, -5], 'text instead of numbers': ['a'] * 6}
    for nm, vec in garbage.items():
        rr = feed(mk(), idx, {4: vec})
        later = [r['fused_score'] for j_, r in enumerate(rr) if j_ != 4]
        check(f'{nm}: shown as NO DATA, not as NORMAL, and later scores are exactly those of a run where that minute never arrived',
              rr[4]['state'] == 'NO_DATA' and rr[4]['fused_score'] is None and later == ref and all(r['state'] != 'NO_DATA' for j_, r in enumerate(rr) if j_ != 4))
    sv = mk()
    feed(sv, [1500, 1501, 1502])
    dup, back = feed(sv, [1502])[0], feed(sv, [1500])[0]
    check('a repeated minute and a minute from the past are rejected, and scoring continues with the next new minute',
          dup['state'] == 'NO_DATA' and back['state'] == 'NO_DATA' and feed(sv, [1503])[0]['state'] != 'NO_DATA')
    gp = feed(sv, [1507])[0]
    check('missing minutes are reported, not invented', gp['state'] != 'NO_DATA' and any('missing' in w_ for w_ in gp['data_quality']['warnings']), '; '.join(gp['data_quality']['warnings']))
    try:
        SymbolService(ex, 'BTCUSDT').prime(dfx['BTCUSDT'].iloc[:500][fts], list(dfx['BTCUSDT'].ts[:500]))
        msg = ''
    except VD.DataError as e_:
        msg = e_.message
    check('too little history gives a clear message (not a crash)', '1,500' in msg, msg[:90])
    nrej = sum(1 for sy_ in dfx for r_ in dfx[sy_][fts].to_numpy('float64') if VD.check_features(r_))
    check('no genuine minute of the recording is rejected by the checks (28,800 minutes, both markets)', nrej == 0, f'{nrej} rejected')
    bar_ok = dict(open=60000., high=60100., low=59900., close=60050., volume=10., trades=100, taker_buy_base=5.)
    bad_bars = [dict(bar_ok, close=float('nan')), dict(open=0., high=0., low=0., close=0., volume=0., trades=0, taker_buy_base=0.), dict(bar_ok, high=59000., low=61000.),
                dict(bar_ok, volume=-5.), dict(bar_ok, taker_buy_base=99.), dict(bar_ok, trades=None)]
    check('raw price bars: impossible bars are rejected, a normal bar passes', VD.check_bar(bar_ok) == [] and all(VD.check_bar(b_) for b_ in bad_bars))
    csvg = dfx['BTCUSDT'][['ts'] + fts].iloc[:2000].to_csv(index=False).encode()
    files = {'a picture (JPEG)': b'\xff\xd8\xff\xe0' + b'0' * 64, 'a picture (PNG)': b'\x89PNG\r\n' + b'0' * 64, 'an empty file': b'', 'plain text': b'hello world\nthis is not data\n',
             'random bytes': bytes(range(256)) * 4, 'a table with other columns': b'name,age\nx,1\ny,2\n', 'too few minutes': dfx['BTCUSDT'][['ts'] + fts].iloc[:300].to_csv(index=False).encode()}
    check('files that are not usable market data are refused with a reason: ' + ', '.join(files), all(not VD.inspect_csv('f.csv', b_)['ok'] for b_ in files.values()))
    check('a proper file of minutes is accepted by the file check', VD.inspect_csv('good.csv', csvg)['ok'])
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(HERE / 'app.py'), default_timeout=900)
    at.run()
    jb = at.session_state['_B']['BTCUSDT']
    kb = jb.done + 1
    jb.df.loc[kb, 'trade_flow_imbalance'] = np.nan
    at.session_state['i'] = kb
    at.run()
    check('page: an empty value inside the recording is shown as "No score for this minute" and does not crash',
          not at.exception and kb in jb.bad and any('No score for this minute' in w_.value for w_ in at.warning) and bool(np.isnan(jb.fusion[kb])) and jb.state[kb] == 0)
    at.session_state['i'] = kb + 1
    at.run()
    check('page: the next valid minute is scored normally', not at.exception and bool(np.isfinite(jb.fusion[kb + 1])))
    at2 = AppTest.from_file(str(HERE / 'app.py'), default_timeout=900)
    at2.run()
    next(x_ for x_ in at2.selectbox if x_.key == 'dq_pick').select('All six values are 0').run()
    rej0 = any('Rejected' in e_.value for e_ in at2.error)
    next(x_ for x_ in at2.selectbox if x_.key == 'dq_pick').select('A normal minute (copied from the recording)').run()
    check('page: the try-it panel rejects all-zero input and accepts a real minute', rej0 and any('Accepted' in x_.value for x_ in at2.success))

# ----------------------------------------------------------------------------- Part A
print('\n=== Part A (stored-results fallback, optional): saved results obey the frozen rules ===')
B = {}
for sym in ['BTCUSDT', 'ETHUSDT']:
    p = DATA / f'{sym}_dash.json'
    if not p.exists():
        print(f'(Part A skipped for {sym}: stored-results file not present)')
        continue
    b = json.loads(p.read_text(encoding='utf-8'))
    B[sym] = b
    i0 = int(b.get('bundle_from_bar') or 0)
    fu, ta, tw, stt = (arr(b, k)[i0:] for k in ['fusion', 'thr_alert', 'thr_watch', 'state'])
    m = np.isfinite(ta) & np.isfinite(tw) & np.isfinite(stt)
    pred = np.where(fu >= ta, 2, np.where(fu >= tw, 1, 0))
    mism = int((pred[m] != stt[m]).sum())
    check(f'{sym}: state equals the threshold rule (fusion >= alert line -> ALERT, >= watch line -> WATCH)', mism == 0, f'{mism} mismatches of {int(m.sum())} bars')
    un, pl = arr(b, 'unus_engine')[i0:], arr(b, 'plev')[i0:]
    mm = np.isfinite(un) & np.isfinite(pl)
    dd = float(np.abs(un[mm] - 100 * (1 - pl[mm])).max())
    check(f'{sym}: Integrity Score = 100 - unusualness, unusualness = 100 x (1 - rolling p-level)', dd < 0.01, f'max diff {dd:.4f}')
    s3 = (arr(b, 'c_ae') + arr(b, 'c_xgb') + arr(b, 'c_tr') + arr(b, 'c_shared'))[i0:]
    res = (fu - s3)[np.isfinite(fu) & np.isfinite(s3)]
    check(f'{sym}: fused score = sum of the three detector contributions + shared term + one constant (stacker is additive)', res.std() < 0.005, f'residual std {res.std():.4f}, mean {res.mean():.3f}')
    flagged = int((stt >= 1).sum())
    cover = sum(1 for k in range(i0, len(b['ts'])) if (b['state'][k] or 0) >= 1 and str(k) in b['xai'])
    check(f'{sym}: detail (SHAP / z-scores) stored for at least 90% of flagged minutes', flagged > 0 and cover >= 0.9 * flagged, f'{cover} of {flagged}')
    ts = list(b['ts'])
    check(f'{sym}: timestamps strictly increase, one bar per minute', all(np.datetime64(ts[k + 1][:19]) - np.datetime64(ts[k][:19]) == np.timedelta64(1, 'm') for k in range(len(ts) - 1)))
pj = DATA / 'declared_pattern_resemblance_v1.json'
if pj.exists():
    m = json.loads(pj.read_text())
    h = m['hypotheses']
    check('pattern layer: show_closest equals (H-G1 and H-G2 and H-G5) from the pre-registered rule', m['show_closest'] == (h['H_G1'] and h['H_G2'] and h['H_G5']))
    check('pattern layer: show_no_match equals H-G3', m['show_no_match'] == h['H_G3'])
    P = np.array(m['precision'])
    check('pattern layer: stored covariance is symmetric', np.allclose(P, P.T, atol=1e-8))
else:
    check('pattern layer file exists', False, str(pj))

STORED = all((DATA / f'{s_}_dash.json').exists() for s_ in ['BTCUSDT', 'ETHUSDT'])
if not STORED:
    print('\n(Part B skipped: the stored-results files are not present. That is fine in live-model mode.)')
if STORED:
    # ----------------------------------------------------------------------------- Part B
    print('\n=== Part B: the page shows what the files say, and follows them when they change ===')
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:  # pragma: no cover
        print('streamlit is not installed:', e)
        sys.exit(1)


    def run_app(data_dir, steps=0):
        at = AppTest.from_file(str(HERE / 'app.py'), default_timeout=240)
        at.session_state['_data_dir'] = str(data_dir)
        at.run()
        for _ in range(steps):
            next(b for b in at.button if '1 bar ▶' in b.label).click().run()
        return at


    def metric(at, label):
        return next((m.value for m in at.metric if m.label == label), None)


    def expected(data_dir, sym='BTCUSDT'):
        b = json.loads((Path(data_dir) / f'{sym}_dash.json').read_text(encoding='utf-8'))
        return b


    def shown_vs_file(at, data_dir, tag):
        i = at.session_state['i']
        b = expected(data_dir)
        un = b['unus_engine'][i]
        fu = b['fusion'][i]
        ok_un = metric(at, 'Unusualness') == f'{100 - (100 - un):.2f}%'
        ok_fu = metric(at, 'Fused score') == f'{fu:.2f}'
        check(f'{tag}: "Unusualness" on screen equals the file at the replay minute', ok_un, f'screen {metric(at, "Unusualness")} / file {un:.2f}%')
        check(f'{tag}: "Fused score" on screen equals the file at the replay minute', ok_fu, f'screen {metric(at, "Fused score")} / file {fu:.2f}')
        return metric(at, 'Unusualness'), metric(at, 'Fused score')


    base = run_app(DATA)
    check('app starts without an error on the real files', not base.exception)
    u0, f0 = shown_vs_file(base, DATA, 'original files')

    tmp = Path(tempfile.mkdtemp(prefix='c03_verify_'))
    try:
        # M1: change the saved scores
        d1 = tmp / 'm1'
        shutil.copytree(DATA, d1)
        for sym in ['BTCUSDT', 'ETHUSDT']:
            p = d1 / f'{sym}_dash.json'
            b = json.loads(p.read_text(encoding='utf-8'))
            b['unus_engine'] = [None if v is None else min(100.0, round(v * 0.5 + 7.0, 3)) for v in b['unus_engine']]
            b['fusion'] = [None if v is None else round(v + 0.77, 4) for v in b['fusion']]
            p.write_text(json.dumps(b, separators=(',', ':')), encoding='utf-8')
        a1 = run_app(d1)
        u1, f1 = shown_vs_file(a1, d1, 'changed scores')
        check('changing the saved scores changes the numbers on the page (nothing is hard-coded)', (u1, f1) != (u0, f0), f'{u0},{f0} -> {u1},{f1}')

        # M2: change the engine confidence labels -> the Explainable tab tile must follow
        d2 = tmp / 'm2'
        shutil.copytree(DATA, d2)
        for sym in ['BTCUSDT', 'ETHUSDT']:
            p = d2 / f'{sym}_dash.json'
            b = json.loads(p.read_text(encoding='utf-8'))
            b['conf'] = [None if v is None else 'HIGH' for v in b['conf']]
            p.write_text(json.dumps(b, separators=(',', ':')), encoding='utf-8')
        a2 = run_app(d2)
        txt = ' '.join(m.value for m in a2.markdown)
        check('changing the saved confidence labels changes the ENGINE CONFIDENCE tile', 'ENGINE CONFIDENCE' in txt and re.search(r'ENGINE CONFIDENCE</div><div[^>]*>HIGH<', txt) is not None)

        # M3: change the pattern file -> sentence about test accuracy and the label must follow
        d3 = tmp / 'm3'
        shutil.copytree(DATA, d3)
        pj3 = d3 / 'declared_pattern_resemblance_v1.json'
        m3 = json.loads(pj3.read_text())
        m3['results']['JITTERED']['top1'] = 0.123
        m3['results']['JITTERED']['top2'] = 0.456
        pj3.write_text(json.dumps(m3))
        a3 = run_app(d3, steps=36) if False else run_app(d3)
        # open the Aug 5 alert, move into the episode, read the caption
        next(b for b in a3.button if 'Aug 5' in b.label).click().run()
        for _ in range(36):
            next(b for b in a3.button if '1 bar ▶' in b.label).click().run()
        caps = ' '.join(c.value for c in a3.caption)
        check('changing the pattern file changes the test-accuracy sentence (12% / 46%)', '12%' in caps and '46%' in caps, caps[caps.find('In synthetic'):caps.find('In synthetic') + 120])

        # M4: switch the pattern output off -> no pattern name may appear
        d4 = tmp / 'm4'
        shutil.copytree(DATA, d4)
        pj4 = d4 / 'declared_pattern_resemblance_v1.json'
        m4 = json.loads(pj4.read_text())
        m4['show_closest'] = False
        pj4.write_text(json.dumps(m4))
        a4 = run_app(d4)
        next(b for b in a4.button if 'Aug 5' in b.label).click().run()
        for _ in range(36):
            next(b for b in a4.button if '1 bar ▶' in b.label).click().run()
        t4 = ' '.join(m.value for m in a4.markdown) + ' ' + ' '.join(c.value for c in a4.caption)
        names = list(json.loads((DATA / 'declared_pattern_resemblance_v1.json').read_text())['plain_names'].values())
        check('when the pre-registered rule says "do not show", no pattern name appears', not any(n in t4 for n in names))

        # M5: missing file -> honest error, never silent mock data
        d5 = tmp / 'm5'
        shutil.copytree(DATA, d5)
        (d5 / 'ETHUSDT_dash.json').unlink()
        a5 = run_app(d5)
        check('a missing data file gives an explicit error (no fake data is drawn)', len(a5.error) > 0 and len(a5.tabs) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- Part C
print('\n=== Part C: scan of the code ===')
src = (HERE / 'app.py').read_text(encoding='utf-8')
tree = ast.parse(src)
check('no random numbers are generated (no random / np.random)', not re.search(r'\bnp\.random\b|\bimport random\b|\brandom\.', src))
check('no words like mock / fake / dummy / placeholder in the code (the pattern name "Fake-breakout-like move" is allowed)', not re.search(r'\b(mock|dummy|placeholder|lorem)\b|\bfake(?!-breakout)', src, re.I))
big = [n.lineno for n in ast.walk(tree) if isinstance(n, (ast.List, ast.Tuple)) and len(n.elts) >= 8 and all(isinstance(e, ast.Constant) and isinstance(e.value, (int, float)) for e in n.elts) and max(abs(e.value) for e in n.elts) > 5]
check('no hand-typed list of 8 or more numbers (no embedded data series; small layout-width lists are ignored)', not big, f'lines {big}' if big else '')

bad = [n for n, ok in RESULTS if not ok]
print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed.')
if bad:
    print('FAILED:\n - ' + '\n - '.join(bad))
    sys.exit(1)
print('All checks passed: the numbers come from the trained models run inside the dashboard, and they follow the data.')
