"""diag_parity.py - finds exactly WHERE the live models differ from the notebook reference (read-only, changes nothing)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import live_engine as LE

try:
    import xgboost, torch, pandas, numpy, scipy
    print('versions: xgboost', xgboost.__version__, '| torch', torch.__version__, '| pandas', pandas.__version__, '| numpy', numpy.__version__, '| python', sys.version.split()[0])
except Exception as e:
    print('version info:', e)

found, why = LE.find_backend(HERE)
if found is None:
    sys.exit('backend not found: ' + str(why))
sys.path.insert(0, str(found[0]))
from c03_engine.store import Export

ex = Export(found[1], verify=True)
N = 600
refs = {'fusion': 'exp_fusion', 'ae': 'exp_ae', 'xgb': 'exp_xgb', 'tr': 'exp_tr', 'thr_alert': 'exp_thr_alert', 'thr_watch': 'exp_thr_watch8',
        'c_ae': 'exp_contrib_ae', 'c_xgb': 'exp_contrib_xgb', 'c_tr': 'exp_contrib_tr', 'c_shared': 'exp_contrib_shared'}
for sym in ['BTCUSDT', 'ETHUSDT']:
    j = LE.make_live(ex, sym, None)
    j.advance(LE.FIRST + N - 1)
    sl = slice(LE.FIRST, LE.FIRST + N)
    print('\n==', sym, '==')
    for mine, ref in refs.items():
        a, r = getattr(j, mine)[sl], j.df[ref].to_numpy('float64')[sl]
        m = np.isfinite(a) & np.isfinite(r)
        d = np.abs(a[m] - r[m])
        idx = np.where(m)[0]
        nbig = int((d > 1e-3).sum())
        k = int(idx[d.argmax()]) if len(d) else -1
        print(f'{mine:9s} max diff {d.max():.2e}  minutes with diff>1e-3: {nbig:4d}  finite pairs {int(m.sum())}/{N}  worst at bar {LE.FIRST + k}')
        if nbig and mine in ('xgb', 'thr_alert', 'thr_watch'):
            bad = idx[d > 1e-3][:4]
            for b in bad:
                print(f'     bar {LE.FIRST + int(b)}: engine {a[b]:.5f} | notebook {r[b]:.5f}')
    ref_state = j.df.exp_state.map(LE.STATE_NUM).to_numpy()[sl]
    dif = np.where(ref_state != j.state[sl])[0]
    print('state differences:', len(dif))
    for b in dif[:6]:
        k = LE.FIRST + int(b)
        print(f'   bar {k}: engine {j.state[k]} (fusion {j.fusion[k]:.4f}, watch line {j.thr_watch[k]:.4f}, alert line {j.thr_alert[k]:.4f})'
              f' | notebook {ref_state[b]} (fusion {j.df.exp_fusion[k]:.4f}, watch line {j.df.exp_thr_watch8[k]:.4f}, alert line {j.df.exp_thr_alert[k]:.4f})')
