"""Adds the real calibration histograms (CAL-3 fused scores: genuine normal bars vs injected abnormal bars) to the dashboard bundle JSON files.
No model is run and nothing is trained. It reads the DECLARED conformal tables of notebook 49 (conformal_tables_v1.npz, calibration set = August 2024, written before any TEST bar was read).

    python tools/add_calibration.py --npz <path>/conformal_tables_v1.npz --data ../frontend_streamlit/data
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--npz', required=True); ap.add_argument('--data', required=True); ap.add_argument('--bins', type=int, default=40)
    a = ap.parse_args(); npz = Path(a.npz); T = np.load(npz); sha = hashlib.sha256(npz.read_bytes()).hexdigest()[:16]
    for sym in ['BTCUSDT', 'ETHUSDT']:
        norm, abn = np.asarray(T[f'{sym}_norm'], 'float64'), np.asarray(T[f'{sym}_abn'], 'float64')
        norm, abn = norm[np.isfinite(norm)], abn[np.isfinite(abn)]
        lo = float(np.percentile(np.concatenate([norm, abn]), 0.5)); hi = float(np.percentile(np.concatenate([norm, abn]), 99.5))
        edges = np.linspace(lo, hi, a.bins + 1)
        hn, _ = np.histogram(np.clip(norm, lo, hi), bins=edges, density=True); ha, _ = np.histogram(np.clip(abn, lo, hi), bins=edges, density=True)
        cal = {'edges': [round(float(x), 4) for x in edges], 'normal': [round(float(x), 5) for x in hn], 'abnormal': [round(float(x), 5) for x in ha],
               'n_normal': int(len(norm)), 'n_abnormal': int(len(abn)), 'source': 'CAL-3 (August 2024) fused scores, declared conformal tables of notebook 49', 'npz_sha16': sha}
        f = Path(a.data) / f'{sym}_dash.json'; d = json.loads(f.read_text()); d['calibration'] = cal
        f.write_text(json.dumps(d, separators=(',', ':'))); print(f'{sym}: calibration added ({len(norm)} normal, {len(abn)} abnormal bars) -> {f}')


if __name__ == '__main__':
    main()
