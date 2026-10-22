"""Replay parity: Streamlit stored bundle vs engine API, 2024-08-05 (UTC).
Compares, minute by minute, the fused score (tolerance 1e-4, because the stored bundle has 4 decimals) and the Integrity State (must be identical).
Needs: the engine running in REPLAY mode (not your soak server) and the two *_dash.json bundles of the Streamlit app.
Usage: python p_replay_parity.py --api http://127.0.0.1:8001 --data C:\\...\\streamlit_app\\data
Read-only: it only reads /bars and the json files. Writes replay_parity_result.json next to this script.
"""
import argparse, json, os, sys, time, urllib.request

TOL = 1e-4   # the stored bundles keep the fused score with 4 decimals (rounding error up to 5e-5), so 1e-5 cannot be measured against them; 1e-4 is the finest honest limit
SN = {0: 'NORMAL', 1: 'WATCH', 2: 'ALERT'}   # Streamlit bundle stores the state as 0/1/2
DAY0, DAY1 = '2024-08-05T00:00', '2024-08-06T00:00'


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def post(url, body):
    rq = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(rq, timeout=60) as r:
        return json.loads(r.read())


def norm(ts):                       # '2024-08-05 00:00:00+00:00' or '2024-08-05T00:00:00Z' -> '2024-08-05T00:00'
    return str(ts).replace(' ', 'T')[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default='http://127.0.0.1:8001')
    ap.add_argument('--data', required=True)
    ap.add_argument('--start', action='store_true', help='start a fast replay (speed 100000) from 2024-08-05 before comparing')
    a = ap.parse_args()
    base = a.api.rstrip('/') + '/api/c03'
    h = get(base + '/health')
    if h.get('mode') != 'replay':
        sys.exit(f"STOP: engine mode is '{h.get('mode')}', not 'replay'. Do not run this against the live soak server.")
    if a.start:
        post(base + '/replay/start', {'from': '2024-08-05T00:00:00', 'to': '2024-08-06T00:00:00', 'speed': 100000})
    res = {'tolerance': TOL, 'day': '2024-08-05', 'symbols': {}}
    ok_all = True
    for sym in ('BTCUSDT', 'ETHUSDT'):
        for _ in range(120):        # wait until the whole day is replayed
            bars = get(base + f'/bars?symbol={sym}&limit=3000')
            got = {norm(b['bar_open_utc']): b for b in bars}
            if sum(1 for k in got if DAY0 <= k < DAY1) >= 1440:
                break
            time.sleep(2)
        d = json.load(open(os.path.join(a.data, f'{sym}_dash.json')))
        n = mx = bad_state = n_over = 0
        worst = None
        for i, t in enumerate(d['ts']):
            k = norm(t)
            if not (DAY0 <= k < DAY1) or k not in got:
                continue
            b = got[k]
            if b.get('fused_score') is None or d['fusion'][i] is None:
                continue
            diff = abs(float(b['fused_score']) - float(d['fusion'][i]))
            n += 1
            n_over += diff > TOL
            if diff > mx:
                mx, worst = diff, k
            if str(b['state']).upper() != SN.get(d['state'][i], str(d['state'][i])):
                bad_state += 1
        ok = n >= 1400 and mx <= TOL and bad_state == 0
        ok_all &= ok
        res['symbols'][sym] = {'minutes_compared': n, 'max_abs_fused_diff': mx, 'at': worst, 'minutes_over_tolerance': int(n_over), 'state_mismatches': bad_state, 'pass': ok}
        print(f"{sym}: compared {n} minutes, max fused diff {mx:.2e} (at {worst}), minutes over {TOL:g}: {n_over}, state mismatches {bad_state} -> {'PASS' if ok else 'FAIL'}")
    res['pass'] = ok_all
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'replay_parity_result.json'), 'w'), indent=1)
    print('REPLAY PARITY', 'PASS' if ok_all else 'FAIL (report it as failed, do not edit the tolerance)')


if __name__ == '__main__':
    main()
