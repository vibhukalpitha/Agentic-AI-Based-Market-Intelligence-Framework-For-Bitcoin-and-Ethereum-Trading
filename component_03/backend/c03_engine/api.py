"""FastAPI + WebSocket service (api_contract v0.1). Modes: 'replay' (saved bars through the same engine) and 'live' (Binance public streams)."""
import asyncio, json, time, datetime as dt
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .store import Export
from .service import SymbolService, iso

VERSION = '0.1.0'


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, set): return sorted(o)
    return o


class Hub:
    def __init__(self): self.clients = {}
    async def send(self, msg):
        data = json.dumps(clean(msg)); sym = (msg.get('data') or {}).get('symbol')
        for ws, syms in list(self.clients.items()):
            if sym is not None and syms and sym not in syms: continue
            try: await ws.send_text(data)
            except Exception: self.clients.pop(ws, None)


def create_app(export_root, mode='replay', verify=True, replay_default=None):
    ex = Export(export_root, verify=verify); hub = Hub(); S = {'svc': {}, 'mode': mode, 'replay': None, 'last_bar': None, 'task': None, 'started': time.time()}
    man_sha = ex.manifest.get('library_sha16', '')

    def fresh(): return {s: SymbolService(ex, s) for s in ex.symbols}

    async def run_replay(sym_list, start, end, speed):
        """Replay saved bars through a fresh set of engines (same code as live)."""
        svc = fresh(); S['svc'] = svc; S['mode'] = 'replay'; S['replay'] = {'from': start, 'to': end, 'speed': speed, 'running': True}
        dfs = {s: pd.read_parquet(Path(export_root) / 'replay' / f'{s}_replay_2024-08-01_to_08-10.parquet') for s in sym_list}
        idx = {}
        for s, df in dfs.items():
            i0 = int(np.searchsorted(df.ts.str.replace(' ', 'T').to_numpy(), start)); i1 = int(np.searchsorted(df.ts.str.replace(' ', 'T').to_numpy(), end)) if end else len(df); i0 = max(i0, 1500); idx[s] = (i0, max(i1, i0 + 1))
            await asyncio.to_thread(svc[s].prime, df.iloc[:i0][ex.B.FEATS], list(df.ts[:i0]))
        n = max(i1 - i0 for i0, i1 in idx.values()); delay = 60.0 / max(speed, 1e-6)
        try:
            for k in range(n):
                for s, df in dfs.items():
                    i0, i1 = idx[s]
                    if i0 + k >= i1: continue
                    row = df.iloc[i0 + k]; status, events = await asyncio.to_thread(svc[s].on_bar, row[ex.B.FEATS].to_numpy('float64'), row.ts, {'close': float(row.close)}); S['last_bar'] = time.time()
                    for e in events: await hub.send(e)
                await asyncio.sleep(delay)
        finally: S['replay']['running'] = False

    @asynccontextmanager
    async def lifespan(app):
        if mode == 'live':
            from .live import run_live
            S['task'] = asyncio.create_task(run_live(ex, S, hub, fresh))
        elif replay_default:
            S['task'] = asyncio.create_task(run_replay(ex.symbols, replay_default['from'], replay_default.get('to'), replay_default.get('speed', 60)))
        hb = asyncio.create_task(heartbeat())
        yield
        for t in (S['task'], hb):
            if t: t.cancel()

    async def heartbeat():
        while True:
            await asyncio.sleep(15); await hub.send({'type': 'heartbeat', 'data': {'server_utc': dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), 'last_bar_utc': next((v.last['bar_open_utc'] for v in S['svc'].values() if v.last), None)}})

    app = FastAPI(title='C03 IntegritySense', version=VERSION, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['*'], allow_headers=['*'])
    P = '/api/c03'

    def svc_of(symbol):
        if symbol not in S['svc']: raise HTTPException(404, {'error': {'code': 'unknown_symbol', 'message': f'{symbol} not available (is the engine warming up?)'}})
        return S['svc'][symbol]

    @app.get(P + '/health')
    def health():
        lb = [v.last['bar_open_utc'] for v in S['svc'].values() if v.last]
        return {'status': 'ok' if S['svc'] else 'warming_up', 'warmup': S.get('warmup'), 'last_error': S.get('last_error'), 'mode': S['mode'], 'engine_version': VERSION, 'manifest_sha16': man_sha, 'last_bar_utc': max(lb) if lb else None, 'data_age_sec': None if S['last_bar'] is None else round(time.time() - S['last_bar'], 1)}

    @app.get(P + '/status')
    def status(symbol: str = None):
        syms = [symbol] if symbol else list(S['svc'])
        out = {s: clean(svc_of(s).last) for s in syms if svc_of(s).last}
        if not out: raise HTTPException(503, {'error': {'code': 'warming_up', 'message': 'no scored bar yet'}})
        for v in out.values(): v['data_age_sec'] = None if S['last_bar'] is None else round(time.time() - S['last_bar'], 1)
        return out[symbol] if symbol else out

    @app.get(P + '/bars')
    def bars(symbol: str, frm: str = Query(None, alias='from'), to: str = None, limit: int = 500, detail: bool = False):
        b = list(svc_of(symbol).bars); b = [x for x in b if (not frm or x['bar_open_utc'] >= frm) and (not to or x['bar_open_utc'] <= to)]
        b = b[-min(limit, 2880):]
        return clean(b if detail else [{k: v for k, v in x.items() if k != 'x'} for x in b])      # v0.2.1: detail=true adds the per-bar engine detail 'x' (thresholds, z, contributions, reliability checks, reasons)

    @app.get(P + '/windows')
    def windows(symbol: str, horizon: str = '15m', frm: str = Query(None, alias='from'), to: str = None):
        if horizon not in ('15m', '1h'): raise HTTPException(400, {'error': {'code': 'bad_horizon', 'message': 'horizon must be 15m or 1h'}})
        return clean([w for w in svc_of(symbol).windows[horizon] if (not frm or w['window_start_utc'] >= frm) and (not to or w['window_start_utc'] <= to)])

    @app.get(P + '/alerts')
    def alerts(symbol: str = None, since: str = None, limit: int = 100):
        out = []
        for s in ([symbol] if symbol else list(S['svc'])): out += svc_of(s).episodes_view(since, limit)
        return clean(sorted(out, key=lambda e: e['start_utc'], reverse=True)[:limit])

    @app.get(P + '/day')
    def day(symbol: str, date: str): return clean(svc_of(symbol).day_view(date))

    @app.get(P + '/reliability')
    def reliability(symbol: str, limit: int = 120): return clean(svc_of(symbol).reliability_view(min(limit, 2880)))

    @app.get(P + '/explain')
    def explain(symbol: str, at: str = None):
        r = svc_of(symbol).explain(at)
        if r is None: raise HTTPException(404, {'error': {'code': 'no_explanation', 'message': 'only the latest scored bar and the last 300 flagged bars can be explained'}})
        return clean(r)

    @app.get(P + '/similar')
    def similar(symbol: str, k: int = 5, at: str = None): return clean(svc_of(symbol).similar(max(1, min(k, 10)), at))

    @app.get(P + '/model-info')
    def model_info():
        d = ex.decl; return {'models': {'ae': 'AE-ctx (A6, mean of 10 bars)', 'xgboost': d['final']['models']['XGBoost v2']['configuration'], 'transformer': d['final']['models']['Transformer v2']['configuration'], 'fusion': 'fusion v0 (logistic stacker)', 'tgnn': None},
                'alert_rule': 'fused score above its rolling 7-day quantile (2 genuine episodes/day)', 'watch_rule': 'lower threshold (never an alert)', 'test_results': json.loads((Path(export_root) / 'config' / 'test_b_results_v1.json').read_text()).get('macro8_budget2') if (Path(export_root) / 'config' / 'test_b_results_v1.json').exists() else None,
                'limitations': ['Labels are synthetic mechanisms; an alert is a statistical flag, not proof of manipulation.', 'Abnormal volume bursts are ambiguous at 1-minute level (shown as WATCH with a caveat).', 'hl_spread_ratio is a candle price range, not the bid-ask spread.', 'Order-book channel (TGNN) is not active yet.']}

    @app.post(P + '/replay/start')
    async def replay_start(body: dict):
        if S['task']: S['task'].cancel()
        syms = [body['symbol']] if body.get('symbol') else ex.symbols
        S['task'] = asyncio.create_task(run_replay(syms, body.get('from', '2024-08-05T00:00:00'), body.get('to'), float(body.get('speed', 60)))); return {'started': True}

    @app.post(P + '/replay/stop')
    async def replay_stop():
        if S['task']: S['task'].cancel()
        return {'stopped': True}

    @app.websocket('/ws/c03/stream')
    async def ws(websocket: WebSocket, symbols: str = ''):
        await websocket.accept(); hub.clients[websocket] = set(x for x in symbols.split(',') if x)
        try:
            while True:
                m = json.loads(await websocket.receive_text())
                if m.get('type') == 'subscribe': hub.clients[websocket] = set(m.get('symbols', []))
        except WebSocketDisconnect: hub.clients.pop(websocket, None)
        except Exception: hub.clients.pop(websocket, None)

    app.state.S, app.state.hub, app.state.ex = S, hub, ex
    return app
