import argparse, uvicorn
from .api import create_app

p = argparse.ArgumentParser(description='C03 IntegritySense engine + API (monitor-only, public data)')
p.add_argument('--export', required=True, help='path to the engine_export_v1 folder'); p.add_argument('--mode', choices=['replay', 'live'], default='replay')
p.add_argument('--from', dest='frm', default='2024-08-05T00:00:00', help='replay start (UTC)'); p.add_argument('--to', default=None); p.add_argument('--speed', type=float, default=60, help='replay speed (60 = 1 bar per second)')
p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8000); a = p.parse_args()
app = create_app(a.export, a.mode, replay_default={'from': a.frm, 'to': a.to, 'speed': a.speed} if a.mode == 'replay' else None)
uvicorn.run(app, host=a.host, port=a.port)
