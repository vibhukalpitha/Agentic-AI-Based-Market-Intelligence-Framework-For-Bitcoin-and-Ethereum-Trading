# C03 IntegritySense backend (monitor-only: public Binance data, no keys, no orders)

```
backend/
  c03_engine/      engine package (scoring core, features, live source, FastAPI)
  export/          engine_export_v1 from notebook 46 (models via Git LFS, manifest.json, config/, code/, replay/)
  tests/           golden test, feature test, live-pipeline test
```
## Run
```
pip install -r requirements.txt
python -m c03_engine --export export --mode replay --from 2024-08-05T00:00:00 --speed 60   # demo: replay of 5 Aug 2024 at 1 bar/s
python -m c03_engine --export export --mode live                                            # live Binance public streams (7-day warm-up first)
```
API: http://127.0.0.1:8000/docs  ·  WebSocket: ws://127.0.0.1:8000/ws/c03/stream?symbols=BTCUSDT,ETHUSDT  (see docs/api_contract.md)
## Tests
```
set C03_EXPORT=export          (Windows: set, Linux/Mac: export C03_EXPORT=export)
python -m pytest -q tests
```
`test_golden` proves bar-by-bar streaming equals the batch notebooks (models to 1e-5, fused score to 1e-3, states identical).
## Notes
- Every file in export/ is checked against manifest.json (sha256) at start-up.
- Live warm-up uses REST klines; large_trade_count history before start-up is 0 (not in klines), the live trade stream fills it.
- An alert is a statistical flag, not proof of manipulation.
