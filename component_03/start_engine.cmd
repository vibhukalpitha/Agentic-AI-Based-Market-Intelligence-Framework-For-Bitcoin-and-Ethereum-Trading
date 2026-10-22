@echo off
rem TradeSense C03: engine on the live Binance feed (public data, no keys). Run first, keep this window open.
set "HERE=%~dp0"
set "C03_ASSETS=%HERE%backend\data_local\n3n4_assets"
cd /d "%HERE%backend"
python -m c03_engine --mode live --export "%HERE%backend\engine_export_v1"
