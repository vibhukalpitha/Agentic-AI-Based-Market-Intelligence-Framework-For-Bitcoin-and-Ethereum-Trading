# TradeSense C03: start the engine on the live Binance feed (public data, no keys). Run first. Keep this window open.
# Warm-up loads the last 7 days (a few minutes); the app shows "warming up" until it is done.
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path          # component_03 folder
$Backend = Join-Path $Here 'backend'
$env:C03_ASSETS = Join-Path $Backend 'data_local\n3n4_assets'
Set-Location $Backend
python -m c03_engine --mode live --export (Join-Path $Backend 'engine_export_v1')
