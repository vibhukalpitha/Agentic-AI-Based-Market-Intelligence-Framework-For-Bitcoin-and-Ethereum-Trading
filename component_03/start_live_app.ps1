# TradeSense C03: start the Streamlit app on the live engine. Run second, in another window.
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path          # component_03 folder
$env:C03_ASSETS = Join-Path $Here 'backend\data_local\n3n4_assets'
$env:C03_API = 'http://127.0.0.1:8000'
$env:C03_START_SRC = 'live'
Set-Location (Join-Path $Here 'frontend_streamlit')
python -m streamlit run app.py
