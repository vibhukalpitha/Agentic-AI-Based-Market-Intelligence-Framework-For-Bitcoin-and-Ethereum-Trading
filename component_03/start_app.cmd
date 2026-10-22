@echo off
rem TradeSense C03: Streamlit app on the live engine. Run second, in another window.
set "HERE=%~dp0"
set "C03_ASSETS=%HERE%backend\data_local\n3n4_assets"
set "C03_API=http://127.0.0.1:8000"
set "C03_START_SRC=live"
cd /d "%HERE%frontend_streamlit"
python -m streamlit run app.py
