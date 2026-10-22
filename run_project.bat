@echo off
title Market Intelligence Framework - Launcher
echo ============================================================
echo Starting All Services (Component 01, Main Backend, Frontend)
echo ============================================================

echo [1/3] Starting Component 01 Regime Service (Port 8001)...
start "Component 01 (Port 8001)" cmd /k "cd /d "%~dp0component_01" && python -W ignore -m uvicorn src.api.service:app --host 127.0.0.1 --port 8001"

timeout /t 3 /nobreak >nul

echo [2/3] Starting Main Backend AI Models (Port 8000)...
start "Main Backend (Port 8000)" cmd /k "cd /d "%~dp0Vibhu Kalpitha" && python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

echo [3/3] Starting Frontend React Dashboard (Port 5173)...
start "Frontend Dashboard (Port 5173)" cmd /k "cd /d "%~dp0frontend" && npm.cmd run dev"

timeout /t 2 /nobreak >nul
echo Opening Browser at http://localhost:5173 ...
start http://localhost:5173

echo.
echo ============================================================
echo All services have been launched in separate terminal windows!
echo Keep those windows open while using the application.
echo ============================================================
pause
