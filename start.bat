@echo off
chcp 65001 >nul 2>&1
title Image Retrieval System
cd /d "%~dp0"
echo ============================================
echo   Starting server... Do not close this window.
echo ============================================
echo.
echo   Browser will open http://127.0.0.1:8000
echo.
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
