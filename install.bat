@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo ============================================
echo   Installing dependencies...
echo ============================================
echo.
pip install --no-index --find-links=wheels fastapi uvicorn python-multipart Pillow colorama
if %errorlevel% neq 0 (
    echo.
    echo Installation failed! Please check Python is installed correctly.
    pause
    exit /b 1
)
echo.
echo ============================================
echo   Done! Double-click start.bat to run.
echo ============================================
pause
