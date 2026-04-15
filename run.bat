@echo off
echo =======================================
echo   HELMET VIOLATION DETECTION SYSTEM
echo =======================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo Please run install.bat first.
    pause
    exit /b
)

:: Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Required packages not installed.
    echo Please run install.bat first.
    pause
    exit /b
)

:: Check if model files exist
if not exist "yolo_helmet_model.pt" (
    echo [ERROR] yolo_helmet_model.pt not found.
    echo Make sure model files are in the same folder as this file.
    pause
    exit /b
)

if not exist "plate_model.pt" (
    echo [ERROR] plate_model.pt not found.
    echo Make sure model files are in the same folder as this file.
    pause
    exit /b
)

echo [OK] All checks passed.
echo.
echo Starting app... It will open in your browser automatically.
echo If it doesn't open, go to: http://localhost:8501
echo.
echo To stop the app, close this window or press Ctrl+C
echo.

streamlit run app.py
pause
