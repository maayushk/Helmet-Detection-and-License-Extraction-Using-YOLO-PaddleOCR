@echo off
echo =======================================
echo   HELMET VIOLATION DETECTION SYSTEM
echo =======================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9 or above from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b
)

:: Check if pip packages are installed, install if not
echo [1/2] Checking and installing required packages...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install packages. Check your internet connection.
    pause
    exit /b
)

echo [2/2] Starting Helmet Violation Detection System...
echo.
echo The app will open in your browser automatically.
echo If it doesn't open, go to: http://localhost:8501
echo.
echo To stop the app, close this window or press Ctrl+C
echo.

streamlit run app.py

pause
