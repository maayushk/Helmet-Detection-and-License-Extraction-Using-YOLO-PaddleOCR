@echo off
echo =======================================
echo   HELMET DETECTION - INSTALLATION
echo =======================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo Please install Python 3.9, 3.10 or 3.11 from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b
)

echo [OK] Python found.
echo.

:: Detect GPU
echo Detecting GPU...
echo.

nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected.
    echo Installing CPU-only PaddlePaddle (most compatible with all CUDA versions)...
    echo Note: YOLO will still use your NVIDIA GPU automatically via PyTorch.
    goto :install
)

echo [INFO] No NVIDIA GPU detected (AMD / Intel / No GPU).
echo Installing CPU versions of all packages...
echo All features will work — processing will be slightly slower.
echo.

:install
echo.
echo [1/6] Installing Streamlit...
pip install streamlit --quiet

echo [2/6] Installing OpenCV...
pip install opencv-python --quiet

echo [3/6] Installing NumPy...
pip install "numpy>=1.24,<2.0" --quiet

echo [4/6] Installing Pandas and Pillow...
pip install pandas Pillow --quiet

echo [5/6] Installing Ultralytics (YOLOv8)...
pip install ultralytics --quiet

echo [6/6] Installing PaddlePaddle and PaddleOCR...
pip install paddlepaddle==2.6.2 --quiet
pip install paddleocr==2.7.3 --quiet

echo.
echo =======================================
echo   Installation Complete!
echo   Now double-click run.bat to start.
echo =======================================
echo.
pause
