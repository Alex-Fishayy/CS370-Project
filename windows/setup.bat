@echo off
REM Windows setup — run once from this folder
REM Requires Python 3.11 installed and on PATH

echo [1/3] Creating virtual environment at C:\venv ...
python -m venv C:\venv

echo [2/3] Installing dependencies ...
C:\venv\Scripts\pip install --upgrade pip
C:\venv\Scripts\pip install --no-deps mediapipe==0.10.14
C:\venv\Scripts\pip install opencv-python numpy filterpy scipy ultralytics
REM PyTorch CPU build (smaller, sufficient for running)
C:\venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo [3/3] Done!
echo.
echo To run:
echo   C:\venv\Scripts\python run_video.py --source 0
pause
