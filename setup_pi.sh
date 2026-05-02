#!/usr/bin/env bash
# Setup script for Raspberry Pi 4/5 (64-bit OS, aarch64)
# Run once: bash setup_pi.sh
set -e

echo "[1/4] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libopenblas-dev libatlas-base-dev

echo "[2/4] Creating virtual environment at ~/attention_venv ..."
python3 -m venv ~/attention_venv
source ~/attention_venv/bin/activate

echo "[3/4] Installing Python packages (this will take a while on Pi)..."
pip install --upgrade pip wheel
# Install mediapipe with no-deps first (avoids slow jaxlib download), then deps
pip install --no-deps mediapipe==0.10.14
pip install numpy opencv-python filterpy scipy
# ultralytics pulls in torch/torchvision for aarch64 automatically
pip install ultralytics

echo "[4/4] Done!"
echo ""
echo "To run:"
echo "  source ~/attention_venv/bin/activate"
echo "  python run_video.py --source 0"
echo ""
echo "If using the Pi Camera Module (CSI) instead of USB webcam:"
echo "  sudo modprobe bcm2835-v4l2   # legacy camera stack"
echo "  python run_video.py --source 0"
