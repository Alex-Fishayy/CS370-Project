#!/usr/bin/env bash
# Raspberry Pi 4/5 setup — run once from this folder
# Requires: 64-bit Raspberry Pi OS, Python 3.11
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

echo "[3/4] Installing Python packages (takes a while on Pi)..."
pip install --upgrade pip wheel
# mediapipe: no-deps first to skip slow jaxlib download, then install deps manually
pip install --no-deps mediapipe==0.10.14
pip install numpy opencv-python filterpy scipy
# ultralytics auto-pulls torch/torchvision for aarch64
pip install ultralytics

echo "[4/4] Done!"
echo ""
echo "To run:"
echo "  source ~/attention_venv/bin/activate"
echo "  python run_video.py --source 0"
echo ""
echo "If using Pi Camera Module (CSI ribbon cable):"
echo "  sudo modprobe bcm2835-v4l2   # only needed once per boot"
echo "  python run_video.py --source 0"
