#!/usr/bin/env bash
# Raspberry Pi 4/5 setup — run once from this folder
# Requires: 64-bit Raspberry Pi OS Bookworm
set -e

echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-venv python3.11 python3.11-venv \
    libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender-dev \
    libopenblas-dev

echo "[2/5] Creating virtual environment with Python 3.11 at ~/attention_venv ..."
# mediapipe requires Python 3.11 — it has no aarch64 wheel for 3.12+
python3.11 -m venv ~/attention_venv
source ~/attention_venv/bin/activate

echo "[3/5] Installing Python packages (takes a while on Pi)..."
pip install --upgrade pip wheel
# mediapipe: force PyPI only (piwheels has no aarch64 wheel)
pip install --index-url https://pypi.org/simple/ 'mediapipe>=0.10.9,<0.11'
pip install numpy opencv-python filterpy scipy
# ultralytics auto-pulls torch/torchvision for aarch64
pip install ultralytics

echo "[4/5] Verifying mediapipe import..."
python -c "import mediapipe; print('mediapipe OK:', mediapipe.__version__)"

echo "[5/5] Done!"
echo ""
echo "To run:"
echo "  source ~/attention_venv/bin/activate"
echo "  python run_video.py --source 0"
echo ""
echo "If using Pi Camera Module (CSI ribbon cable):"
echo "  sudo modprobe bcm2835-v4l2   # only needed once per boot"
echo "  python run_video.py --source 0"
