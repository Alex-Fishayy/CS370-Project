#!/usr/bin/env bash
# Raspberry Pi 4/5 setup — run once from this folder
# Requires: 64-bit Raspberry Pi OS Bookworm (aarch64)
set -e

echo "[1/5] Installing system dependencies..."
# Recover from any previously interrupted dpkg run
sudo dpkg --configure -a
# Wait for any background apt/dpkg process to release the lock (auto-updater runs on boot)
echo "  Waiting for apt lock..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    sleep 2
done
sudo apt-get update -qq
sudo apt-get install -y \
    libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender-dev \
    libopenblas-dev wget curl unzip

echo "[2/5] Installing Miniforge (pre-built Python 3.11 for aarch64, no compile needed)..."
MINIFORGE_INSTALLER="$HOME/Miniforge3-aarch64.sh"
MINIFORGE_ROOT="$HOME/miniforge3"

if ! [ -x "$MINIFORGE_ROOT/bin/conda" ]; then
    # Remove any partial/broken install before re-installing
    rm -rf "$MINIFORGE_ROOT"
    wget -q --show-progress \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh" \
        -O "$MINIFORGE_INSTALLER"
    bash "$MINIFORGE_INSTALLER" -b -p "$MINIFORGE_ROOT"
    rm "$MINIFORGE_INSTALLER"
fi

CONDA="$MINIFORGE_ROOT/bin/conda"
PYTHON311="$MINIFORGE_ROOT/envs/attention/bin/python"

echo "[3/5] Creating conda env 'attention' with Python 3.11..."
if ! "$CONDA" env list | grep -q "^attention "; then
    "$CONDA" create -n attention python=3.11 -y
fi

# Confirm 3.11
"$PYTHON311" --version | grep -q "3.11" || { echo "ERROR: Python 3.11 not found in conda env"; exit 1; }

echo "[4/5] Installing Python packages..."
PIP="$MINIFORGE_ROOT/envs/attention/bin/pip"
# PIP_CONFIG_FILE=/dev/null blocks /etc/pip.conf which adds piwheels as extra-index
# (--no-extra-index-url was removed in pip 26)
export PIP_CONFIG_FILE=/dev/null
"$PIP" install --upgrade pip wheel
# mediapipe: PyPI only, pin to 0.10.9 (avoids jax dependency added in later versions)
"$PIP" install \
    --index-url https://pypi.org/simple/ \
    'mediapipe==0.10.9'
"$PIP" install numpy opencv-python filterpy scipy
# tflite-runtime replaces ultralytics — no PyTorch, tiny wheel (~5MB)
"$PIP" install tflite-runtime

echo "  Downloading SSD MobileNet V1 COCO TFLite model (~4MB)..."
mkdir -p models
wget -q --show-progress \
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip" \
    -O models/detect.zip
unzip -o models/detect.zip detect.tflite -d models/
rm models/detect.zip

echo "[5/5] Done!"
echo ""
echo "To run:"
echo "  source ~/miniforge3/bin/activate attention"
echo "  python run_video.py --source 0"
echo ""
echo "If using Pi Camera Module (CSI ribbon cable):"
echo "  sudo modprobe bcm2835-v4l2   # only needed once per boot"
echo "  python run_video.py --source 0"
