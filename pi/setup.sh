#!/usr/bin/env bash
# Raspberry Pi 4/5 setup — run once from this folder
# Requires: 64-bit Raspberry Pi OS Bookworm
set -e

echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-venv \
    libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender-dev \
    libopenblas-dev \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev git curl

echo "[2/5] Installing Python 3.11 via pyenv (mediapipe has no aarch64 wheel for 3.12+)..."
if ! command -v pyenv &>/dev/null; then
    curl -fsSL https://pyenv.run | bash
fi
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install -s 3.11.9   # -s = skip if already installed
pyenv local 3.11.9

echo "[3/5] Creating virtual environment with Python 3.11 at ~/attention_venv ..."
python -m venv ~/attention_venv
source ~/attention_venv/bin/activate

echo "[4/5] Installing Python packages (takes a while on Pi)..."
pip install --upgrade pip wheel
# mediapipe: force PyPI only (piwheels has no aarch64 wheel)
pip install --index-url https://pypi.org/simple/ 'mediapipe>=0.10.9,<0.11'
pip install numpy opencv-python filterpy scipy
# ultralytics auto-pulls torch/torchvision for aarch64
pip install ultralytics

echo "[5/5] Done!"
echo ""
echo "Add pyenv to your shell permanently by adding these lines to ~/.bashrc:"
echo '  export PYENV_ROOT="$HOME/.pyenv"'
echo '  export PATH="$PYENV_ROOT/bin:$PATH"'
echo '  eval "$(pyenv init -)"'
echo ""
echo "To run:"
echo "  source ~/attention_venv/bin/activate"
echo "  python run_video.py --source 0"
echo ""
echo "If using Pi Camera Module (CSI ribbon cable):"
echo "  sudo modprobe bcm2835-v4l2   # only needed once per boot"
echo "  python run_video.py --source 0"
