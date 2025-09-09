#!/usr/bin/env bash
set -euo pipefail

# ==== Config nhẹ ====
VENV_DIR="$HOME/venv"          # Đường dẫn venv
PROJECT_DIR="$HOME"            # Thư mục làm việc mặc định
USE_GPU="auto"                 # "auto" | "yes" | "no"

echo "==> Updating apt..."
sudo apt update && sudo apt upgrade -y

echo "==> Installing system packages..."
sudo apt install -y python3 python3-pip python3-venv build-essential git

# Tạo venv nếu chưa có
if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# Kích hoạt venv
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing Jupyter (Notebook + Lab) inside venv..."
pip install notebook jupyterlab

echo "==> Installing core ML deps..."
pip install "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer sentencepiece protobuf


pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "==> Installing LLM training stack..."
pip install xformers==0.0.32.post2
pip install --no-deps bitsandbytes accelerate peft trl triton cut_cross_entropy unsloth_zoo
pip install unsloth
pip install "transformers==4.55.4"

echo
echo "✅ Setup hoàn tất!"
