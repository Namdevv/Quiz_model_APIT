#!/usr/bin/env bash
set -euo pipefail

# ==== Config nhẹ ====
VENV_DIR="$HOME/venv"          # Đường dẫn venv
PROJECT_DIR="$HOME"            # Thư mục làm việc mặc định

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

echo
echo "✅ Setup hoàn tất!"
echo "• Kích hoạt venv: source \"$VENV_DIR/bin/activate\""