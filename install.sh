#!/usr/bin/env bash
# FlexSfM Studio installer for Linux.
#   ./install.sh            create ./.venv and install everything (GPU build of PyTorch when an NVIDIA driver is found)
#   ./install.sh --cpu      force the CPU build of PyTorch
#   VENV=/path ./install.sh install into another virtual environment
# No root needed. If no Python >= 3.10 is found, uv (https://docs.astral.sh/uv/) is used to fetch one into your home folder.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
VENV="${VENV:-.venv}"; FORCE_CPU=0
for a in "$@"; do case "$a" in --cpu) FORCE_CPU=1;; -h|--help) sed -n '2,6p' "$0"; exit 0;; esac; done
say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*"; exit 1; }

# ---------------------------------------------------------------- 1. Python >= 3.10
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY="$(command -v "$c")"; break; fi
done
UV="$(command -v uv || true)"; [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
if [ ! -x "$VENV/bin/python" ]; then
  if [ -n "$PY" ]; then
    say "creating $VENV with $PY ($("$PY" -V 2>&1))"; "$PY" -m venv "$VENV"
  else
    if [ -z "$UV" ]; then
      warn "no Python 3.10+ with venv found; installing uv into ~/.local/bin to fetch one (no root needed)"
      command -v curl >/dev/null || die "curl is needed to download uv - or install python3.10+ and python3-venv with your package manager"
      curl -LsSf https://astral.sh/uv/install.sh | sh; UV="$HOME/.local/bin/uv"
    fi
    say "creating $VENV with a uv-managed Python 3.12"; "$UV" venv --seed --python 3.12 "$VENV"
  fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools

# ---------------------------------------------------------------- 2. PyTorch matching the GPU driver
IDX="https://download.pytorch.org/whl/cpu"; KIND="CPU"
if [ "$FORCE_CPU" = 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
  CU="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | awk '{print $3}' || true)"
  if [ -n "$CU" ]; then
    MAJ="${CU%%.*}"; MIN="${CU##*.}"; V=$((MAJ * 100 + MIN))
    if   [ $V -ge 1208 ]; then IDX="https://download.pytorch.org/whl/cu128"; KIND="CUDA 12.8"
    elif [ $V -ge 1206 ]; then IDX="https://download.pytorch.org/whl/cu126"; KIND="CUDA 12.6"
    elif [ $V -ge 1108 ]; then IDX="https://download.pytorch.org/whl/cu118"; KIND="CUDA 11.8"
    else warn "NVIDIA driver supports only CUDA $CU - using the CPU build (update the driver for GPU speed)"; fi
  fi
fi
if python -c 'import torch, torchvision' 2>/dev/null; then
  say "PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
  say "installing PyTorch ($KIND build)"; python -m pip install torch torchvision --index-url "$IDX"
fi

# ---------------------------------------------------------------- 3. FlexSfM Studio + dependencies
command -v git >/dev/null || die "git is needed (FlexGlue and LightGlue are installed from GitHub): sudo apt install git"
say "installing FlexSfM Studio and its dependencies"
python -m pip install -e ".[gui]"

# ---------------------------------------------------------------- 4. check
say "checking the installation"
python -m flexsfm doctor --full || {
  warn "some checks failed - see the arrows above for the fix, then run:  $VENV/bin/python -m flexsfm doctor"; exit 1; }
cat <<EOF

Done. Start the app with   ./flexsfm.sh
or use the command line:   $VENV/bin/flexsfm --help
EOF
