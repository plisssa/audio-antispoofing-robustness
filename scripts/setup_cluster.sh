#!/usr/bin/env bash
set -euo pipefail

module purge
PYTORCH_ENV="${PYTORCH_ENV:-/opt/software/python/envs/pytorch2_4}"
[ -x "$PYTORCH_ENV/bin/python" ] || { echo "base python env missing: $PYTORCH_ENV" >&2; exit 1; }
export PATH="$PYTORCH_ENV/bin:$PATH"

mkdir -p runs data

"$PYTORCH_ENV/bin/python" -m venv --system-site-packages .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install --only-binary=:all: soundfile scikit-learn pandas pyyaml
./.venv/bin/python -m pip install -e . --no-deps

echo "core installed:"
./.venv/bin/python -c "import numpy,scipy,soundfile,sklearn,pandas,torch,adfd; print('  ok numpy',numpy.__version__,'torch',torch.__version__)"
