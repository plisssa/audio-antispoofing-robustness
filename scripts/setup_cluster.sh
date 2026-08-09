#!/usr/bin/env bash
set -euo pipefail

module purge
module load Python/PyTorch_GPU_v2.4

mkdir -p runs data

python -m venv --system-site-packages .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install --only-binary=:all: soundfile scikit-learn pandas pyyaml
./.venv/bin/python -m pip install -e . --no-deps

echo "core installed:"
./.venv/bin/python -c "import numpy,scipy,soundfile,sklearn,pandas,torch,adfd; print('  ok numpy',numpy.__version__,'torch',torch.__version__)"
