#!/usr/bin/env bash
#SBATCH --job-name=adfd-smoke
#SBATCH --account=proj_1912
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --output=runs/slurm-%x-%j.log
#SBATCH --error=runs/slurm-%x-%j.err
set -uo pipefail

cd "$HOME/adfd-robustness"
module purge
PYTORCH_ENV="${PYTORCH_ENV:-/opt/software/python/envs/pytorch2_4}"
[ -x "$PYTORCH_ENV/bin/python" ] || { echo "base python env missing: $PYTORCH_ENV" >&2; exit 1; }
export PATH="$PYTORCH_ENV/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/software/python/envs/tensorflow-gpu2_9/lib"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

python -u - <<'PY'
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("no gpu visible")
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

x = torch.randn(256, 256, device="cuda")
print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))

from mamba_ssm import Mamba
block = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
y = block(torch.randn(2, 32, 64, device="cuda"))
print("mamba cuda kernel ok:", tuple(y.shape))

from causal_conv1d import causal_conv1d_fn
w = torch.randn(64, 4, device="cuda")
z = causal_conv1d_fn(torch.randn(2, 64, 32, device="cuda"), w)
print("causal_conv1d ok:", tuple(z.shape))

import transformers
print("transformers", transformers.__version__)
PY
