#!/usr/bin/env bash
#SBATCH --job-name=adfd-gpu
#SBATCH --account=proj_1912
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/slurm-%x-%j.log
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

cd "$HOME/adfd-robustness"
module purge
PYTORCH_ENV="${PYTORCH_ENV:-/opt/software/python/envs/pytorch2_4}"
[ -x "$PYTORCH_ENV/bin/python" ] || { echo "base python env missing: $PYTORCH_ENV" >&2; exit 1; }
export PATH="$PYTORCH_ENV/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/software/python/envs/tensorflow-gpu2_9/lib"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

DETECTOR="${1:?usage: slurm_gpu.sh DETECTOR [MANIFEST] [RUNDIR] [extra adfd args]}"
MANIFEST="${2:-data/in_the_wild/manifest_small.csv}"
RUNDIR="${3:-runs/${DETECTOR}_run}"
shift 3 2>/dev/null || true

python -u -m adfd.cli sweep --detector "$DETECTOR" --manifest "$MANIFEST" --run-dir "$RUNDIR" "$@"
