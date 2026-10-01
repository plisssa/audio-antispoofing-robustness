#!/usr/bin/env bash
#SBATCH --job-name=adfd-attack
#SBATCH --account=proj_1912
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
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

DETECTOR="${1:?usage: slurm_attack.sh DETECTOR [MANIFEST] [TRANSFER...]}"
MANIFEST="${2:-data/asvspoof2019_la/eval_small.csv}"
shift 2 2>/dev/null || true
RUNDIR="runs/${DETECTOR}_$(basename "${MANIFEST%.*}")_attack"

python -u -m adfd.cli attack --detector "$DETECTOR" --manifest "$MANIFEST" --budget snr --run-dir "$RUNDIR" ${*:+--transfer "$@"}
