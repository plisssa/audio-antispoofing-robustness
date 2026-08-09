#!/usr/bin/env bash
#SBATCH --job-name=adfd-aasist3
#SBATCH --account=proj_1877
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
module load Python/PyTorch_GPU_v2.4
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/software/python/envs/tensorflow-gpu2_9/lib"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

MANIFEST="${1:-data/in_the_wild/manifest_small.csv}"
RUNDIR="${2:-runs/aasist3_itw}"
shift 2 2>/dev/null || true

python -u -m adfd.cli sweep --detector aasist3 --manifest "$MANIFEST" --run-dir "$RUNDIR" "$@"
