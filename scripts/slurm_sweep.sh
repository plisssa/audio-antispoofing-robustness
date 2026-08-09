#!/usr/bin/env bash
#SBATCH --job-name=adfd-sweep
#SBATCH --account=proj_1877
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

module purge
module load Python/PyTorch_GPU_v2.4
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/software/python/envs/tensorflow-gpu2_9/lib"
source .venv/bin/activate
export HF_HUB_OFFLINE=1

DETECTOR="${1:-aasist3}"
MANIFEST="${2:-data/in_the_wild/manifest.csv}"
RUNDIR="runs/${DETECTOR}_$(basename "${MANIFEST%.*}")_sweep"

python -u -m adfd.cli sweep --detector "$DETECTOR" --manifest "$MANIFEST" --run-dir "$RUNDIR"
