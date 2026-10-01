#!/usr/bin/env bash
#SBATCH --job-name=adfd-sweep-cpu
#SBATCH --account=proj_1912
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/slurm-%x-%j.log
#SBATCH --error=runs/slurm-%x-%j.err
set -euo pipefail

module purge
PYTORCH_ENV="${PYTORCH_ENV:-/opt/software/python/envs/pytorch2_4}"
[ -x "$PYTORCH_ENV/bin/python" ] || { echo "base python env missing: $PYTORCH_ENV" >&2; exit 1; }
export PATH="$PYTORCH_ENV/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/opt/software/python/envs/tensorflow-gpu2_9/lib"
source .venv/bin/activate

DETECTOR="${1:-reference}"
MANIFEST="${2:-data/in_the_wild/manifest.csv}"
RUNDIR="runs/${DETECTOR}_$(basename "${MANIFEST%.*}")_sweep"

python -u -m adfd.cli sweep --detector "$DETECTOR" --manifest "$MANIFEST" --run-dir "$RUNDIR"
