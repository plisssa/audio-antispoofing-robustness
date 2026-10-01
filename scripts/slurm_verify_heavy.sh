#!/usr/bin/env bash
#SBATCH --job-name=adfd-verify-heavy
#SBATCH --account=proj_1912
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
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

MANIFEST="${1:-data/asvspoof2019_la/eval_small.csv}"

for det in rawnet2 ssl_aasist xlsr_mamba nes2net; do
    echo "===================== verify $det ====================="
    python -u -m adfd.cli verify-model --detector "$det" --manifest "$MANIFEST" || echo "NOT READY: $det"
done
echo "done. EER около 0.0-0.5 = модель загрузилась и работает; EER>0.5 = проверь spoof_index; ошибка загрузки = чекпойнт/зависимость не на месте."
