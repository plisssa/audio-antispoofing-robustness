#!/usr/bin/env bash
#SBATCH --job-name=adfd-manifests
#SBATCH --account=proj_1912
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
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

N="${1:-4000}"

adfd subset --manifest data/asvspoof2019_la/eval.csv --out data/asvspoof2019_la/eval_large.csv --n "$N"
adfd subset --manifest data/in_the_wild/manifest.csv --out data/in_the_wild/manifest_large.csv --n "$N"
adfd subset --manifest data/asvspoof2021_la/eval.csv --out data/asvspoof2021_la/eval_large.csv --n "$N"

adfd merge --train-from data/asvspoof2019_la/train_small.csv --eval-from data/asvspoof2019_la/eval_large.csv --out data/reference/native_large.csv
adfd merge --train-from data/asvspoof2019_la/train_small.csv --eval-from data/in_the_wild/manifest_large.csv --out data/reference/crossdomain_large.csv
adfd merge --train-from data/asvspoof2019_la/train_small.csv --eval-from data/asvspoof2021_la/eval_large.csv --out data/reference/crossdomain21_large.csv

for f in data/asvspoof2019_la/eval_large.csv data/in_the_wild/manifest_large.csv data/asvspoof2021_la/eval_large.csv \
         data/reference/native_large.csv data/reference/crossdomain_large.csv data/reference/crossdomain21_large.csv; do
    printf '%-48s %s\n' "$f" "$([ -f "$f" ] && wc -l < "$f" || echo MISSING)"
done
