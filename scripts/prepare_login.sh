#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/adfd-robustness"
module purge
PYTORCH_ENV="${PYTORCH_ENV:-/opt/software/python/envs/pytorch2_4}"
[ -x "$PYTORCH_ENV/bin/python" ] || { echo "base python env missing: $PYTORCH_ENV" >&2; exit 1; }
export PATH="$PYTORCH_ENV/bin:$PATH"
source .venv/bin/activate

warn() { echo "WARN: $*"; }

echo "=== 0. prerequisites ==="
command -v ffmpeg >/dev/null || warn "ffmpeg missing — codec/telephony stages will fail"

echo "=== 1. datasets (skipped if already present) ==="
bash scripts/download_in_the_wild.sh data/raw/in_the_wild || warn "In-the-Wild download failed"
bash scripts/download_asvspoof2019_la.sh data/raw/asvspoof2019_la || warn "ASVspoof2019 download failed"

echo "=== 2. noise / rir corpora ==="
bash scripts/download_corpora.sh data/raw/corpora data/corpora || warn "corpora download failed (real_* stages will skip)"
for corpus in data/corpora/musan_noise data/corpora/esc50_birds data/corpora/rirs; do
    count=$(find -L "$corpus" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "  $corpus: $count files"
done

echo "=== 3. model weights (cached for offline compute nodes) ==="
bash scripts/precache_aasist3.sh || warn "aasist3 precache failed"
python - <<'PY' || warn "encodec precache failed — sweep_encodec stays optional"
from encodec import EncodecModel
EncodecModel.encodec_model_24khz()
print("encodec 24k weights cached")
PY

echo "=== 4. manifests (skipped if already present) ==="
[ -f data/in_the_wild/manifest.csv ] || adfd manifest --dataset in_the_wild --root data/raw/in_the_wild/release_in_the_wild --out data/in_the_wild/manifest.csv --train-ratio 0.2 || warn "ITW manifest not built"
[ -f data/asvspoof2019_la/train.csv ] || adfd manifest --dataset asvspoof2019_la --root data/raw/asvspoof2019_la/LA --split train --out data/asvspoof2019_la/train.csv || warn "ASV2019 train manifest not built"
[ -f data/asvspoof2019_la/eval.csv ] || adfd manifest --dataset asvspoof2019_la --root data/raw/asvspoof2019_la/LA --split eval --out data/asvspoof2019_la/eval.csv || warn "ASV2019 eval manifest not built"

echo "=== 5. balanced subsets ==="
adfd subset --manifest data/in_the_wild/manifest.csv --out data/in_the_wild/manifest_small.csv --n 800 || warn "ITW subset failed"
adfd subset --manifest data/asvspoof2019_la/eval.csv --out data/asvspoof2019_la/eval_small.csv --n 800 || warn "ASV2019 eval subset failed"
adfd subset --manifest data/asvspoof2019_la/train.csv --out data/asvspoof2019_la/train_small.csv --n 2000 --split train || warn "ASV2019 train subset failed"

echo "=== 6. cross-domain reference manifests ==="
adfd merge --train-from data/asvspoof2019_la/train_small.csv --eval-from data/asvspoof2019_la/eval_small.csv --out data/reference/native.csv || warn "reference native merge failed"
adfd merge --train-from data/asvspoof2019_la/train_small.csv --eval-from data/in_the_wild/manifest_small.csv --out data/reference/crossdomain.csv || warn "reference crossdomain merge failed"

echo "=== 6b. ASVspoof2021 LA (optional, last — never blocks the rest) ==="
if [ "${DOWNLOAD_ASV2021:-0}" = "1" ]; then
    bash scripts/download_asvspoof2021.sh LA data/raw/asvspoof2021_la || warn "ASVspoof2021 LA download failed (optional)"
    if [ -d data/raw/asvspoof2021_la/ASVspoof2021_LA_eval ]; then
        [ -f data/asvspoof2021_la/eval.csv ] || adfd manifest --dataset asvspoof2021_la --root data/raw/asvspoof2021_la --out data/asvspoof2021_la/eval.csv || warn "ASV2021 manifest not built"
        [ -f data/asvspoof2021_la/eval.csv ] && { adfd subset --manifest data/asvspoof2021_la/eval.csv --out data/asvspoof2021_la/eval_small.csv --n 800 || warn "ASV2021 subset failed"; }
    fi
else
    echo "  skipped (set DOWNLOAD_ASV2021=1 to fetch — it is a bonus 3rd domain)"
fi

echo "=== 7. summary ==="
for corpus in data/corpora/musan_noise data/corpora/esc50_birds data/corpora/rirs; do
    echo "  $corpus: $(find -L "$corpus" -type f 2>/dev/null | wc -l | tr -d ' ') files"
done
echo "  (model loading is verified inside the pipeline job, not here — login node kills heavy processes)"

echo "done. launch: sbatch scripts/slurm_pipeline.sh   |   watch: ./.venv/bin/adfd progress"
