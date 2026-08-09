#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/adfd-robustness"
module purge
module load Python/PyTorch_GPU_v2.4
source .venv/bin/activate
mkdir -p third_party

warn() { echo "WARN: $*"; }

echo "=== deps: HuggingFace transformers (XLS-R фронтенд вместо fairseq) ==="
pip install -q "transformers>=4.30,<5" safetensors huggingface_hub gdown || warn "pip deps failed"

echo "=== HF-кэш: config XLS-R 300M (для offline compute) ==="
python - <<'PY' || warn "HF snapshot failed"
from huggingface_hub import snapshot_download
p = snapshot_download("facebook/wav2vec2-xls-r-300m", allow_patterns=["config.json", "preprocessor_config.json"])
print("cached:", p)
PY

echo "=== [1] RawNet2 — интегрирован ранее ==="
[ -f third_party/rawnet2/pre_trained_DF_RawNet2.pth ] && echo "  checkpoint present" || warn "rawnet2 checkpoint отсутствует"

echo "=== [2] Nes2Net v1 (CtrSVDD, s3prl WavLM — args в models.yaml) ==="
[ -f third_party/Nes2Net/WavLM_Nes2Net_X.pth ] && echo "  checkpoint present" || warn "nes2net checkpoint отсутствует"

echo "=== [3] SSL-AASIST (HF-фронтенд, БЕЗ fairseq) ==="
[ -d third_party/SSL_Anti-spoofing ] || git clone --depth 1 https://github.com/TakHemlata/SSL_Anti-spoofing third_party/SSL_Anti-spoofing || warn "clone SSL_Anti-spoofing failed"
if [ ! -f third_party/SSL_Anti-spoofing/Best_SSL_model_LA.pth ]; then
    echo "  качаю Best_SSL_model_LA.pth с Google Drive (папка репо)"
    gdown --folder --remaining-ok -O third_party/SSL_Anti-spoofing/_gdl "${SSL_AASIST_GDRIVE_FOLDER:-https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB}" 2>/dev/null || warn "gdown folder failed"
    src=$(find third_party/SSL_Anti-spoofing/_gdl -iname 'Best_SSL_model_LA.pth' 2>/dev/null | head -1)
    if [ -n "${src:-}" ]; then cp "$src" third_party/SSL_Anti-spoofing/Best_SSL_model_LA.pth; echo "  ok"; else echo "  MANUAL: положи Best_SSL_model_LA.pth в third_party/SSL_Anti-spoofing/ (или задай SSL_AASIST_GDRIVE_FOLDER)"; fi
fi

echo "=== [4] XLSR-Mamba (HF-фронтенд + mamba-ssm) ==="
[ -d third_party/XLSR-Mamba ] || git clone --depth 1 https://github.com/swagshaw/XLSR-Mamba third_party/XLSR-Mamba || warn "clone XLSR-Mamba failed"
[ -f third_party/XLSR-Mamba/xlsr_mamba_LA.pth ] && echo "  checkpoint present" || warn "xlsr_mamba checkpoint отсутствует"
if python -c "import mamba_ssm" 2>/dev/null; then
    echo "  mamba-ssm установлен"
else
    echo "  mamba-ssm НЕ установлен. cu124 → готовых wheels нет, нужна сборка на GPU-узле:"
    echo "    1) login: pip download --no-binary mamba-ssm,causal-conv1d -d third_party/mamba_src mamba-ssm==1.1.4 causal-conv1d==1.1.3.post1 ninja packaging einops"
    echo "    2) sbatch scripts/build_mamba.sh"
fi

echo
echo "done (login part). Проверка моделей на compute-узле:  sbatch scripts/slurm_verify_heavy.sh"
