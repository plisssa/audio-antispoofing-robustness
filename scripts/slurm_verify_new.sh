#!/usr/bin/env bash
#SBATCH --job-name=adfd-verify-new
#SBATCH --account=proj_1912
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
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

MANIFEST="${1:-data/reference/native.csv}"

for det in xlsr_linear wavlm_linear nes2net_v2; do
    echo "===================== verify $det ====================="
    python -u -m adfd.cli verify-model --detector "$det" --manifest "$MANIFEST" || echo "NOT READY: $det"
done

echo "===================== градиенты доходят до входа? ====================="
python -u - <<'PY'
import torch, yaml
from adfd.models.base import build_detector

models = yaml.safe_load(open("configs/models.yaml"))
for name in ("xlsr_linear", "wavlm_linear", "nes2net_v2"):
    try:
        detector = build_detector(name, sample_rate=16000, **(models.get(name) or {}))
        detector.load()
        if getattr(detector, "trainable", False):
            detector.head = detector.head
        x = torch.randn(1, detector.crop, device=detector.device, requires_grad=True)
        logits = detector.logits_tensor(x)
        grad = torch.autograd.grad(logits.sum(), x, allow_unused=True)[0]
        state = "None (граф разорван)" if grad is None else f"{grad.abs().sum().item():.3e}"
        print(f"  {name:14} logits={tuple(logits.shape)} grad={state}")
    except Exception as error:
        print(f"  {name:14} ОШИБКА {type(error).__name__}: {str(error)[:90]}")
PY
