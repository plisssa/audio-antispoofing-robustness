#!/usr/bin/env bash
#SBATCH --job-name=adfd-diag-nes2net
#SBATCH --account=proj_1912
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
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

python -u - <<'PY'
import torch, yaml
from adfd.models.base import build_detector

cfg = yaml.safe_load(open("configs/models.yaml"))["nes2net"]
det = build_detector("nes2net", sample_rate=16000, **cfg)
det.load()
top = det.model
ssl = getattr(top, "ssl_model", None)
hub = getattr(ssl, "model", None)

def probe(name, fn):
    x = torch.randn(1, det.crop, device=det.device, requires_grad=True)
    try:
        out = fn(x)
        if isinstance(out, dict):
            hs = out.get("hidden_states")
            out = hs[-1] if hs is not None else next(iter(out.values()))
        if isinstance(out, (list, tuple)):
            out = out[-1]
        g = torch.autograd.grad(out.float().sum(), x, allow_unused=True, retain_graph=False)[0]
        gs = "None (граф разорван)" if g is None else f"{g.abs().sum().item():.4e}"
        print(f"{name:38} out={tuple(out.shape)} grad_fn={type(out.grad_fn).__name__ if out.grad_fn else None} grad_по_входу={gs}")
    except Exception as error:
        print(f"{name:38} ОШИБКА: {type(error).__name__}: {error}")

print("=== где рвётся граф ===")
if hub is not None:
    probe("1. s3prl WavLM (ssl_model.model)", lambda x: hub(x))
if ssl is not None:
    probe("2. SSLModel (агрегация слоёв)", lambda x: ssl(x))
probe("3. полная модель Nes2Net", lambda x: top(x))
probe("4. logits_tensor адаптера", lambda x: det.logits_tensor(x))

print()
print("=== training-флаги (влияет на hooks в s3prl) ===")
for tag, module in (("top", top), ("ssl_model", ssl), ("s3prl hub", hub)):
    if module is not None:
        print(f"  {tag:12} training={module.training}")
PY
