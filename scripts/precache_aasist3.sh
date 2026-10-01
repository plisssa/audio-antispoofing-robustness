#!/usr/bin/env bash
set -euo pipefail

module purge
PYTORCH_ENV="${PYTORCH_ENV:-/opt/software/python/envs/pytorch2_4}"
[ -x "$PYTORCH_ENV/bin/python" ] || { echo "base python env missing: $PYTORCH_ENV" >&2; exit 1; }
export PATH="$PYTORCH_ENV/bin:$PATH"
source .venv/bin/activate

mkdir -p third_party
[ -e third_party/AASIST3 ] || ln -s ~/AASIST3 third_party/AASIST3

export PYTHONPATH="third_party/AASIST3:${PYTHONPATH:-}"
python - <<'PY'
from model import aasist3
m = aasist3.from_pretrained("MTUCI/AASIST3", w2v_cache_dir="third_party/AASIST3/weights")
print("cached aasist3 params:", sum(p.numel() for p in m.parameters()))
PY
echo "done; weights are cached, jobs can run offline (HF_HUB_OFFLINE=1)"
