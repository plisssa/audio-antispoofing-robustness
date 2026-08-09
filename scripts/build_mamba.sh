#!/usr/bin/env bash
#SBATCH --job-name=adfd-build-mamba
#SBATCH --account=proj_1877
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=runs/slurm-%x-%j.log
#SBATCH --error=runs/slurm-%x-%j.err
set -uo pipefail

cd "$HOME/adfd-robustness"
module purge
module load Python/PyTorch_GPU_v2.4
source .venv/bin/activate

for candidate in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12.4; do
    if [ -n "$candidate" ] && [ -x "$candidate/bin/nvcc" ]; then
        export CUDA_HOME="$candidate"
        export PATH="$CUDA_HOME/bin:$PATH"
        break
    fi
done
echo "CUDA_HOME=${CUDA_HOME:-unset}"
command -v nvcc >/dev/null && nvcc --version | tail -2 || echo "nvcc НЕ НАЙДЕН — сборка CUDA-ядер невозможна"
echo "gcc: $(gcc -dumpversion 2>/dev/null)"

export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_SKIP_CUDA_BUILD=FALSE
export CAUSAL_CONV1D_SKIP_CUDA_BUILD=FALSE
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
OPTS="--no-deps --no-build-isolation"

build_from_git() {
    local dir="$1" name="$2"
    if [ ! -d "$dir/csrc" ]; then
        echo "SKIP $name: нет $dir/csrc (нужен git clone на login-узле)"
        return 1
    fi
    echo "=== сборка $name из $dir ==="
    pip install $OPTS "$dir" && echo "$name: OK" || echo "$name: BUILD FAILED"
}

build_from_git third_party/causal-conv1d causal-conv1d
build_from_git third_party/mamba mamba-ssm

echo "=== проверка ==="
python -c "import causal_conv1d; print('causal_conv1d OK')" 2>/dev/null || echo "causal_conv1d НЕ импортируется"
python -c "import mamba_ssm; print('mamba_ssm OK', getattr(mamba_ssm,'__version__','?'))" 2>/dev/null || echo "mamba_ssm НЕ импортируется"
python -c "import selective_scan_cuda; print('selective_scan_cuda OK')" 2>/dev/null || echo "selective_scan_cuda НЕ импортируется"
echo "done. если OK — sbatch scripts/slurm_verify_heavy.sh"
