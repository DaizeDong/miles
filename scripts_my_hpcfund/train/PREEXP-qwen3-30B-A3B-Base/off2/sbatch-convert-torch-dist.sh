#!/bin/bash
# One-shot: convert Qwen3-30B-A3B-Base HF safetensors to Megatron torch_dist.
# Required before any miles training run that uses REF_MODEL_PATH.
# Single-node, 8 GPU. Output ~60 GB at REF_MODEL_PATH.
#SBATCH --job-name=qwen3-30b-td-convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --account=hwang
#SBATCH --partition=mi3008x
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.log
#SBATCH --error=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.err
#SBATCH --open-mode=append

set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B-Base}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-/work1/hwang/dzdong/checkpoints_hf/${MODEL_NAME}}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-/work1/hwang/dzdong/checkpoints_torch_dist/${MODEL_NAME}}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/work1/hwang/dzdong/images/miles-mi300.sif}"

mkdir -p "$(dirname "${REF_MODEL_PATH}")" "$(dirname "$0")/../../../../miles_logs_tmp/sbatch" || true
mkdir -p /work1/hwang/dzdong/miles_logs_tmp/sbatch

if [ ! -d "${HF_MODEL_PATH}" ]; then
  echo "[ERROR] HF checkpoint not found: ${HF_MODEL_PATH}" >&2
  exit 1
fi

if [ ! -f "${CONTAINER_IMAGE}" ]; then
  echo "[ERROR] SIF not found: ${CONTAINER_IMAGE}" >&2
  exit 1
fi

if [ -f "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt" ] && \
   [ "$(cat "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt")" = "release" ]; then
  echo "[INFO] torch_dist checkpoint already exists at ${REF_MODEL_PATH}, skipping."
  exit 0
fi

echo "[INFO] HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "[INFO] REF_MODEL_PATH=${REF_MODEL_PATH}"
echo "[INFO] CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "[INFO] host node=$(hostname), gpus visible:"
rocm-smi --showproductname 2>/dev/null | grep -E "Card Series|GFX Version" | head -16 || true

srun \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --export=ALL \
  apptainer exec --rocm --writable-tmpfs \
    --bind /work1/hwang/dzdong:/work1/hwang/dzdong \
    --bind /home1/dzdong/workspace/miles:/root/miles \
    "${CONTAINER_IMAGE}" \
  bash -c '
set -euo pipefail
set -x

rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED || true
cd /root/miles
pip install -e . --no-deps || true

# Make miles + miles_plugins importable from source. The host-side `pip
# install -e .` is unreliable across Py 3.10 (container) vs Py 3.13 (host)
# and produces a cp313-tagged wheel that fails to register packages inside
# the container venv. Source-on-PYTHONPATH is the robust fallback.
export PYTHONPATH="/root/miles:/app/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"

python3 - <<PY
import miles, miles_plugins, miles_plugins.mbridge
print("[INFO] miles imports OK:", miles.__file__)
print("[INFO] miles_plugins:", miles_plugins.__file__)
PY
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEPRECATED_MEGATRON_COMPATIBLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456

source "${MODEL_CONFIG_SCRIPT:-/root/miles/scripts/models/qwen3-30B-A3B.sh}"

torchrun --nproc-per-node 8 /root/miles/tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "'"${HF_MODEL_PATH}"'" \
  --save "'"${REF_MODEL_PATH}"'"
'

echo "[INFO] torch_dist ckpt at ${REF_MODEL_PATH}"
ls -lh "${REF_MODEL_PATH}" | head -20 || true
