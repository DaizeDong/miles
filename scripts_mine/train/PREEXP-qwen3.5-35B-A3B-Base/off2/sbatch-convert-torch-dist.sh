#!/bin/bash
# Convert Qwen3.5 HF checkpoint to Megatron torch_dist format for Miles training.
#SBATCH --job-name=qwen35-td-convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --account=k2m
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --exclusive
#SBATCH --output=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.log
#SBATCH --error=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J.%t.err
#SBATCH --open-mode=append

set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B-Base}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-/mnt/weka/shrd/k2m/haolong.jia/checkpoint/${MODEL_NAME}}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/weka/shrd/k2m/haolong.jia/checkpoint_torch_dist/${MODEL_NAME}}"
export TRANSFORMERS_HOST_ROOT="${TRANSFORMERS_HOST_ROOT:-/mnt/weka/home/hongyi.wang/workspace/transformers-main}"
export SGLANG_HOST_ROOT="${SGLANG_HOST_ROOT:-/mnt/weka/home/hongyi.wang/workspace/xllm-sglang}"
export TRANSFORMERS_SRC_IN_CONTAINER="${TRANSFORMERS_SRC_IN_CONTAINER:-/root/transformers-main/src}"
export SGLANG_SRC_IN_CONTAINER="${SGLANG_SRC_IN_CONTAINER:-/root/xllm-sglang/python}"
export RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-${SGLANG_SRC_IN_CONTAINER}:/root/miles:/root/Megatron-LM}"
export RUNTIME_PIP_INSTALL_COMMAND="${RUNTIME_PIP_INSTALL_COMMAND:-pip install --no-cache-dir --no-deps --force-reinstall transformers==4.57.1 huggingface-hub==0.36.0 tokenizers==0.22.2 jinja2==3.1.6 timm==1.0.23}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/mnt/weka/shrd/k2m/hongyi.wang/containers/slimerl+slime+latest.sqsh}"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/mnt/weka/shrd/k2m/haolong.jia:/mnt/weka/shrd/k2m/haolong.jia:rw,/mnt/weka/shrd/k2m/hongyi.wang:/mnt/weka/shrd/k2m/hongyi.wang:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/miles:/root/miles:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/verl:/root/verl:ro,${TRANSFORMERS_HOST_ROOT}:/root/transformers-main:ro,${SGLANG_HOST_ROOT}:/root/xllm-sglang:ro}"

mkdir -p "$(dirname "${REF_MODEL_PATH}")"

if [ ! -d "${HF_MODEL_PATH}" ]; then
  echo "[ERROR] HF checkpoint not found: ${HF_MODEL_PATH}" >&2
  exit 1
fi

if [ ! -d "${SGLANG_HOST_ROOT}" ]; then
  echo "[ERROR] sglang source not found: ${SGLANG_HOST_ROOT}" >&2
  exit 1
fi

if [ -f "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt" ] && [ "$(cat "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt")" = "release" ]; then
  echo "[INFO] torch_dist checkpoint already exists at ${REF_MODEL_PATH}, skipping."
  exit 0
fi

srun \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --export=ALL \
  bash -lc '
set -euo pipefail
set -x

rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED || true

cd /root/miles
bash -lc "'"${RUNTIME_PIP_INSTALL_COMMAND}"'"
pip install -e . --no-deps

export PYTHONPATH="'"${RUNTIME_PYTHONPATH}"'${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEPRECATED_MEGATRON_COMPATIBLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456

source /root/miles/scripts/models/qwen3.5-35B-A3B.sh

torchrun --nproc-per-node 8 /root/miles/tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "'"${HF_MODEL_PATH}"'" \
  --save "'"${REF_MODEL_PATH}"'"
'
