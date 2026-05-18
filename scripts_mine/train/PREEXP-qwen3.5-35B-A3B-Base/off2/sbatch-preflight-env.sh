#!/bin/bash
# Preflight import check for Qwen3.5 runtime dependencies inside the training container.
#SBATCH --job-name=qwen35-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=k2m
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --output=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.log
#SBATCH --error=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.err
#SBATCH --open-mode=append

set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B-Base}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-/mnt/weka/shrd/k2m/haolong.jia/checkpoint/${MODEL_NAME}}"
export TRANSFORMERS_HOST_ROOT="${TRANSFORMERS_HOST_ROOT:-/mnt/weka/home/hongyi.wang/workspace/transformers-main}"
export SGLANG_HOST_ROOT="${SGLANG_HOST_ROOT:-/mnt/weka/home/hongyi.wang/workspace/xllm-sglang}"
export TRANSFORMERS_SRC_IN_CONTAINER="${TRANSFORMERS_SRC_IN_CONTAINER:-/root/transformers-main/src}"
export SGLANG_SRC_IN_CONTAINER="${SGLANG_SRC_IN_CONTAINER:-/root/xllm-sglang/python}"
export RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-${SGLANG_SRC_IN_CONTAINER}:/root/miles:/root/Megatron-LM}"
export RUNTIME_PIP_INSTALL_COMMAND="${RUNTIME_PIP_INSTALL_COMMAND:-pip install --no-cache-dir --no-deps --force-reinstall transformers==4.57.1 huggingface-hub==0.36.0 tokenizers==0.22.2 jinja2==3.1.6 timm==1.0.23 sgl-kernel==0.3.21}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/mnt/weka/shrd/k2m/hongyi.wang/containers/slimerl+slime+latest.sqsh}"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/mnt/weka/shrd/k2m/haolong.jia:/mnt/weka/shrd/k2m/haolong.jia:rw,/mnt/weka/shrd/k2m/hongyi.wang:/mnt/weka/shrd/k2m/hongyi.wang:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/miles:/root/miles:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/verl:/root/verl:ro,${TRANSFORMERS_HOST_ROOT}:/root/transformers-main:ro,${SGLANG_HOST_ROOT}:/root/xllm-sglang:ro}"

if [ ! -d "${HF_MODEL_PATH}" ]; then
  echo "[ERROR] HF checkpoint not found: ${HF_MODEL_PATH}" >&2
  exit 1
fi

if [ ! -d "${SGLANG_HOST_ROOT}" ]; then
  echo "[ERROR] sglang source not found: ${SGLANG_HOST_ROOT}" >&2
  exit 1
fi

srun \
  --ntasks=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --export=ALL \
  bash -lc '
set -euo pipefail

rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED || true
cd /root/miles
bash -lc "'"${RUNTIME_PIP_INSTALL_COMMAND}"'"
pip install -e . --no-deps

export PYTHONPATH="'"${RUNTIME_PYTHONPATH}"'${PYTHONPATH:+:${PYTHONPATH}}"

python3 - <<PY
import importlib
import importlib.metadata
import sglang
import transformers
from packaging.version import Version
from miles.utils.hf_config_utils import get_hf_text_config_dict, load_hf_config

hf_model_path = "'"${HF_MODEL_PATH}"'"
modules = [
    "fla",
    "sglang",
    "sglang.srt.utils.hf_transformers_utils",
    "sglang.srt.models.qwen3_5",
    "miles_plugins.models.qwen3_5",
    "miles_plugins.mbridge.qwen3_5_moe",
]
for name in modules:
    importlib.import_module(name)
    print(f"[OK] import {name}")

print("[INFO] transformers =", transformers.__version__)
print("[INFO] sglang =", getattr(sglang, "__version__", "unknown"))
sgl_kernel_version = importlib.metadata.version("sgl-kernel")
print("[INFO] sgl-kernel =", sgl_kernel_version)
cfg = load_hf_config(hf_model_path, trust_remote_code=True)
print("[INFO] hf model_type =", cfg.model_type)
layer_types = list(get_hf_text_config_dict(hf_model_path)["layer_types"])
print("[INFO] num_layers =", len(layer_types))
print("[INFO] unique layer_types =", sorted(set(layer_types)))
assert cfg.model_type == "qwen3_5_moe", cfg.model_type
assert Version(sgl_kernel_version) >= Version("0.3.21"), sgl_kernel_version
assert len(layer_types) == 40, len(layer_types)
assert "linear_attention" in layer_types
assert "full_attention" in layer_types
print("[OK] Qwen3.5 runtime preflight passed")
PY
'
