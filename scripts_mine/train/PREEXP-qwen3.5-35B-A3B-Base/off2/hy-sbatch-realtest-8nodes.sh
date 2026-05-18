#!/bin/bash
# Qwen3.5-35B-A3B off2 disagg real-task smoke test on 8 nodes.
# Defaults are trimmed to finish one rollout/train/save_hf cycle under lowprio preemption.
#SBATCH --job-name=miles-qwen35-off2-smoke
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --account=k2m
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --exclusive
#SBATCH --output=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.log
#SBATCH --error=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J.%t.err
#SBATCH --open-mode=append

set -euo pipefail

BASE_SCRIPT="/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/scripts_mine/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/hy-sbatch-8nodes.sh"

export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B-Base}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-/mnt/weka/shrd/k2m/haolong.jia/checkpoint/${MODEL_NAME}}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/weka/shrd/k2m/haolong.jia/checkpoint_torch_dist/${MODEL_NAME}}"
export MODEL_SCRIPT_PATH_IN_CONTAINER="${MODEL_SCRIPT_PATH_IN_CONTAINER:-/root/miles/scripts/models/qwen3.5-35B-A3B.sh}"
export PREDICTIVE_REPLAY_ARGS_SCRIPT_IN_CONTAINER="${PREDICTIVE_REPLAY_ARGS_SCRIPT_IN_CONTAINER:-/root/miles/scripts_mine/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/_predictive_replay_args.sh}"
export CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-autofix}"
export APPLY_CHAT_TEMPLATE_KWARGS="${APPLY_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\": false}}"
export TRANSFORMERS_HOST_ROOT="${TRANSFORMERS_HOST_ROOT:-/mnt/weka/home/hongyi.wang/workspace/transformers-main}"
export SGLANG_HOST_ROOT="${SGLANG_HOST_ROOT:-/mnt/weka/home/hongyi.wang/workspace/xllm-sglang}"
export TRANSFORMERS_SRC_IN_CONTAINER="${TRANSFORMERS_SRC_IN_CONTAINER:-/root/transformers-main/src}"
export SGLANG_SRC_IN_CONTAINER="${SGLANG_SRC_IN_CONTAINER:-/root/xllm-sglang/python}"
export RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-${SGLANG_SRC_IN_CONTAINER}:/root/miles:/root/Megatron-LM}"
export RUNTIME_PIP_INSTALL_COMMAND="${RUNTIME_PIP_INSTALL_COMMAND:-pip install --no-cache-dir --no-deps --force-reinstall transformers==4.57.1 huggingface-hub==0.36.0 tokenizers==0.22.2 jinja2==3.1.6 timm==1.0.23 sgl-kernel==0.3.21}"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/mnt/weka/shrd/k2m/haolong.jia:/mnt/weka/shrd/k2m/haolong.jia:rw,/mnt/weka/shrd/k2m/hongyi.wang:/mnt/weka/shrd/k2m/hongyi.wang:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/miles:/root/miles:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/verl:/root/verl:ro,${TRANSFORMERS_HOST_ROOT}:/root/transformers-main:ro,${SGLANG_HOST_ROOT}:/root/xllm-sglang:ro}"

export RUN_POSTFIX="${RUN_POSTFIX:-off2-realtest-smoke-v7}"
export RESOURCE_LAYOUT="${RESOURCE_LAYOUT:-disagg}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-1}"
export ENABLE_KEEP_OLD_ACTOR="${ENABLE_KEEP_OLD_ACTOR:-1}"
export UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"

export USE_MILES_ROUTER="${USE_MILES_ROUTER:-1}"
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-0}"
export USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-0}"
export ENABLE_PREDICTIVE_ROUTING_REPLAY="${ENABLE_PREDICTIVE_ROUTING_REPLAY:-0}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-1024}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
export LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-4096}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
export ENABLE_AUTO_CKPT_EVAL="${ENABLE_AUTO_CKPT_EVAL:-0}"
export USE_WANDB="${USE_WANDB:-0}"
export ENABLE_NO_SAVE_OPTIM="${ENABLE_NO_SAVE_OPTIM:-1}"
export ROUTER_LOGITS_PATH="${ROUTER_LOGITS_PATH:-disabled}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-64}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-64}"
export SGLANG_CUDA_GRAPH_MAX="${SGLANG_CUDA_GRAPH_MAX:-64}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"

if [ ! -d "${HF_MODEL_PATH}" ]; then
  echo "[ERROR] HF checkpoint not found: ${HF_MODEL_PATH}" >&2
  exit 1
fi

if [ ! -d "${SGLANG_HOST_ROOT}" ]; then
  echo "[ERROR] sglang source not found: ${SGLANG_HOST_ROOT}" >&2
  exit 1
fi

if [ ! -d "${REF_MODEL_PATH}" ]; then
  echo "[ERROR] torch_dist checkpoint not found: ${REF_MODEL_PATH}" >&2
  echo "[ERROR] run sbatch-convert-torch-dist.sh first" >&2
  exit 1
fi

if [ ! -f "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt" ] || [ "$(cat "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt")" != "release" ]; then
  echo "[ERROR] invalid torch_dist checkpoint at ${REF_MODEL_PATH}" >&2
  echo "[ERROR] expected latest_checkpointed_iteration.txt to contain release" >&2
  exit 1
fi

exec bash "${BASE_SCRIPT}"
