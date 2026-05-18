#!/bin/bash
# Dedicated 8-node baseline launcher for Qwen3.5-35B-A3B.
# Runtime/training knobs intentionally mirror the Qwen3 off2 baseline script.
# Only model-specific paths/runtime requirements are overridden here.
#SBATCH --job-name=miles-qwen35-off2-baseline
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --account=k2m
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
##SBATCH --reservation=moe
#SBATCH --exclusive
#SBATCH --output=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.log
#SBATCH --error=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.err
#SBATCH --open-mode=append

set -euo pipefail

BASE_SCRIPT="/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/scripts_mine/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/hy-sbatch-8nodes.sh"

export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B-Base}"
export HF_MODEL_PATH="${HF_MODEL_PATH:-/mnt/weka/shrd/k2m/haolong.jia/checkpoint/${MODEL_NAME}}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/weka/shrd/k2m/haolong.jia/checkpoint_torch_dist/${MODEL_NAME}}"
export MODEL_SCRIPT_PATH_IN_CONTAINER="${MODEL_SCRIPT_PATH_IN_CONTAINER:-/root/miles/scripts/models/qwen3.5-35B-A3B.sh}"
export PREDICTIVE_REPLAY_ARGS_SCRIPT_IN_CONTAINER="${PREDICTIVE_REPLAY_ARGS_SCRIPT_IN_CONTAINER:-/root/miles/scripts_mine/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/_predictive_replay_args.sh}"
export CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-autofix}"
export TRANSFORMERS_HOST_ROOT="${TRANSFORMERS_HOST_ROOT:-/mnt/weka/home/hongyi.wang/workspace/transformers-main}"
export SGLANG_HOST_ROOT="${SGLANG_HOST_ROOT:-/mnt/weka/home/hongyi.wang/workspace/xllm-sglang}"
export TRANSFORMERS_SRC_IN_CONTAINER="${TRANSFORMERS_SRC_IN_CONTAINER:-/root/transformers-main/src}"
export SGLANG_SRC_IN_CONTAINER="${SGLANG_SRC_IN_CONTAINER:-/root/xllm-sglang/python}"
export RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH:-${SGLANG_SRC_IN_CONTAINER}:/root/miles:/root/Megatron-LM}"
export RUNTIME_PIP_INSTALL_COMMAND="${RUNTIME_PIP_INSTALL_COMMAND:-pip install --no-cache-dir --no-deps --force-reinstall transformers==4.57.1 huggingface-hub==0.36.0 tokenizers==0.22.2 jinja2==3.1.6 timm==1.0.23 sgl-kernel==0.3.21}"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/mnt/weka/shrd/k2m/haolong.jia:/mnt/weka/shrd/k2m/haolong.jia:rw,/mnt/weka/shrd/k2m/hongyi.wang:/mnt/weka/shrd/k2m/hongyi.wang:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/miles:/root/miles:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/verl:/root/verl:ro,${TRANSFORMERS_HOST_ROOT}:/root/transformers-main:ro,${SGLANG_HOST_ROOT}:/root/xllm-sglang:ro}"

export RUN_POSTFIX="${RUN_POSTFIX:-off2-baseline}"
export RESOURCE_LAYOUT="${RESOURCE_LAYOUT:-disagg}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
export ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-1}"
export ENABLE_KEEP_OLD_ACTOR="${ENABLE_KEEP_OLD_ACTOR:-1}"
export ENABLE_AUTO_CKPT_EVAL="${ENABLE_AUTO_CKPT_EVAL:-1}"
export UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
export USE_MILES_ROUTER="${USE_MILES_ROUTER:-0}"
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-0}"
export USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-0}"
export ENABLE_PREDICTIVE_ROUTING_REPLAY="${ENABLE_PREDICTIVE_ROUTING_REPLAY:-0}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-32768}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-256}"
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-128}"
export DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
export ENABLE_PARTIAL_ROLLOUT="${ENABLE_PARTIAL_ROLLOUT:-1}"
export ENABLE_MASK_OFFPOLICY_IN_PARTIAL_ROLLOUT="${ENABLE_MASK_OFFPOLICY_IN_PARTIAL_ROLLOUT:-1}"
export ROLLOUT_STOP_TOKEN_IDS="${ROLLOUT_STOP_TOKEN_IDS:-248046 248044}"

exec bash "${BASE_SCRIPT}"
