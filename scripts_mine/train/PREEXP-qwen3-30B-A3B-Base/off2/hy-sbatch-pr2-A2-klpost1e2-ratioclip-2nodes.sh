#!/bin/bash
# === Ablation A2: A0 + delta/old-logits magnitude clip ===
# Caps |delta|.mean() to <= 0.015 * |old_logits|.mean(); rejects predictor
# outputs that are too large relative to base routing scores.
#SBATCH --job-name=miles-off2-pr2-A2-klpost1e2-ratioclip
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --account=k2m
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --exclusive
#SBATCH --output=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.log
#SBATCH --error=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.err
#SBATCH --open-mode=append

set -euo pipefail

BASE_SCRIPT="/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/scripts_mine/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/hy-sbatch-8nodes.sh"
export RUN_POSTFIX="${RUN_POSTFIX:-off2-pr2-A2-klpost1e2-ratioclip-r0015}"
export RESOURCE_LAYOUT="${RESOURCE_LAYOUT:-disagg}"
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-540}"
export ENABLE_ASYNC_TRAIN="${ENABLE_ASYNC_TRAIN:-1}"
export ENABLE_KEEP_OLD_ACTOR="${ENABLE_KEEP_OLD_ACTOR:-1}"
export ENABLE_AUTO_CKPT_EVAL="${ENABLE_AUTO_CKPT_EVAL:-1}"
export AUTO_CKPT_EVAL_BENCHMARKS="${AUTO_CKPT_EVAL_BENCHMARKS:-aime2024}"
export UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
export USE_MILES_ROUTER="${USE_MILES_ROUTER:-1}"
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-1}"
export USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-0}"
export ENABLE_PREDICTIVE_ROUTING_REPLAY="${ENABLE_PREDICTIVE_ROUTING_REPLAY:-1}"
export BIAS_PREDICTOR_LOSS_TYPE="${BIAS_PREDICTOR_LOSS_TYPE:-kl-post}"
export BIAS_PREDICTOR_LR_MULT="${BIAS_PREDICTOR_LR_MULT:-1e2}"
export PREDICTIVE_DOWNSAMPLE_BATCH_SIZE="${PREDICTIVE_DOWNSAMPLE_BATCH_SIZE:-2}"
export PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT="${PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT:-8192}"
export PREDICTIVE_MAX_TOTAL_TOKENS="${PREDICTIVE_MAX_TOTAL_TOKENS:-4096}"
export PREDICTIVE_STORAGE_DTYPE="${PREDICTIVE_STORAGE_DTYPE:-fp32}"
# --- Enhancement under test ---
export PREDICTIVE_MAX_DELTA_TO_OLD_RATIO="${PREDICTIVE_MAX_DELTA_TO_OLD_RATIO:-0.015}"

exec bash "${BASE_SCRIPT}"
