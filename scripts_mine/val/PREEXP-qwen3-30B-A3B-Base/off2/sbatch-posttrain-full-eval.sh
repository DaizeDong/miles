#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=k2m
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --output=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.log
#SBATCH --error=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.err
#SBATCH --open-mode=append

set -euo pipefail

: "${TRAIN_JOB_ID:?TRAIN_JOB_ID environment variable must be set}"
: "${RUN_NAME:?RUN_NAME environment variable must be set}"
: "${CKPT_ROOT:?CKPT_ROOT environment variable must be set}"

RESULTS_ROOT="${RESULTS_ROOT:-/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/results}"
WAIT_INTERVAL="${WAIT_INTERVAL:-30}"

bash /mnt/weka/home/hongyi.wang/workspace/rlhf/miles/scripts_mine/val/PREEXP-qwen3-30B-A3B-Base/off2/watch_train_and_eval.sh
