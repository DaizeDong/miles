#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=hwang
#SBATCH --partition=mi3008x
#SBATCH --time=12:00:00
#SBATCH --output=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.log
#SBATCH --error=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.err
#SBATCH --open-mode=append

set -euo pipefail

: "${TRAIN_JOB_ID:?TRAIN_JOB_ID environment variable must be set}"
: "${RUN_NAME:?RUN_NAME environment variable must be set}"
: "${CKPT_ROOT:?CKPT_ROOT environment variable must be set}"

RESULTS_ROOT="${RESULTS_ROOT:-/home1/dzdong/workspace/miles/results}"
WAIT_INTERVAL="${WAIT_INTERVAL:-30}"

bash /home1/dzdong/workspace/miles/scripts_my_hpcfund/val/PREEXP-qwen3-30B-A3B-Base/off2/watch_train_and_eval.sh
