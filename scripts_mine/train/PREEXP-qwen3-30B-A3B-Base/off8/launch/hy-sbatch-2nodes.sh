#!/bin/bash
set -euo pipefail

BASE_SCRIPT="/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/scripts_mine/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/hy-sbatch-2nodes.sh"
export OFF_POLICY_LABEL="${OFF_POLICY_LABEL:-off8}"
export RUN_POSTFIX="${RUN_POSTFIX:-off8-2nodes-baseline}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-8}"

exec bash "${BASE_SCRIPT}"
