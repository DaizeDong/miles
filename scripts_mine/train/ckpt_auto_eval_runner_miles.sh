#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=k2m
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --output=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.log
#SBATCH --error=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.err
#SBATCH --open-mode=append

set -euo pipefail

: "${RUN_NAME:?RUN_NAME environment variable must be set}"
: "${CKPT_ROOT:?CKPT_ROOT environment variable must be set}"
: "${MODEL_PATH:?MODEL_PATH environment variable must be set}"
: "${STEP_LABEL:?STEP_LABEL environment variable must be set}"
: "${RESULTS_ROOT:?RESULTS_ROOT environment variable must be set}"
: "${AUTO_CKPT_EVAL_DONE_DIR:?AUTO_CKPT_EVAL_DONE_DIR environment variable must be set}"
: "${AUTO_CKPT_EVAL_FAILED_DIR:?AUTO_CKPT_EVAL_FAILED_DIR environment variable must be set}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${MILES_REPO_ROOT:-}" ]; then
  REPO_ROOT="${MILES_REPO_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/scripts_mine" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi
RUN_FULL_EVAL_SCRIPT="${AUTO_CKPT_EVAL_RUN_FULL_EVAL_SCRIPT:-${REPO_ROOT}/scripts_mine/val/PREEXP-qwen3-30B-A3B-Base/off2/run_full_eval.sh}"

mkdir -p "${AUTO_CKPT_EVAL_DONE_DIR}" "${AUTO_CKPT_EVAL_FAILED_DIR}"
SUBMITTED_MARKER=""
if [ -n "${AUTO_CKPT_EVAL_SUBMITTED_DIR:-}" ]; then
  mkdir -p "${AUTO_CKPT_EVAL_SUBMITTED_DIR}"
  SUBMITTED_MARKER="${AUTO_CKPT_EVAL_SUBMITTED_DIR}/${STEP_LABEL}.submitted"
fi

status="failed"
cleanup() {
  if [ -n "${SUBMITTED_MARKER}" ]; then
    rm -f "${SUBMITTED_MARKER}"
  fi
  if [ "${status}" = "success" ]; then
    touch "${AUTO_CKPT_EVAL_DONE_DIR}/${STEP_LABEL}.done"
    rm -f "${AUTO_CKPT_EVAL_FAILED_DIR}/${STEP_LABEL}.failed"
  else
    printf '%s\n' "failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${AUTO_CKPT_EVAL_FAILED_DIR}/${STEP_LABEL}.failed"
  fi
}
trap cleanup EXIT

echo "[auto-eval-runner] RUN_NAME=${RUN_NAME}"
echo "[auto-eval-runner] STEP_LABEL=${STEP_LABEL}"
echo "[auto-eval-runner] MODEL_PATH=${MODEL_PATH}"
echo "[auto-eval-runner] RESULTS_ROOT=${RESULTS_ROOT}"

bash "${RUN_FULL_EVAL_SCRIPT}"

status="success"
