#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --account=k2m
#SBATCH --qos=lowprio
#SBATCH --partition=lowprio
#SBATCH --exclusive
#SBATCH --output=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.log
#SBATCH --error=/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/logs_tmp/sbatch/%x-%J/%N.%J.%t.err
#SBATCH --open-mode=append

set -euo pipefail

: "${CKPT_ROOT:?CKPT_ROOT environment variable must be set}"

EXPORT_STEP_LABEL="${EXPORT_STEP_LABEL:-}"
EXPORT_SUBMITTED_MARKER=""
EXPORT_FAILED_MARKER=""
EXPORT_STATUS="failed"
if [ -n "${EXPORT_STEP_LABEL}" ] && [ -n "${AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR:-}" ]; then
  mkdir -p "${AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR}"
  EXPORT_SUBMITTED_MARKER="${AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR}/${EXPORT_STEP_LABEL}.submitted"
fi
if [ -n "${EXPORT_STEP_LABEL}" ] && [ -n "${AUTO_CKPT_EVAL_EXPORT_FAILED_DIR:-}" ]; then
  mkdir -p "${AUTO_CKPT_EVAL_EXPORT_FAILED_DIR}"
  EXPORT_FAILED_MARKER="${AUTO_CKPT_EVAL_EXPORT_FAILED_DIR}/${EXPORT_STEP_LABEL}.failed"
fi

cleanup() {
  if [ "${EXPORT_STATUS}" = "success" ]; then
    rm -f "${EXPORT_SUBMITTED_MARKER}" "${EXPORT_FAILED_MARKER}"
  else
    rm -f "${EXPORT_SUBMITTED_MARKER}"
    if [ -n "${EXPORT_FAILED_MARKER}" ]; then
      printf '%s\n' "failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${EXPORT_FAILED_MARKER}"
    fi
  fi
}
trap cleanup EXIT

MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B-Base}"
HF_MODEL_PATH="${HF_MODEL_PATH:-/mnt/weka/shrd/k2m/haolong.jia/checkpoint/${MODEL_NAME}}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/weka/shrd/k2m/haolong.jia/checkpoint_torch_dist/${MODEL_NAME}}"
DATA_CACHE_ROOT="${DATA_CACHE_ROOT:-/mnt/weka/shrd/k2m/hongyi.wang/datasets_miles/PREEXP-${MODEL_NAME}}"
MILES_TRAIN_FILE="${MILES_TRAIN_FILE:-${DATA_CACHE_ROOT}/dapo-math-17k-miles.jsonl}"
MODEL_SCRIPT_PATH_IN_CONTAINER="${MODEL_SCRIPT_PATH_IN_CONTAINER:-/root/miles/scripts/models/qwen3-30B-A3B.sh}"
EXPORT_HF_FROM_MEGATRON_SCRIPT_IN_CONTAINER="${EXPORT_HF_FROM_MEGATRON_SCRIPT_IN_CONTAINER:-/root/miles/scripts_mine/val/PREEXP-qwen3-30B-A3B-Base/off2/export_hf_from_megatron.py}"
if [ -z "${SAVE_HF_TEMPLATE:-}" ]; then
  SAVE_HF_TEMPLATE="$(printf '%s/hf/rollout_{rollout_id:04d}' "${CKPT_ROOT}")"
fi
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/mnt/weka/shrd/k2m/hongyi.wang/containers/slimerl+slime+latest.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/mnt/weka/shrd/k2m/haolong.jia:/mnt/weka/shrd/k2m/haolong.jia:rw,/mnt/weka/shrd/k2m/hongyi.wang:/mnt/weka/shrd/k2m/hongyi.wang:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/miles:/root/miles:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/verl:/root/verl:ro}"

echo "================================================================"
echo "Starting HF export from Megatron checkpoint"
echo "  CKPT_ROOT: ${CKPT_ROOT}"
echo "  HF_MODEL_PATH: ${HF_MODEL_PATH}"
echo "  SAVE_HF_TEMPLATE: ${SAVE_HF_TEMPLATE}"
echo "  EXPORT_STEP_LABEL: ${EXPORT_STEP_LABEL:-latest}"
echo "  MODEL_SCRIPT_PATH_IN_CONTAINER: ${MODEL_SCRIPT_PATH_IN_CONTAINER}"
echo "  EXPORT_HF_FROM_MEGATRON_SCRIPT_IN_CONTAINER: ${EXPORT_HF_FROM_MEGATRON_SCRIPT_IN_CONTAINER}"
echo "  Node: $(hostname -s)"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "================================================================"

srun \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --export=ALL \
  bash -c '
set -euo pipefail
cd /root/miles
export PYTHONPATH="/root/miles:/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEPRECATED_MEGATRON_COMPATIBLE=1
export EXPORT_LOAD_PATH="${EXPORT_LOAD_PATH:-${CKPT_ROOT}}"
export MEGATRON_LOAD_PATH="${MEGATRON_LOAD_PATH:-${CKPT_ROOT}}"
source "${MODEL_SCRIPT_PATH_IN_CONTAINER}"
EXTRA_ARGS=()
if [ -n "${CHAT_TEMPLATE_PATH:-}" ]; then
  EXTRA_ARGS+=(--chat-template-path "${CHAT_TEMPLATE_PATH}")
fi
if [ -n "${APPLY_CHAT_TEMPLATE_KWARGS:-}" ]; then
  EXTRA_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi
torchrun --nproc-per-node 8 "'"${EXPORT_HF_FROM_MEGATRON_SCRIPT_IN_CONTAINER}"'" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "'"${HF_MODEL_PATH}"'" \
  --ref-load "'"${REF_MODEL_PATH}"'" \
  --load "'"${CKPT_ROOT}"'" \
  --save-hf "'"${SAVE_HF_TEMPLATE}"'" \
  --prompt-data "'"${MILES_TRAIN_FILE}"'" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --prompt-truncation left \
  --rollout-shuffle \
  --rm-type dapo \
  --reward-key score \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 8 \
  --rollout-max-prompt-len "'"${MAX_PROMPT_LENGTH}"'" \
  --rollout-max-response-len "'"${MAX_RESPONSE_LENGTH}"'" \
  --rollout-temperature 1 \
  --rollout-top-p 1 \
  --rollout-top-k -1 \
  --global-batch-size 8 \
  --balance-data \
  "${EXTRA_ARGS[@]}" \
  --tensor-model-parallel-size 1 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --no-load-optim \
  --no-load-rng \
  --finetune \
  --attention-backend flash
'

EXPORT_STATUS="success"
