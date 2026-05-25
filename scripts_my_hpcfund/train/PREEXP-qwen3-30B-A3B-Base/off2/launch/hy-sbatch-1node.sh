#!/bin/bash
# Verl-aligned 1-node baseline-style config.
#SBATCH --job-name=miles-qwen30b-1node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=hwang
#SBATCH --partition=mi3008x
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.log
#SBATCH --error=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.err
#SBATCH --open-mode=append

set -euo pipefail

LAUNCH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${LAUNCH_SCRIPT_DIR}/../../../../.." && pwd)"
export MILES_REPO_ROOT="${MILES_REPO_ROOT:-${REPO_ROOT}}"

resolve_repo_path() {
  local path="$1"
  case "$path" in
    /*) printf '%s\n' "$path" ;;
    *) printf '%s/%s\n' "${MILES_REPO_ROOT}" "${path#./}" ;;
  esac
}

source "${LAUNCH_SCRIPT_DIR}/_predictive_replay_args.sh"

###########################################
# basic info
export MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B-Base}"
export OFF_POLICY_LABEL="${OFF_POLICY_LABEL:-off2}"
export RUN_POSTFIX="${RUN_POSTFIX:-${OFF_POLICY_LABEL}-1node-baseline}"
###########################################

export TIME="$(date +%Y%m%d-%H%M%S)"
export LOGROOT="${LOGROOT:-/work1/hwang/dzdong/miles_logs/${MODEL_NAME}/${RUN_POSTFIX}/${RUN_POSTFIX}-${SLURM_JOB_ID}}"
mkdir -p "$LOGROOT"

log_file="${TIME}.${SLURMD_NODENAME}.${SLURM_JOB_ID}.${SLURM_PROCID}.log"
err_file="${TIME}.${SLURMD_NODENAME}.${SLURM_JOB_ID}.${SLURM_PROCID}.err"

exec 3>&1 4>&2
trap 'exec 1>&3 2>&4 3>&- 4>&-' EXIT

exec > >(tee -a "$LOGROOT/${log_file}" >&3)
exec 2> >(tee -a "$LOGROOT/${err_file}" >&4)

echo "[INFO] stdout redirected $LOGROOT/${log_file}"
echo "[INFO] stderr redirected $LOGROOT/${err_file}"
echo "========== SLURM ENVIRONMENT DUMP =========="
env | grep '^SLURM' | sort || true
echo "============================================"

export HF_MODEL_PATH="${HF_MODEL_PATH:-/work1/hwang/dzdong/checkpoints_hf/${MODEL_NAME}}"
export MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-/work1/hwang/dzdong/checkpoints_torch_dist/${MODEL_NAME}}"
export RAW_TRAIN_FILE="${RAW_TRAIN_FILE:-/work1/hwang/dzdong/data/DAPO-Math-17k/dapo-math-17k.jsonl}"
export RAW_EVAL_FILE="${RAW_EVAL_FILE:-/work1/hwang/dzdong/data/aime-2024/aime-2024.jsonl}"
export DATA_CACHE_ROOT="${DATA_CACHE_ROOT:-/work1/hwang/dzdong/datasets_miles/${MODEL_NAME}}"
export MILES_TRAIN_FILE="${MILES_TRAIN_FILE:-${DATA_CACHE_ROOT}/dapo-math-17k-miles.jsonl}"
export MILES_EVAL_FILE="${MILES_EVAL_FILE:-${DATA_CACHE_ROOT}/aime-2024-miles.jsonl}"
export SAVE_PATH="${SAVE_PATH:-/work1/hwang/dzdong/checkpoint_miles/${MODEL_NAME}-${RUN_POSTFIX}}"
export LOAD_PATH="${LOAD_PATH:-${SAVE_PATH}}"
export LOAD_AS_FINETUNE="${LOAD_AS_FINETUNE:-0}"
if [ -z "${SAVE_HF_TEMPLATE:-}" ]; then
  export SAVE_HF_TEMPLATE
  SAVE_HF_TEMPLATE="$(printf '%s/hf/rollout_{rollout_id:04d}' "${SAVE_PATH}")"
fi
export ROUTER_LOGITS_PATH="${ROUTER_LOGITS_PATH:-${SAVE_PATH}/router_logits}"
export ROUTER_LOGITS_SAVE_FREQ="${ROUTER_LOGITS_SAVE_FREQ:-10}"
export ROUTER_LOGITS_MAX_TOKENS="${ROUTER_LOGITS_MAX_TOKENS-100000}"
export CKPT_MONITOR_SCRIPT="$(resolve_repo_path "${CKPT_MONITOR_SCRIPT:-scripts_my_hpcfund/train/ckpt_monitor_miles.sh}")"
export CKPT_KEEP_LATEST="${CKPT_KEEP_LATEST:-1}"
export CKPT_MONITOR_INTERVAL="${CKPT_MONITOR_INTERVAL:-120}"
export CKPT_MONITOR_LOG="${CKPT_MONITOR_LOG:-${SAVE_PATH}/ckpt_monitor.host.log}"
export ENABLE_AUTO_CKPT_EVAL="${ENABLE_AUTO_CKPT_EVAL:-1}"
export AUTO_CKPT_EVAL_MONITOR_SCRIPT="$(resolve_repo_path "${AUTO_CKPT_EVAL_MONITOR_SCRIPT:-scripts_my_hpcfund/train/ckpt_auto_eval_monitor_miles.sh}")"
export AUTO_CKPT_EVAL_RUNNER_SCRIPT="$(resolve_repo_path "${AUTO_CKPT_EVAL_RUNNER_SCRIPT:-scripts_my_hpcfund/train/ckpt_auto_eval_runner_miles.sh}")"
export AUTO_CKPT_EVAL_RESULTS_ROOT="${AUTO_CKPT_EVAL_RESULTS_ROOT:-${SAVE_PATH}}"
export AUTO_CKPT_EVAL_STATE_ROOT="${AUTO_CKPT_EVAL_STATE_ROOT:-/work1/hwang/dzdong/miles_logs_tmp/auto_ckpt_eval/$(basename "${SAVE_PATH}")}"
export AUTO_CKPT_EVAL_DONE_DIR="${AUTO_CKPT_EVAL_DONE_DIR:-${AUTO_CKPT_EVAL_STATE_ROOT}/done}"
export AUTO_CKPT_EVAL_FAILED_DIR="${AUTO_CKPT_EVAL_FAILED_DIR:-${AUTO_CKPT_EVAL_STATE_ROOT}/failed}"
export AUTO_CKPT_EVAL_SUBMITTED_DIR="${AUTO_CKPT_EVAL_SUBMITTED_DIR:-${AUTO_CKPT_EVAL_STATE_ROOT}/submitted}"
export AUTO_CKPT_EVAL_RUN_NAME="${AUTO_CKPT_EVAL_RUN_NAME:-eval}"
export AUTO_CKPT_EVAL_POLL_INTERVAL="${AUTO_CKPT_EVAL_POLL_INTERVAL:-120}"
export AUTO_CKPT_EVAL_READY_GRACE_SEC="${AUTO_CKPT_EVAL_READY_GRACE_SEC:-30}"
export AUTO_CKPT_EVAL_WAIT_INTERVAL="${AUTO_CKPT_EVAL_WAIT_INTERVAL:-30}"
export AUTO_CKPT_EVAL_BENCHMARKS="${AUTO_CKPT_EVAL_BENCHMARKS:-}"
export AUTO_CKPT_EVAL_EXPORT_ROOT="${AUTO_CKPT_EVAL_EXPORT_ROOT:-/home1/dzdong/workspace/miles/scripts_my_hpcfund/val}"
export AUTO_CKPT_EVAL_EXPORT_SERIES_NAME="${AUTO_CKPT_EVAL_EXPORT_SERIES_NAME:-PREEXP-${MODEL_NAME}-${RUN_POSTFIX}}"
export AUTO_CKPT_EVAL_LOG="${AUTO_CKPT_EVAL_LOG:-${SAVE_PATH}/auto_ckpt_eval.host.log}"

export NUM_ROLLOUT="${NUM_ROLLOUT:-270}"
export START_ROLLOUT_ID="${START_ROLLOUT_ID:-}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT))}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export EVAL_MAX_RESPONSE_LENGTH="${EVAL_MAX_RESPONSE_LENGTH:-16384}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-18432}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
export LR="${LR:-2e-6}"
export RM_TYPE="${RM_TYPE:-dapo}"
export REWARD_KEY="${REWARD_KEY:-}"
export PROMPT_TRUNCATION="${PROMPT_TRUNCATION:-left}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.7}"
export EVAL_TOP_K="${EVAL_TOP_K:--1}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_KEY_FILE="${WANDB_KEY_FILE:-/root/miles/scripts_my_hpcfund/wandb_key_sandy.txt}"
export WANDB_PROJECT="${WANDB_PROJECT:-miles+${MODEL_NAME}}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_POSTFIX}}"
export ENABLE_CPU_OPTIMIZER_OFFLOAD="${ENABLE_CPU_OPTIMIZER_OFFLOAD:-1}"
export ENABLE_NO_SAVE_OPTIM="${ENABLE_NO_SAVE_OPTIM:-0}"
export ENABLE_TP_COMM_OVERLAP="${ENABLE_TP_COMM_OVERLAP:-0}"
export ENABLE_MOE_DEEPEP="${ENABLE_MOE_DEEPEP:-0}"
export ENABLE_SGLANG_EP_MOE="${ENABLE_SGLANG_EP_MOE:-0}"
export SGLANG_EXPERT_PARALLEL_SIZE="${SGLANG_EXPERT_PARALLEL_SIZE:-8}"
export ENABLE_SGLANG_DP_ATTENTION="${ENABLE_SGLANG_DP_ATTENTION:-0}"
export SGLANG_DP_SIZE="${SGLANG_DP_SIZE:-8}"
export SGLANG_MOE_DENSE_TP_SIZE="${SGLANG_MOE_DENSE_TP_SIZE:-1}"
export ENABLE_SGLANG_DP_LM_HEAD="${ENABLE_SGLANG_DP_LM_HEAD:-0}"
export ENABLE_SGLANG_DEEPEP_MOE="${ENABLE_SGLANG_DEEPEP_MOE:-0}"
export SGLANG_DEEPEP_MODE="${SGLANG_DEEPEP_MODE:-auto}"
export SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-}"
export SGLANG_MOE_A2A_BACKEND="${SGLANG_MOE_A2A_BACKEND:-}"
export SGLANG_MOE_RUNNER_BACKEND="${SGLANG_MOE_RUNNER_BACKEND:-}"
export SGLANG_CUDA_GRAPH_MAX="${SGLANG_CUDA_GRAPH_MAX:-512}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-512}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-512}"
export USE_ROUTING_REPLAY="${USE_ROUTING_REPLAY:-0}"
export USE_MILES_ROUTER="${USE_MILES_ROUTER:-0}"
export USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-0}"
export ENABLE_PREDICTIVE_ROUTING_REPLAY="${ENABLE_PREDICTIVE_ROUTING_REPLAY:-0}"
export BIAS_PREDICTOR_LOSS_TYPE="${BIAS_PREDICTOR_LOSS_TYPE:-kl-post}"
export BIAS_PREDICTOR_LR_MULT="${BIAS_PREDICTOR_LR_MULT:-1e3}"
export PREDICTIVE_DOWNSAMPLE_BATCH_SIZE="${PREDICTIVE_DOWNSAMPLE_BATCH_SIZE:-2}"
export PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT="${PREDICTIVE_DOWNSAMPLE_MAX_LEN_LIMIT:-4096}"
export PREDICTIVE_MAX_TOTAL_TOKENS="${PREDICTIVE_MAX_TOTAL_TOKENS:-}"
export PREDICTIVE_LAYER_SCALE_SCHEDULE="${PREDICTIVE_LAYER_SCALE_SCHEDULE:-none}"
export PREDICTIVE_LAYER_SCALE_MIN="${PREDICTIVE_LAYER_SCALE_MIN:-1.0}"
export PREDICTIVE_MAX_DELTA_TO_OLD_RATIO="${PREDICTIVE_MAX_DELTA_TO_OLD_RATIO:-}"
export PREDICTIVE_STORAGE_DTYPE="${PREDICTIVE_STORAGE_DTYPE:-fp32}"

if [ "${EVAL_INTERVAL}" -eq 0 ]; then
  echo "[WARN] EVAL_INTERVAL=0 disables in-training eval and W&B eval curves."
else
  echo "[INFO] In-training eval enabled every ${EVAL_INTERVAL} rollout steps with ${MILES_EVAL_FILE}"
fi

export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/work1/hwang/dzdong/images/miles-mi300.sif}"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/work1/hwang/dzdong:/work1/hwang/dzdong:rw,/work1/hwang/dzdong:/work1/hwang/dzdong:rw,/home1/dzdong/workspace/miles:/root/miles:rw,/work1/hwang/dzdong/verl:/root/verl:ro}"

if [ "${SAVE_INTERVAL}" -le 0 ]; then
  echo "[WARN] Megatron requires save_interval, overriding SAVE_INTERVAL=${SAVE_INTERVAL} -> 1"
  export SAVE_INTERVAL=1
fi

if [ -n "${ROUTER_LOGITS_PATH}" ] && [ "${ROUTER_LOGITS_SAVE_FREQ}" -le 0 ]; then
  echo "[ERROR] ROUTER_LOGITS_SAVE_FREQ must be positive when ROUTER_LOGITS_PATH is enabled, got ${ROUTER_LOGITS_SAVE_FREQ}" >&2
  exit 1
fi
if [ -n "${ROUTER_LOGITS_MAX_TOKENS}" ] && [ "${ROUTER_LOGITS_MAX_TOKENS}" -le 0 ]; then
  echo "[ERROR] ROUTER_LOGITS_MAX_TOKENS must be positive when set, got ${ROUTER_LOGITS_MAX_TOKENS}" >&2
  exit 1
fi

echo "[INFO] SAVE_HF_TEMPLATE=${SAVE_HF_TEMPLATE:-disabled}"
echo "[INFO] ROUTER_LOGITS_PATH=${ROUTER_LOGITS_PATH:-disabled}"
echo "[INFO] ROUTER_LOGITS_MAX_TOKENS=${ROUTER_LOGITS_MAX_TOKENS:-disabled}"

export ROLLOUT_SAMPLES_PER_STEP=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

for required_positive in \
  ROLLOUT_BATCH_SIZE \
  N_SAMPLES_PER_PROMPT \
  NUM_STEPS_PER_ROLLOUT \
  GLOBAL_BATCH_SIZE \
  ROLLOUT_NUM_GPUS_PER_ENGINE \
  SGLANG_MAX_RUNNING_REQUESTS \
  SGLANG_SERVER_CONCURRENCY; do
  if [ "${!required_positive}" -le 0 ]; then
    echo "[ERROR] ${required_positive} must be positive, got ${!required_positive}" >&2
    exit 1
  fi
done

if [ $((ROLLOUT_SAMPLES_PER_STEP % NUM_STEPS_PER_ROLLOUT)) -ne 0 ]; then
  echo "[ERROR] rollout samples per step (${ROLLOUT_SAMPLES_PER_STEP}) must be divisible by NUM_STEPS_PER_ROLLOUT (${NUM_STEPS_PER_ROLLOUT})" >&2
  exit 1
fi

export EXPECTED_GLOBAL_BATCH_SIZE=$((ROLLOUT_SAMPLES_PER_STEP / NUM_STEPS_PER_ROLLOUT))
if [ "${GLOBAL_BATCH_SIZE}" -ne "${EXPECTED_GLOBAL_BATCH_SIZE}" ]; then
  echo "[ERROR] GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must equal rollout samples per step (${ROLLOUT_SAMPLES_PER_STEP}) / NUM_STEPS_PER_ROLLOUT (${NUM_STEPS_PER_ROLLOUT}) = ${EXPECTED_GLOBAL_BATCH_SIZE} for ${OFF_POLICY_LABEL} launchers." >&2
  exit 1
fi

for required_path in "$HF_MODEL_PATH" "$REF_MODEL_PATH" "$RAW_TRAIN_FILE" "$RAW_EVAL_FILE"; do
  if [ ! -e "$required_path" ]; then
    echo "[ERROR] required path not found: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$DATA_CACHE_ROOT" "$SAVE_PATH"

prepare_miles_jsonl() {
  local src="$1"
  local dst="$2"
  local tag="$3"

  if [ -s "$dst" ] && [ "$dst" -nt "$src" ]; then
    echo "[INFO] reuse cached ${tag} dataset: $dst"
    return
  fi

  echo "[INFO] building ${tag} dataset for miles: $src -> $dst"
  python - "$src" "$dst" <<'PY'
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]

if src.endswith(".parquet"):
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(src).iter_batches()
    rows = (row for batch in reader for row in batch.to_pylist())
elif src.endswith(".jsonl"):
    def iter_jsonl(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    rows = iter_jsonl(src)
else:
    raise ValueError(f"Unsupported dataset format: {src}")

os.makedirs(os.path.dirname(dst), exist_ok=True)

count = 0
with open(dst, "w", encoding="utf-8") as fout:
    for row in rows:
        prompt = row.get("prompt")
        if prompt is None:
            raise KeyError(f"'prompt' not found in source row for {src}")

        label = row.get("label")
        if label is None:
            reward_model = row.get("reward_model") or {}
            if isinstance(reward_model, dict):
                label = reward_model.get("ground_truth")

        if label is None:
            raise KeyError(f"Cannot derive label from row in {src}")

        metadata = {}
        for key in ("data_source", "ability", "extra_info"):
            if key in row and row[key] is not None:
                metadata[key] = row[key]
        if "reward_model" in row and isinstance(row["reward_model"], dict):
            metadata["reward_model"] = row["reward_model"]

        out = {
            "prompt": prompt,
            "label": str(label),
        }
        if metadata:
            out["metadata"] = metadata

        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        count += 1

print(f"[INFO] wrote {count} rows to {dst}")
PY
}

prepare_miles_jsonl "$RAW_TRAIN_FILE" "$MILES_TRAIN_FILE" "train"
prepare_miles_jsonl "$RAW_EVAL_FILE" "$MILES_EVAL_FILE" "eval"

echo "[INFO] HF_MODEL_PATH=$HF_MODEL_PATH"
echo "[INFO] REF_MODEL_PATH=$REF_MODEL_PATH"
echo "[INFO] MILES_TRAIN_FILE=$MILES_TRAIN_FILE"
echo "[INFO] MILES_EVAL_FILE=$MILES_EVAL_FILE"
echo "[INFO] SAVE_PATH=$SAVE_PATH"
echo "[INFO] LOAD_PATH=${LOAD_PATH:-disabled}"
echo "[INFO] rollout samples per step=${ROLLOUT_SAMPLES_PER_STEP}, num steps per rollout=${NUM_STEPS_PER_ROLLOUT}, global batch size=${GLOBAL_BATCH_SIZE}"
echo "[INFO] CONTAINER_IMAGE=$CONTAINER_IMAGE"
echo "[INFO] CKPT_MONITOR_SCRIPT=${CKPT_MONITOR_SCRIPT}"
echo "[INFO] AUTO_CKPT_EVAL_MONITOR_SCRIPT=${AUTO_CKPT_EVAL_MONITOR_SCRIPT}"
echo "[INFO] AUTO_CKPT_EVAL_RUNNER_SCRIPT=${AUTO_CKPT_EVAL_RUNNER_SCRIPT}"
echo "[INFO] optional optimizations: cpu_offload=${ENABLE_CPU_OPTIMIZER_OFFLOAD}, no_save_optim=${ENABLE_NO_SAVE_OPTIM}, tp_comm_overlap=${ENABLE_TP_COMM_OVERLAP}, megatron_deepep=${ENABLE_MOE_DEEPEP}, sglang_ep_moe=${ENABLE_SGLANG_EP_MOE}, sglang_ep_size=${SGLANG_EXPERT_PARALLEL_SIZE}, sglang_dp_attention=${ENABLE_SGLANG_DP_ATTENTION}, sglang_dp_size=${SGLANG_DP_SIZE}, sglang_dp_lm_head=${ENABLE_SGLANG_DP_LM_HEAD}, sglang_deepep=${ENABLE_SGLANG_DEEPEP_MOE}, sglang_moe_a2a_backend=${SGLANG_MOE_A2A_BACKEND:-none}, sglang_moe_runner_backend=${SGLANG_MOE_RUNNER_BACKEND:-none}, router_logits_path=${ROUTER_LOGITS_PATH:-disabled}, router_logits_save_freq=${ROUTER_LOGITS_SAVE_FREQ}, sglang_kv_cache_dtype=${SGLANG_KV_CACHE_DTYPE:-none}, sglang_cuda_graph_max=${SGLANG_CUDA_GRAPH_MAX}"

require_host_script() {
  local label="$1"
  local path="$2"
  if [ ! -f "${path}" ]; then
    echo "[ERROR] ${label} not found: ${path}" >&2
    exit 1
  fi
}

if [ "${CKPT_KEEP_LATEST}" -ne -1 ]; then
  require_host_script "checkpoint monitor script" "${CKPT_MONITOR_SCRIPT}"
fi

if [ "${ENABLE_AUTO_CKPT_EVAL}" = "1" ] && [ -n "${SAVE_HF_TEMPLATE}" ]; then
  require_host_script "auto checkpoint eval monitor script" "${AUTO_CKPT_EVAL_MONITOR_SCRIPT}"
  require_host_script "auto checkpoint eval runner script" "${AUTO_CKPT_EVAL_RUNNER_SCRIPT}"
fi

AUTO_CKPT_EVAL_HOST_PID=""
if [ "${ENABLE_AUTO_CKPT_EVAL}" = "1" ]; then
  if [ -z "${SAVE_HF_TEMPLATE}" ]; then
    echo "[WARN] ENABLE_AUTO_CKPT_EVAL=1 but SAVE_HF_TEMPLATE is empty; auto eval monitor will not start."
  else
    echo "[INFO] Launching auto checkpoint eval monitor on host..."
    echo "[INFO] AUTO_CKPT_EVAL_LOG=${AUTO_CKPT_EVAL_LOG}"
    : > "${AUTO_CKPT_EVAL_LOG}"
    nohup bash "${AUTO_CKPT_EVAL_MONITOR_SCRIPT}" "${SAVE_PATH}" >> "${AUTO_CKPT_EVAL_LOG}" 2>&1 < /dev/null &
    AUTO_CKPT_EVAL_HOST_PID=$!
    echo "[INFO] AUTO_CKPT_EVAL_HOST_PID=${AUTO_CKPT_EVAL_HOST_PID}"
  fi
fi

set +e
srun \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --export=ALL \
  apptainer exec --rocm --writable-tmpfs \
    --bind /work1/hwang/dzdong:/work1/hwang/dzdong \
    --bind /home1/dzdong/workspace/miles:/root/miles \
    "${CONTAINER_IMAGE}" \
  bash -c '
set -euo pipefail
set -x

rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED || true

cd /root/miles
pip install -e . --no-deps
source /root/miles/scripts_my_hpcfund/train/PREEXP-qwen3-30B-A3B-Base/off2/launch/_predictive_replay_args.sh

if [ ! -d /app/Megatron-LM ]; then
  echo "[ERROR] /app/Megatron-LM not found in container." >&2
  exit 1
fi

ray stop --force || true

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l || true)
if [ "${NVLINK_COUNT:-0}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected ${NVLINK_COUNT:-0} NVLink references)"

MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-/root/miles/scripts/models/qwen3-30B-A3B.sh}"
source "${MODEL_CONFIG_SCRIPT}"
if [ "${MOE_GROUPED_GEMM:-1}" = "0" ]; then
  _MA_FILTERED=()
  for _ma in "${MODEL_ARGS[@]}"; do
    [ "${_ma}" = "--moe-grouped-gemm" ] || _MA_FILTERED+=("${_ma}")
  done
  MODEL_ARGS=("${_MA_FILTERED[@]}")
  echo "[INFO] MOE_GROUPED_GEMM=0 -> stripped --moe-grouped-gemm (SequentialMLP)"
fi

SGLANG_SERVER_FLAGS="$(python3 - <<PY
import argparse
from sglang.srt.server_args import ServerArgs

parser = argparse.ArgumentParser(add_help=False)
ServerArgs.add_cli_args(parser)
flags = sorted(
    {
        flag
        for action in parser._actions
        for flag in action.option_strings
        if flag.startswith("--")
    }
)
print("\n".join(flags))
PY
)"

sglang_flag_supported() {
  local flag="$1"
  grep -Fxq -- "${flag}" <<< "${SGLANG_SERVER_FLAGS}"
}

	CKPT_ARGS=(
	  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
	  --hf-checkpoint "${HF_MODEL_PATH}"
	  --ref-load "${REF_MODEL_PATH}"
	  --save "${SAVE_PATH}"
	  --save-interval "${SAVE_INTERVAL}"
  --override-opt_param-scheduler
	)

	if [ -n "${LOAD_PATH}" ]; then
	  CKPT_ARGS+=(--load "${LOAD_PATH}")
	  if [ "${LOAD_AS_FINETUNE}" = "1" ]; then
	    CKPT_ARGS+=(--no-load-optim --no-load-rng --finetune)
	  fi
	fi

if [ -n "${SAVE_HF_TEMPLATE}" ]; then
  CKPT_ARGS+=(--save-hf "${SAVE_HF_TEMPLATE}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${MILES_TRAIN_FILE}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --prompt-truncation "${PROMPT_TRUNCATION}"
  --rollout-shuffle
  --rm-type "${RM_TYPE}"
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
  --rollout-max-prompt-len "${MAX_PROMPT_LENGTH}"
  --rollout-max-response-len "${MAX_RESPONSE_LENGTH}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-top-p "${ROLLOUT_TOP_P}"
  --rollout-top-k "${ROLLOUT_TOP_K}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

if [ -n "${START_ROLLOUT_ID}" ]; then
  ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi

if [ -z "${REWARD_KEY}" ] && [ "${RM_TYPE}" = "dapo" ]; then
  export REWARD_KEY=score
fi

if [ -n "${REWARD_KEY}" ]; then
  ROLLOUT_ARGS+=(--reward-key "${REWARD_KEY}")
fi

EVAL_ARGS=()
if [ "${EVAL_INTERVAL}" -gt 0 ]; then
  EVAL_ARGS+=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data aime24 "${MILES_EVAL_FILE}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-prompt-len "${MAX_PROMPT_LENGTH}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LENGTH}"
    --eval-temperature "${EVAL_TEMPERATURE}"
    --eval-top-p "${EVAL_TOP_P}"
    --eval-top-k "${EVAL_TOP_K}"
  )
  if [ -n "${REWARD_KEY}" ]; then
    EVAL_ARGS+=(--eval-reward-key "${REWARD_KEY}")
  fi
fi

if [ -f "${SAVE_PATH}/latest_checkpointed_iteration.txt" ]; then
  EVAL_ARGS+=(--skip-eval-before-train)
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
  --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)
if [ "${USE_DYNAMIC_BATCH_SIZE:-1}" = "1" ]; then
  PERF_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
fi

if [ "${ENABLE_TP_COMM_OVERLAP}" = "1" ]; then
  PERF_ARGS+=(--tp-comm-overlap)
fi

if [ "${ENABLE_MOE_DEEPEP}" = "1" ]; then
  PERF_ARGS+=(
    --moe-token-dispatcher-type flex
    --moe-enable-deepep
  )
fi
if [ -n "${MOE_TOKEN_DISPATCHER:-}" ]; then
  PERF_ARGS+=(--moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER}")
fi

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

if [ "${USE_KL_LOSS:-0}" = "1" ]; then
  GRPO_ARGS+=(--use-kl-loss)
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

if [ "${ENABLE_CPU_OPTIMIZER_OFFLOAD}" = "1" ]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi

if [ "${ENABLE_NO_SAVE_OPTIM}" = "1" ]; then
  OPTIMIZER_ARGS+=(--no-save-optim)
fi

ROUTER_LOGITS_ARGS=()
if [ -n "${ROUTER_LOGITS_PATH}" ]; then
  ROUTER_LOGITS_ARGS+=(
    --router-logits-path "${ROUTER_LOGITS_PATH}"
    --router-logits-save-freq "${ROUTER_LOGITS_SAVE_FREQ}"
  )
  if [ -n "${ROUTER_LOGITS_MAX_TOKENS}" ]; then
    ROUTER_LOGITS_ARGS+=(--router-logits-max-tokens "${ROUTER_LOGITS_MAX_TOKENS}")
  fi
fi

build_replay_args REPLAY_ARGS

WANDB_ARGS=()
if [ "${USE_WANDB}" = "1" ]; then
  if [ ! -f "${WANDB_KEY_FILE}" ]; then
    echo "[ERROR] WANDB_KEY_FILE not found: ${WANDB_KEY_FILE}" >&2
    exit 1
  fi
  xtrace_was_on=0
  if [[ "$-" == *x* ]]; then
    xtrace_was_on=1
    set +x
  fi
  export WANDB_API_KEY="$(<"${WANDB_KEY_FILE}")"
  if [ "${xtrace_was_on}" -eq 1 ]; then
    set -x
  fi
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
  )
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 "${SGLANG_CUDA_GRAPH_MAX}")
)

if [ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then
  SGLANG_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
fi

if [ "${ENABLE_SGLANG_EP_MOE}" = "1" ]; then
  if sglang_flag_supported "--enable-ep-moe"; then
    SGLANG_ARGS+=(--sglang-enable-ep-moe)
  else
    echo "[WARN] SGLang flag --sglang-enable-ep-moe is unsupported in this container; falling back to expert parallel size only." >&2
  fi
  if sglang_flag_supported "--ep-size"; then
    SGLANG_ARGS+=(--sglang-ep-size "${SGLANG_EXPERT_PARALLEL_SIZE}")
  else
    echo "[WARN] SGLang flag --sglang-ep-size is unsupported; requested EP size override will be skipped." >&2
  fi
fi

if [ "${ENABLE_SGLANG_DP_ATTENTION}" = "1" ]; then
  if sglang_flag_supported "--enable-dp-attention"; then
    SGLANG_ARGS+=(--sglang-enable-dp-attention)
  else
    echo "[WARN] SGLang flag --sglang-enable-dp-attention is unsupported; requested DP attention optimization will be skipped." >&2
  fi
  if sglang_flag_supported "--dp-size"; then
    SGLANG_ARGS+=(--sglang-dp-size "${SGLANG_DP_SIZE}")
  else
    echo "[WARN] SGLang flag --sglang-dp-size is unsupported; skipping DP size override." >&2
  fi
  if sglang_flag_supported "--moe-dense-tp-size"; then
    SGLANG_ARGS+=(--sglang-moe-dense-tp-size "${SGLANG_MOE_DENSE_TP_SIZE}")
  else
    echo "[WARN] SGLang flag --sglang-moe-dense-tp-size is unsupported; skipping dense TP override." >&2
  fi
  if [ "${ENABLE_SGLANG_DP_LM_HEAD}" = "1" ]; then
    if sglang_flag_supported "--enable-dp-lm-head"; then
      SGLANG_ARGS+=(--sglang-enable-dp-lm-head)
    else
      echo "[WARN] SGLang flag --sglang-enable-dp-lm-head is unsupported; skipping DP LM head optimization." >&2
    fi
  fi
fi

if [ "${ENABLE_SGLANG_DEEPEP_MOE}" = "1" ]; then
  if sglang_flag_supported "--enable-deepep-moe"; then
    SGLANG_ARGS+=(--sglang-enable-deepep-moe)
  else
    echo "[WARN] SGLang flag --sglang-enable-deepep-moe is unsupported; skipping deepep enable flag." >&2
  fi
  if sglang_flag_supported "--deepep-mode"; then
    SGLANG_ARGS+=(--sglang-deepep-mode "${SGLANG_DEEPEP_MODE}")
  else
    echo "[WARN] SGLang flag --sglang-deepep-mode is unsupported; skipping deepep mode override." >&2
  fi
fi

if [ -n "${SGLANG_KV_CACHE_DTYPE}" ]; then
  if sglang_flag_supported "--kv-cache-dtype"; then
    SGLANG_ARGS+=(--sglang-kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}")
  else
    echo "[WARN] SGLang flag --sglang-kv-cache-dtype is unsupported; skipping kv cache dtype override." >&2
  fi
fi

if [ -n "${SGLANG_MOE_A2A_BACKEND}" ]; then
  if sglang_flag_supported "--moe-a2a-backend"; then
    SGLANG_ARGS+=(--sglang-moe-a2a-backend "${SGLANG_MOE_A2A_BACKEND}")
  else
    echo "[WARN] SGLang flag --sglang-moe-a2a-backend is unsupported; skipping MoE A2A backend override." >&2
  fi
fi

if [ -n "${SGLANG_MOE_RUNNER_BACKEND}" ]; then
  if sglang_flag_supported "--moe-runner-backend"; then
    SGLANG_ARGS+=(--sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND}")
  else
    echo "[WARN] SGLang flag --sglang-moe-runner-backend is unsupported; skipping MoE runner backend override." >&2
  fi
fi

if [ "${USE_MILES_ROUTER}" = "1" ]; then
  SGLANG_ARGS+=(--use-miles-router)
fi

if [ "${USE_ROLLOUT_ROUTING_REPLAY}" = "1" ]; then
  if [ "${USE_MILES_ROUTER}" != "1" ]; then
    echo "[ERROR] USE_ROLLOUT_ROUTING_REPLAY=1 requires USE_MILES_ROUTER=1" >&2
    exit 1
  fi
  SGLANG_ARGS+=(--use-rollout-routing-replay)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend "${MEGATRON_ATTENTION_BACKEND:-flash}"
)

export PYTHONBUFFERED=16
export DEPRECATED_MEGATRON_COMPATIBLE=1
export MASTER_ADDR=127.0.0.1
ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${SLURM_GPUS_ON_NODE:-8}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/work1/hwang/dzdong/python_extras:/root/miles:/app/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"DEPRECATED_MEGATRON_COMPATIBLE\": \"1\"
  }
}"

CKPT_MONITOR_PID=""
if [ "${CKPT_KEEP_LATEST}" -ne -1 ]; then
  echo "[INFO] Launching Miles checkpoint monitor in background..."
  mkdir -p "${SAVE_PATH}"
  echo "[INFO] CKPT_MONITOR_LOG=${CKPT_MONITOR_LOG}"
  : > "${CKPT_MONITOR_LOG}"
  CKPT_MONITOR_INTERVAL="${CKPT_MONITOR_INTERVAL}" \
    nohup bash "${CKPT_MONITOR_SCRIPT}" "${CKPT_KEEP_LATEST}" "${SAVE_PATH}" \
    >> "${CKPT_MONITOR_LOG}" 2>&1 < /dev/null &
  CKPT_MONITOR_PID=$!
  echo "[INFO] CKPT_MONITOR_PID=${CKPT_MONITOR_PID}"
fi

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- \
  python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --colocate \
    "${REPLAY_ARGS[@]}" \
    "${ROUTER_LOGITS_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"

job_status=$?
if [ -n "${CKPT_MONITOR_PID}" ]; then
  kill "${CKPT_MONITOR_PID}" 2>/dev/null || true
  wait "${CKPT_MONITOR_PID}" 2>/dev/null || true
  echo "[INFO] Running final checkpoint cleanup scan..."
  bash "${CKPT_MONITOR_SCRIPT}" --once "${CKPT_KEEP_LATEST}" "${SAVE_PATH}" >> "${CKPT_MONITOR_LOG}" 2>&1 || true
fi
ray stop --force || true
exit $job_status
'
job_status=$?
set -e

if [ -n "${AUTO_CKPT_EVAL_HOST_PID}" ]; then
  kill "${AUTO_CKPT_EVAL_HOST_PID}" 2>/dev/null || true
  wait "${AUTO_CKPT_EVAL_HOST_PID}" 2>/dev/null || true
  echo "[INFO] Running final auto checkpoint eval scan on host..."
  bash "${AUTO_CKPT_EVAL_MONITOR_SCRIPT}" --once "${SAVE_PATH}" >> "${AUTO_CKPT_EVAL_LOG}" 2>&1 || true
fi

exit "${job_status}"
