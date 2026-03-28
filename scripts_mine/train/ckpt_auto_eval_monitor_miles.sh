#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: ckpt_auto_eval_monitor_miles.sh [--once] <ckpt_root>
EOF
}

ONCE=0
if [ "${1:-}" = "--once" ]; then
  ONCE=1
  shift
fi

if [ "$#" -ne 1 ]; then
  usage
  exit 2
fi

CKPT_ROOT="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${MILES_REPO_ROOT:-}" ]; then
  REPO_ROOT="${MILES_REPO_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/scripts_mine" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi

resolve_repo_path() {
  local path="$1"
  case "$path" in
    /*) printf '%s\n' "$path" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "${path#./}" ;;
  esac
}

AUTO_CKPT_EVAL_RUNNER_SCRIPT="$(resolve_repo_path "${AUTO_CKPT_EVAL_RUNNER_SCRIPT:-scripts_mine/train/ckpt_auto_eval_runner_miles.sh}")"
AUTO_CKPT_EVAL_EXPORT_HF_SCRIPT="$(resolve_repo_path "${AUTO_CKPT_EVAL_EXPORT_HF_SCRIPT:-scripts_mine/val/PREEXP-qwen3-30B-A3B-Base/off2/sbatch-export-hf.sh}")"
AUTO_CKPT_EVAL_RUN_NAME="${AUTO_CKPT_EVAL_RUN_NAME:-eval}"
AUTO_CKPT_EVAL_RESULTS_ROOT="${AUTO_CKPT_EVAL_RESULTS_ROOT:-${CKPT_ROOT}}"
AUTO_CKPT_EVAL_POLL_INTERVAL="${AUTO_CKPT_EVAL_POLL_INTERVAL:-120}"
AUTO_CKPT_EVAL_READY_GRACE_SEC="${AUTO_CKPT_EVAL_READY_GRACE_SEC:-30}"
AUTO_CKPT_EVAL_WAIT_INTERVAL="${AUTO_CKPT_EVAL_WAIT_INTERVAL:-30}"
AUTO_CKPT_EVAL_BENCHMARKS="${AUTO_CKPT_EVAL_BENCHMARKS:-}"
AUTO_CKPT_EVAL_STATE_ROOT="${AUTO_CKPT_EVAL_STATE_ROOT:-${CKPT_ROOT}/auto_ckpt_eval}"
AUTO_CKPT_EVAL_DONE_DIR="${AUTO_CKPT_EVAL_DONE_DIR:-${AUTO_CKPT_EVAL_STATE_ROOT}/done}"
AUTO_CKPT_EVAL_FAILED_DIR="${AUTO_CKPT_EVAL_FAILED_DIR:-${AUTO_CKPT_EVAL_STATE_ROOT}/failed}"
AUTO_CKPT_EVAL_SUBMITTED_DIR="${AUTO_CKPT_EVAL_SUBMITTED_DIR:-${AUTO_CKPT_EVAL_STATE_ROOT}/submitted}"
AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR="${AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR:-${AUTO_CKPT_EVAL_STATE_ROOT}/export_submitted}"
AUTO_CKPT_EVAL_EXPORT_FAILED_DIR="${AUTO_CKPT_EVAL_EXPORT_FAILED_DIR:-${AUTO_CKPT_EVAL_STATE_ROOT}/export_failed}"
if [ -z "${SAVE_HF_TEMPLATE:-}" ]; then
  SAVE_HF_TEMPLATE="$(printf '%s/hf/rollout_{rollout_id:04d}' "${CKPT_ROOT}")"
fi
HF_ROOT="${CKPT_ROOT}/hf"

mkdir -p \
  "${AUTO_CKPT_EVAL_DONE_DIR}" \
  "${AUTO_CKPT_EVAL_FAILED_DIR}" \
  "${AUTO_CKPT_EVAL_SUBMITTED_DIR}" \
  "${AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR}" \
  "${AUTO_CKPT_EVAL_EXPORT_FAILED_DIR}"

if [ ! -f "${AUTO_CKPT_EVAL_RUNNER_SCRIPT}" ]; then
  echo "[auto-eval-monitor] runner script not found: ${AUTO_CKPT_EVAL_RUNNER_SCRIPT}" >&2
  exit 1
fi
if [ ! -f "${AUTO_CKPT_EVAL_EXPORT_HF_SCRIPT}" ]; then
  echo "[auto-eval-monitor] export script not found: ${AUTO_CKPT_EVAL_EXPORT_HF_SCRIPT}" >&2
  exit 1
fi

checkpoint_ready() {
  local model_dir="$1"
  local grace_sec="$2"

  python3 - "${model_dir}" "${grace_sec}" <<'PY'
import json
import sys
import time
from pathlib import Path

model_dir = Path(sys.argv[1]).resolve()
grace_sec = int(float(sys.argv[2]))
if not model_dir.is_dir():
    raise SystemExit(1)

required_files = [model_dir / "config.json"]
tokenizer_candidates = (
    model_dir / "tokenizer.json",
    model_dir / "tokenizer.model",
    model_dir / "vocab.json",
)
if not any(path.exists() for path in tokenizer_candidates):
    raise SystemExit(1)

single_weight = model_dir / "model.safetensors"
index_file = model_dir / "model.safetensors.index.json"
weight_files = []
if single_weight.exists():
    weight_files = [single_weight]
elif index_file.exists():
    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    mapped_files = sorted(
        {
            value
            for value in index_data.get("weight_map", {}).values()
            if isinstance(value, str) and value
        }
    )
    if not mapped_files:
        raise SystemExit(1)
    weight_files = [model_dir / value for value in mapped_files]
    required_files.append(index_file)
else:
    raise SystemExit(1)

required_files.extend(weight_files)
for path in required_files:
    if not path.exists():
        raise SystemExit(1)

latest_mtime = max(path.stat().st_mtime for path in required_files)
if time.time() - latest_mtime < grace_sec:
    raise SystemExit(1)
PY
}

step_label_to_rollout_id() {
  local step_label="$1"
  if [[ "${step_label}" =~ ^rollout_([0-9]+)$ ]]; then
    printf '%d\n' "$((10#${BASH_REMATCH[1]}))"
    return 0
  fi
  return 1
}

iter_dir_for_step_label() {
  local step_label="$1"
  local rollout_id
  rollout_id="$(step_label_to_rollout_id "${step_label}")" || return 1
  printf '%s/iter_%07d\n' "${CKPT_ROOT}" "${rollout_id}"
}

submit_eval() {
  local model_dir="$1"
  local step_label="$2"
  local submitted_file="${AUTO_CKPT_EVAL_SUBMITTED_DIR}/${step_label}.submitted"
  local job_output
  local job_id

  if ! job_output="$(
    RUN_NAME="${AUTO_CKPT_EVAL_RUN_NAME}" \
    CKPT_ROOT="${CKPT_ROOT}" \
    MODEL_PATH="${model_dir}" \
    STEP_LABEL="${step_label}" \
    RESULTS_ROOT="${AUTO_CKPT_EVAL_RESULTS_ROOT}" \
    AUTO_CKPT_EVAL_DONE_DIR="${AUTO_CKPT_EVAL_DONE_DIR}" \
    AUTO_CKPT_EVAL_FAILED_DIR="${AUTO_CKPT_EVAL_FAILED_DIR}" \
    AUTO_CKPT_EVAL_SUBMITTED_DIR="${AUTO_CKPT_EVAL_SUBMITTED_DIR}" \
    WAIT_INTERVAL="${AUTO_CKPT_EVAL_WAIT_INTERVAL}" \
    BENCHMARKS_CSV="${AUTO_CKPT_EVAL_BENCHMARKS}" \
    sbatch \
      --job-name="auto-eval-${AUTO_CKPT_EVAL_RUN_NAME}-${step_label}" \
      --export=ALL \
      "${AUTO_CKPT_EVAL_RUNNER_SCRIPT}"
  )"; then
    echo "[auto-eval-monitor] failed to submit ${step_label}" >&2
    return 1
  fi

  job_id="$(echo "${job_output}" | grep -oE '[0-9]+' | tail -n 1 || true)"
  printf 'job_id=%s submitted_at=%s\n' "${job_id:-unknown}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${submitted_file}"
  echo "[auto-eval-monitor] submitted ${step_label}: ${job_output}"
}

submit_export() {
  local step_label="$1"
  local iter_dir="$2"
  local submitted_file="${AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR}/${step_label}.submitted"
  local job_output
  local job_id

  if ! job_output="$(
    CKPT_ROOT="${CKPT_ROOT}" \
    EXPORT_LOAD_PATH="${iter_dir}" \
    MEGATRON_LOAD_PATH="${CKPT_ROOT}" \
    SAVE_HF_TEMPLATE="${SAVE_HF_TEMPLATE}" \
    EXPORT_STEP_LABEL="${step_label}" \
    AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR="${AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR}" \
    AUTO_CKPT_EVAL_EXPORT_FAILED_DIR="${AUTO_CKPT_EVAL_EXPORT_FAILED_DIR}" \
    sbatch \
      --job-name="auto-export-${AUTO_CKPT_EVAL_RUN_NAME}-${step_label}" \
      --export=ALL \
      "${AUTO_CKPT_EVAL_EXPORT_HF_SCRIPT}"
  )"; then
    echo "[auto-eval-monitor] failed to submit HF export for ${step_label}" >&2
    return 1
  fi

  job_id="$(echo "${job_output}" | grep -oE '[0-9]+' | tail -n 1 || true)"
  printf 'job_id=%s submitted_at=%s iter_dir=%s\n' \
    "${job_id:-unknown}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${iter_dir}" > "${submitted_file}"
  echo "[auto-eval-monitor] submitted HF export for ${step_label}: ${job_output}"
}

scan_once() {
  local model_dir
  local step_label
  local done_file
  local failed_file
  local submitted_file
  local export_submitted_file
  local export_failed_file
  local iter_dir

  if [ ! -d "${HF_ROOT}" ]; then
    echo "[auto-eval-monitor] waiting for HF export root: ${HF_ROOT}"
    return 0
  fi

  mapfile -t model_dirs < <(find "${HF_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'rollout_*' | sort)
  for model_dir in "${model_dirs[@]}"; do
    step_label="$(basename "${model_dir}")"
    done_file="${AUTO_CKPT_EVAL_DONE_DIR}/${step_label}.done"
    failed_file="${AUTO_CKPT_EVAL_FAILED_DIR}/${step_label}.failed"
    submitted_file="${AUTO_CKPT_EVAL_SUBMITTED_DIR}/${step_label}.submitted"
    export_submitted_file="${AUTO_CKPT_EVAL_EXPORT_SUBMITTED_DIR}/${step_label}.submitted"
    export_failed_file="${AUTO_CKPT_EVAL_EXPORT_FAILED_DIR}/${step_label}.failed"

    if [ -f "${done_file}" ] || [ -f "${failed_file}" ] || [ -f "${submitted_file}" ]; then
      continue
    fi

    if checkpoint_ready "${model_dir}" "${AUTO_CKPT_EVAL_READY_GRACE_SEC}"; then
      submit_eval "${model_dir}" "${step_label}" || true
      continue
    fi

    if [ -f "${export_submitted_file}" ] || [ -f "${export_failed_file}" ]; then
      if [ "${ONCE}" = "1" ]; then
        if [ -f "${export_submitted_file}" ]; then
          echo "[auto-eval-monitor] HF export already submitted: ${step_label}"
        else
          echo "[auto-eval-monitor] HF export previously failed: ${step_label}"
        fi
      fi
      continue
    fi

    iter_dir="$(iter_dir_for_step_label "${step_label}" || true)"
    if [ -n "${iter_dir}" ] && [ -d "${iter_dir}" ]; then
      submit_export "${step_label}" "${iter_dir}" || true
      continue
    fi

    if [ "${ONCE}" = "1" ]; then
      echo "[auto-eval-monitor] checkpoint not ready yet: ${step_label}"
    fi
  done
}

echo "[auto-eval-monitor] CKPT_ROOT=${CKPT_ROOT}"
echo "[auto-eval-monitor] HF_ROOT=${HF_ROOT}"
echo "[auto-eval-monitor] RUN_NAME=${AUTO_CKPT_EVAL_RUN_NAME}"
echo "[auto-eval-monitor] RESULTS_ROOT=${AUTO_CKPT_EVAL_RESULTS_ROOT}"
echo "[auto-eval-monitor] STATE_ROOT=${AUTO_CKPT_EVAL_STATE_ROOT}"
echo "[auto-eval-monitor] RUNNER_SCRIPT=${AUTO_CKPT_EVAL_RUNNER_SCRIPT}"
echo "[auto-eval-monitor] EXPORT_HF_SCRIPT=${AUTO_CKPT_EVAL_EXPORT_HF_SCRIPT}"
echo "[auto-eval-monitor] POLL_INTERVAL=${AUTO_CKPT_EVAL_POLL_INTERVAL}"
echo "[auto-eval-monitor] READY_GRACE_SEC=${AUTO_CKPT_EVAL_READY_GRACE_SEC}"

if [ "${ONCE}" = "1" ]; then
  scan_once
  exit 0
fi

while true; do
  scan_once
  sleep "${AUTO_CKPT_EVAL_POLL_INTERVAL}"
done
