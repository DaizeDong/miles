#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/results}"

RUN_NAME="${RUN_NAME:-}"
CKPT_ROOT="${CKPT_ROOT:-}"
MODEL_PATH="${MODEL_PATH:-}"
STEP_LABEL="${STEP_LABEL:-}"
WAIT_INTERVAL="${WAIT_INTERVAL:-30}"
BENCHMARKS_CSV="${BENCHMARKS_CSV:-}"

if [ -z "${RUN_NAME}" ]; then
  echo "[ERROR] RUN_NAME must be set." >&2
  exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
  if [ -z "${CKPT_ROOT}" ]; then
    echo "[ERROR] Either MODEL_PATH or CKPT_ROOT must be set." >&2
    exit 1
  fi
  MODEL_PATH="$(
    python3 - "${CKPT_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
hf_root = root / "hf"
if not hf_root.is_dir():
    raise SystemExit(f"[ERROR] hf export directory not found: {hf_root}")

dirs = sorted((p for p in hf_root.glob("rollout_*") if p.is_dir()), key=lambda p: p.name)
if not dirs:
    raise SystemExit(f"[ERROR] no hf checkpoints found under {hf_root}")

print(dirs[-1])
PY
  )"
fi

if [ -z "${STEP_LABEL}" ]; then
  STEP_LABEL="$(basename "${MODEL_PATH}")"
fi

declare -A DATA_ROOTS=(
  [math500]="/root/verl/data/eval_math500"
  [math500_to_dapo]="/root/verl/data/eval_math500_to_dapo"
  [aime2024]="/root/verl/data/eval_aime2024"
  [aime2025]="/root/verl/data/eval_aime2025"
  [amc2023]="/root/verl/data/eval_amc2023"
  [hmmt2025]="/root/verl/data/eval_hmmt2025"
)

if [ -n "${BENCHMARKS_CSV}" ]; then
  IFS=',' read -r -a BENCHMARKS <<< "${BENCHMARKS_CSV}"
else
  BENCHMARKS=(math500 math500_to_dapo aime2024 aime2025 amc2023 hmmt2025)
fi

mkdir -p "${RESULTS_ROOT}/${RUN_NAME}"

declare -a JOB_IDS=()
for benchmark in "${BENCHMARKS[@]}"; do
  if [ -z "${benchmark}" ]; then
    continue
  fi
  if [ -z "${DATA_ROOTS[$benchmark]+x}" ]; then
    echo "[ERROR] unsupported benchmark: ${benchmark}" >&2
    exit 1
  fi
  data_root="${DATA_ROOTS[$benchmark]}"
  job_output=$(
    DATA_ROOT="${data_root}" \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    RUN_NAME="${RUN_NAME}" \
    BENCHMARK_NAME="${benchmark}" \
    STEP_LABEL="${STEP_LABEL}" \
    MODEL_PATH="${MODEL_PATH}" \
    sbatch \
      --job-name="eval-${RUN_NAME}-${benchmark}" \
      --export=ALL \
      "${SCRIPT_DIR}/sbatch-single-benchmark.sh"
  )
  job_id="$(echo "${job_output}" | grep -oE '[0-9]+' | tail -n 1)"
  JOB_IDS+=("${job_id}")
  echo "[INFO] submitted ${benchmark}: ${job_id}"
done

while true; do
  running=0
  for job_id in "${JOB_IDS[@]}"; do
    if squeue -h -j "${job_id}" >/dev/null 2>&1 && [ -n "$(squeue -h -j "${job_id}")" ]; then
      running=$((running + 1))
    fi
  done
  if [ "${running}" -eq 0 ]; then
    break
  fi
  echo "[INFO] waiting for ${running}/${#JOB_IDS[@]} eval jobs..."
  sleep "${WAIT_INTERVAL}"
done

failed=0
for job_id in "${JOB_IDS[@]}"; do
  state="$(sacct -j "${job_id}" --format=State --noheader | awk 'NF {print $1; exit}')"
  echo "[INFO] job ${job_id} state=${state}"
  case "${state}" in
    COMPLETED|COMPLETED+) ;;
    *)
      failed=1
      ;;
  esac
done

python3 "${SCRIPT_DIR}/summarize_results.py" "${RESULTS_ROOT}" "${RUN_NAME}"

if [ "${failed}" -ne 0 ]; then
  echo "[ERROR] one or more eval jobs failed." >&2
  exit 1
fi

echo "[INFO] all eval jobs completed successfully."
