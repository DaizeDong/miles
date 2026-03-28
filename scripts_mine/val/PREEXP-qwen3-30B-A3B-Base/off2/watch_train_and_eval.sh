#!/usr/bin/env bash
set -euo pipefail

TRAIN_JOB_ID="${TRAIN_JOB_ID:?TRAIN_JOB_ID must be set}"
RUN_NAME="${RUN_NAME:?RUN_NAME must be set}"
CKPT_ROOT="${CKPT_ROOT:?CKPT_ROOT must be set}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/weka/home/hongyi.wang/workspace/rlhf/miles/results}"
WAIT_INTERVAL="${WAIT_INTERVAL:-120}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "[watcher] waiting for training job ${TRAIN_JOB_ID}"
while squeue -h -j "${TRAIN_JOB_ID}" | grep -q .; do
  echo "[watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) training still running"
  sleep "${WAIT_INTERVAL}"
done

state="$(sacct -j "${TRAIN_JOB_ID}" --format=State --noheader | awk 'NF {print $1; exit}')"
echo "[watcher] training final state=${state}"

model_path="$(
  python3 - "${CKPT_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
hf_root = root / "hf"
if not hf_root.is_dir():
    raise SystemExit(1)

dirs = sorted((p for p in hf_root.glob("rollout_*") if p.is_dir()), key=lambda p: p.name)
if not dirs:
    raise SystemExit(1)

print(dirs[-1])
PY
)" || {
  echo "[watcher] no exported HF checkpoint found under ${CKPT_ROOT}/hf" >&2
  exit 1
}

echo "[watcher] using model_path=${model_path}"
RUN_NAME="${RUN_NAME}" \
CKPT_ROOT="${CKPT_ROOT}" \
MODEL_PATH="${model_path}" \
STEP_LABEL="$(basename "${model_path}")" \
RESULTS_ROOT="${RESULTS_ROOT}" \
WAIT_INTERVAL=30 \
  bash "${SCRIPT_DIR}/run_full_eval.sh"
