#!/usr/bin/env bash

set -euo pipefail

readonly OUTPUT_ROOT=${1:?usage: run_prepare_rebuttal_data.sh OUTPUT_ROOT [--verify-only]}
shift
readonly SCRIPT=experiments/pr2_rebuttal_amd/prepare_rebuttal_data.py
readonly TEST=experiments/pr2_rebuttal_amd/test_prepare_rebuttal_data.py
readonly STDLIB_TEST=experiments.pr2_rebuttal_amd.test_prepare_rebuttal_data_stdlib
readonly MATH_REWARD_TEST=experiments.pr2_rebuttal_amd.test_math_reward_stdlib
readonly EVAL_METRICS_TEST=experiments.pr2_rebuttal_amd.test_eval_metrics_stdlib

case "${OUTPUT_ROOT}" in
  /home/daidong/*) ;;
  *)
    echo "FATAL: output root must be below /home/daidong: ${OUTPUT_ROOT}" >&2
    exit 96
    ;;
esac

readonly JOB_ID=${SPUR_JOB_ID:-${SLURM_JOB_ID:-}}
readonly JOB_NODELIST=${SLURM_JOB_NODELIST:-${SPUR_JOB_NODELIST:-}}
if [[ -z "${JOB_ID}" || -z "${JOB_NODELIST}" ]]; then
  echo "FATAL: create and verify modes require a SPUR/Slurm compute allocation" >&2
  exit 90
fi

/opt/venv/bin/python -m pytest \
  --confcutdir=experiments/pr2_rebuttal_amd \
  -q "${TEST}"

/opt/venv/bin/python -m unittest -v \
  "${STDLIB_TEST}" \
  "${MATH_REWARD_TEST}" \
  "${EVAL_METRICS_TEST}"

/opt/venv/bin/python - <<'PY'
from types import SimpleNamespace

from experiments.pr2_rebuttal_amd.math_reward import score_sample

sample = SimpleNamespace(response=r"\boxed{7}", label="7", metadata={"rm_type": "math"})
score = score_sample(sample)
if score != 1.0:
    raise SystemExit(f"REAL_MATH_REWARD_SMOKE=FAIL score={score!r}")
print("REAL_MATH_REWARD_SMOKE=PASS score=1.0")
PY

exec /opt/venv/bin/python "${SCRIPT}" --output-root "${OUTPUT_ROOT}" "$@"
