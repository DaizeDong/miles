#!/usr/bin/env bash

set -euo pipefail

readonly OUTPUT_ROOT=${1:?usage: run_prepare_rebuttal_data.sh OUTPUT_ROOT [--verify-only]}
shift
readonly SCRIPT=experiments/pr2_rebuttal_amd/prepare_rebuttal_data.py
readonly TEST=experiments/pr2_rebuttal_amd/test_prepare_rebuttal_data.py

/opt/venv/bin/python -m pytest \
  --confcutdir=experiments/pr2_rebuttal_amd \
  -q "${TEST}"

exec /opt/venv/bin/python "${SCRIPT}" --output-root "${OUTPUT_ROOT}" "$@"
