#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --account=hwang
#SBATCH --partition=mi3008x
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.log
#SBATCH --error=/work1/hwang/dzdong/miles_logs_tmp/sbatch/%x.%J.%N.%t.err
#SBATCH --open-mode=append

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT environment variable must be set}"
: "${RESULTS_ROOT:?RESULTS_ROOT environment variable must be set}"
: "${RUN_NAME:?RUN_NAME environment variable must be set}"
: "${BENCHMARK_NAME:?BENCHMARK_NAME environment variable must be set}"
: "${STEP_LABEL:?STEP_LABEL environment variable must be set}"
: "${MODEL_PATH:?MODEL_PATH environment variable must be set}"

HOST_MILES_ROOT="/home1/dzdong/workspace/miles"
CONTAINER_MILES_ROOT="/root/miles"
HOST_VERL_ROOT="/work1/hwang/dzdong/verl"
CONTAINER_VERL_ROOT="/root/verl"
DATA_ROOT_IN_CONTAINER="${DATA_ROOT}"
RESULTS_ROOT_HOST="${RESULTS_ROOT}"
RESULTS_ROOT_IN_CONTAINER="${RESULTS_ROOT}"
EXPORT_ROOT_HOST="${AUTO_CKPT_EVAL_EXPORT_ROOT:-}"
EXPORT_ROOT_IN_CONTAINER="${EXPORT_ROOT_HOST}"
if [[ "${DATA_ROOT}" == "${HOST_MILES_ROOT}"* ]]; then
  DATA_ROOT_IN_CONTAINER="${CONTAINER_MILES_ROOT}${DATA_ROOT#${HOST_MILES_ROOT}}"
elif [[ "${DATA_ROOT}" == "${HOST_VERL_ROOT}"* ]]; then
  DATA_ROOT_IN_CONTAINER="${CONTAINER_VERL_ROOT}${DATA_ROOT#${HOST_VERL_ROOT}}"
fi
if [[ "${RESULTS_ROOT}" == "${HOST_MILES_ROOT}"* ]]; then
  RESULTS_ROOT_IN_CONTAINER="${CONTAINER_MILES_ROOT}${RESULTS_ROOT#${HOST_MILES_ROOT}}"
fi
if [[ -n "${EXPORT_ROOT_HOST}" ]]; then
  if [[ "${EXPORT_ROOT_HOST}" == "${HOST_MILES_ROOT}"* ]]; then
    EXPORT_ROOT_IN_CONTAINER="${CONTAINER_MILES_ROOT}${EXPORT_ROOT_HOST#${HOST_MILES_ROOT}}"
  elif [[ "${EXPORT_ROOT_HOST}" == "${HOST_VERL_ROOT}"* ]]; then
    EXPORT_ROOT_IN_CONTAINER="${CONTAINER_VERL_ROOT}${EXPORT_ROOT_HOST#${HOST_VERL_ROOT}}"
  fi
fi

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/work1/hwang/dzdong/images/miles-mi300.sif}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/work1/hwang/dzdong:/work1/hwang/dzdong:rw,/work1/hwang/dzdong:/work1/hwang/dzdong:rw,/home1/dzdong/workspace/miles:/root/miles:rw,/work1/hwang/dzdong/verl:/root/verl:rw,/work1/hwang/dzdong/Megatron-LM-verl:/root/out-Megatron-LM:rw}"
EVAL_RUNTIME_PYTHONPATH="${EVAL_RUNTIME_PYTHONPATH:-${RUNTIME_PYTHONPATH:-}}"
EVAL_RUNTIME_PIP_INSTALL_COMMAND="${EVAL_RUNTIME_PIP_INSTALL_COMMAND:-${RUNTIME_PIP_INSTALL_COMMAND:-}}"

echo "================================================================"
echo "Starting evaluation"
echo "  Benchmark: ${BENCHMARK_NAME}"
echo "  Run name: ${RUN_NAME}"
echo "  Step label: ${STEP_LABEL}"
echo "  Model path: ${MODEL_PATH}"
echo "  Data root: ${DATA_ROOT}"
echo "  Data root in container: ${DATA_ROOT_IN_CONTAINER}"
echo "  Results root: ${RESULTS_ROOT}"
echo "  Results root in container: ${RESULTS_ROOT_IN_CONTAINER}"
echo "  Results root on host: ${RESULTS_ROOT_HOST}"
echo "  Export root on host: ${EXPORT_ROOT_HOST:-disabled}"
echo "  Export root in container: ${EXPORT_ROOT_IN_CONTAINER:-disabled}"
echo "  Export series name: ${AUTO_CKPT_EVAL_EXPORT_SERIES_NAME:-disabled}"
echo "  Node: $(hostname -s)"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "================================================================"

CONTAINER_CMD="$(cat <<'EOF'
set -euo pipefail

rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

MISSING_PKGS_FILE="/tmp/miles_eval_missing_pkgs.txt"
PIP_LOG_FILE="/tmp/miles_eval_pip.log"

python3 - <<'PY' > "${MISSING_PKGS_FILE}"
import importlib.util

required_packages = (
    ("math_verify", "math-verify"),
    ("tensordict", "tensordict>=0.8.0,<=0.10.0,!=0.9.0"),
    ("torchdata", "torchdata"),
    ("peft", "peft"),
    ("codetiming", "codetiming"),
)

for module_name, package_spec in required_packages:
    if importlib.util.find_spec(module_name) is None:
        print(package_spec)
PY

mapfile -t missing_pkgs < "${MISSING_PKGS_FILE}"
if [ "${#missing_pkgs[@]}" -gt 0 ]; then
  echo "[INFO] installing missing eval dependencies: ${missing_pkgs[*]}"
  python3 -m pip install --disable-pip-version-check "${missing_pkgs[@]}" > "${PIP_LOG_FILE}" 2>&1 || {
    cat "${PIP_LOG_FILE}" >&2
    exit 1
  }
fi

if [ -n "${EVAL_RUNTIME_PIP_INSTALL_COMMAND:-}" ]; then
  echo "[INFO] running eval runtime pip install command"
  bash -lc "${EVAL_RUNTIME_PIP_INSTALL_COMMAND}"
fi

bash /root/miles/scripts_my_hpcfund/val/PREEXP-qwen3-30B-A3B-Base/off2/run_eval_one.sh
EOF
)"

srun \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --mem=0 \
  --export=ALL \
  env DATA_ROOT="${DATA_ROOT_IN_CONTAINER}" RESULTS_ROOT="${RESULTS_ROOT_IN_CONTAINER}" RESULTS_ROOT_HOST="${RESULTS_ROOT_HOST}" AUTO_CKPT_EVAL_EXPORT_ROOT="${EXPORT_ROOT_IN_CONTAINER}" AUTO_CKPT_EVAL_EXPORT_SERIES_NAME="${AUTO_CKPT_EVAL_EXPORT_SERIES_NAME:-}" EVAL_RUNTIME_PYTHONPATH="${EVAL_RUNTIME_PYTHONPATH}" EVAL_RUNTIME_PIP_INSTALL_COMMAND="${EVAL_RUNTIME_PIP_INSTALL_COMMAND}" \
  bash -lc "${CONTAINER_CMD}"
