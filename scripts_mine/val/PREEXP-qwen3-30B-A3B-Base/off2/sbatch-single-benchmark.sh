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

: "${DATA_ROOT:?DATA_ROOT environment variable must be set}"
: "${RESULTS_ROOT:?RESULTS_ROOT environment variable must be set}"
: "${RUN_NAME:?RUN_NAME environment variable must be set}"
: "${BENCHMARK_NAME:?BENCHMARK_NAME environment variable must be set}"
: "${STEP_LABEL:?STEP_LABEL environment variable must be set}"
: "${MODEL_PATH:?MODEL_PATH environment variable must be set}"

HOST_MILES_ROOT="/mnt/weka/home/hongyi.wang/workspace/rlhf/miles"
CONTAINER_MILES_ROOT="/root/miles"
HOST_VERL_ROOT="/mnt/weka/home/hongyi.wang/workspace/rlhf/verl"
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

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/mnt/weka/shrd/k2m/hongyi.wang/containers/verlai+verl+sgl055.latest.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/mnt/weka/shrd/k2m/haolong.jia:/mnt/weka/shrd/k2m/haolong.jia:rw,/mnt/weka/shrd/k2m/hongyi.wang:/mnt/weka/shrd/k2m/hongyi.wang:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/miles:/root/miles:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/verl:/root/verl:rw,/mnt/weka/home/hongyi.wang/workspace/rlhf/Megatron-LM-verl:/root/out-Megatron-LM:rw}"
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

bash /root/miles/scripts_mine/val/PREEXP-qwen3-30B-A3B-Base/off2/run_eval_one.sh
EOF
)"

srun \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --mem=0 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --export=ALL \
  env DATA_ROOT="${DATA_ROOT_IN_CONTAINER}" RESULTS_ROOT="${RESULTS_ROOT_IN_CONTAINER}" RESULTS_ROOT_HOST="${RESULTS_ROOT_HOST}" AUTO_CKPT_EVAL_EXPORT_ROOT="${EXPORT_ROOT_IN_CONTAINER}" AUTO_CKPT_EVAL_EXPORT_SERIES_NAME="${AUTO_CKPT_EVAL_EXPORT_SERIES_NAME:-}" EVAL_RUNTIME_PYTHONPATH="${EVAL_RUNTIME_PYTHONPATH}" EVAL_RUNTIME_PIP_INSTALL_COMMAND="${EVAL_RUNTIME_PIP_INSTALL_COMMAND}" \
  bash -lc "${CONTAINER_CMD}"
