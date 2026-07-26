#!/usr/bin/env bash
#SBATCH --job-name=pr2-data-prep
#SBATCH --account=amd-burst
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Run with:
#   sbatch experiments/pr2_rebuttal_amd/prepare_rebuttal_data.sbatch
# Re-running is safe: a completed bundle is verified offline and never replaced.

set -euo pipefail

readonly REPO=/home/daidong/rebuttal_workspace/miles-pr1362-validation-fixed
readonly SCRIPT=${REPO}/experiments/pr2_rebuttal_amd/prepare_rebuttal_data.py
readonly INNER_SCRIPT=${REPO}/experiments/pr2_rebuttal_amd/run_prepare_rebuttal_data.sh
readonly IMAGE=docker.io/rlsys/miles@sha256:e60a69faa831ae2a146819290e8469cf23c81b97febeaf5d11e8baec2bfca285
readonly OUTPUT_ROOT=${PR2_REBUTTAL_DATA_ROOT:-/home/daidong/rebuttal_workspace/data/pr2-rebuttal-one-copy-v1}
readonly JOB_ID=${SPUR_JOB_ID:-${SLURM_JOB_ID:-}}

if [[ -z "${JOB_ID}" ]]; then
  echo "FATAL: this wrapper must run inside a SPUR/Slurm compute allocation" >&2
  exit 90
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "FATAL: data-preparation program not found: ${SCRIPT}" >&2
  exit 91
fi
if [[ ! -f "${INNER_SCRIPT}" ]]; then
  echo "FATAL: container entry script not found: ${INNER_SCRIPT}" >&2
  exit 94
fi

case "${PR2_REBUTTAL_DATA_MODE:-auto}" in
  auto)
    if [[ -f "${OUTPUT_ROOT}/manifest.json" ]]; then
      readonly MODE_FLAG=--verify-only
    elif [[ -e "${OUTPUT_ROOT}" ]]; then
      echo "FATAL: output exists without a manifest; refusing to replace it: ${OUTPUT_ROOT}" >&2
      exit 92
    else
      readonly MODE_FLAG=
    fi
    ;;
  create)
    readonly MODE_FLAG=
    ;;
  verify)
    readonly MODE_FLAG=--verify-only
    ;;
  *)
    echo "FATAL: PR2_REBUTTAL_DATA_MODE must be auto, create, or verify" >&2
    exit 93
    ;;
esac

echo "DATA_PREP_HOST_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "compute_host=$(hostname) job_id=${JOB_ID} output_root=${OUTPUT_ROOT} mode=${PR2_REBUTTAL_DATA_MODE:-auto}"
docker pull "${IMAGE}"
docker image inspect "${IMAGE}" --format 'image_id={{.Id}} repo_digests={{json .RepoDigests}}'

readonly CONTAINER_NAME=pr2-data-prep-${JOB_ID}
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  --cpus 8 \
  --memory 60g \
  --volume /home/daidong:/home/daidong:rw \
  --workdir "${REPO}" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env MILES_CONTAINER_IMAGE="${IMAGE}" \
  "${IMAGE}" \
  /bin/bash "${INNER_SCRIPT}" "${OUTPUT_ROOT}" ${MODE_FLAG:+"${MODE_FLAG}"}

echo "DATA_PREP_HOST_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "DATA_PREP_WRAPPER_RESULT=PASS"
