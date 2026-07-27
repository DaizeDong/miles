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

: "${MILES_ROOT:?data preparation requires an explicit frozen MILES_ROOT}"
: "${EXPECTED_CODE_SHA:?data preparation requires EXPECTED_CODE_SHA}"
if [[ ! "${EXPECTED_CODE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "FATAL: EXPECTED_CODE_SHA must be a lowercase 40-character Git SHA" >&2
  exit 97
fi
readonly REPO=$(realpath "${MILES_ROOT}")
if [[ "${REPO}" != /home/daidong/* || ! -e "${REPO}/.git" ]]; then
  echo "FATAL: MILES_ROOT must resolve to a Git worktree below /home/daidong: ${REPO}" >&2
  exit 98
fi
readonly ACTUAL_CODE_SHA=$(git -c safe.directory="${REPO}" -C "${REPO}" rev-parse HEAD)
if [[ "${ACTUAL_CODE_SHA}" != "${EXPECTED_CODE_SHA}" ]]; then
  echo "FATAL: MILES_ROOT HEAD=${ACTUAL_CODE_SHA}, expected=${EXPECTED_CODE_SHA}" >&2
  exit 99
fi
if [[ -n "$(git -c safe.directory="${REPO}" -C "${REPO}" status --porcelain --untracked-files=all)" ]]; then
  echo "FATAL: MILES_ROOT must be clean: ${REPO}" >&2
  exit 100
fi
readonly SCRIPT=${REPO}/experiments/pr2_rebuttal_amd/prepare_rebuttal_data.py
readonly INNER_SCRIPT=${REPO}/experiments/pr2_rebuttal_amd/run_prepare_rebuttal_data.sh
readonly IMAGE=docker.io/rlsys/miles@sha256:e60a69faa831ae2a146819290e8469cf23c81b97febeaf5d11e8baec2bfca285
readonly OUTPUT_ROOT=${PR2_REBUTTAL_DATA_ROOT:-/home/daidong/rebuttal_workspace/data/pr2-rebuttal-one-copy-v1}
readonly JOB_ID=${SPUR_JOB_ID:-${SLURM_JOB_ID:-}}
readonly JOB_NODELIST=${SLURM_JOB_NODELIST:-${SPUR_JOB_NODELIST:-}}
readonly HOST_UID=$(id -u)
readonly HOST_GID=$(id -g)

if [[ -z "${JOB_ID}" ]]; then
  echo "FATAL: this wrapper must run inside a SPUR/Slurm compute allocation" >&2
  exit 90
fi
if [[ -z "${JOB_NODELIST}" ]]; then
  echo "FATAL: scheduler node metadata is missing; refusing login-node execution" >&2
  exit 95
fi
case "${OUTPUT_ROOT}" in
  /home/daidong/*) ;;
  *)
    echo "FATAL: output root must be below /home/daidong: ${OUTPUT_ROOT}" >&2
    exit 96
    ;;
esac
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
echo "compute_host=$(hostname) job_id=${JOB_ID} output_root=${OUTPUT_ROOT} mode=${PR2_REBUTTAL_DATA_MODE:-auto} code_sha=${ACTUAL_CODE_SHA}"
docker pull "${IMAGE}"
docker image inspect "${IMAGE}" --format 'image_id={{.Id}} repo_digests={{json .RepoDigests}}'

readonly CONTAINER_NAME=pr2-data-prep-${JOB_ID}
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --user "${HOST_UID}:${HOST_GID}" \
  --network host \
  --cpus 8 \
  --memory 60g \
  --volume /home/daidong:/home/daidong:rw \
  --workdir "${REPO}" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env HOME=/tmp \
  --env MILES_CONTAINER_IMAGE="${IMAGE}" \
  --env MILES_ROOT="${REPO}" \
  --env EXPECTED_CODE_SHA="${EXPECTED_CODE_SHA}" \
  --env SPUR_JOB_ID="${SPUR_JOB_ID:-}" \
  --env SPUR_JOB_NODELIST="${SPUR_JOB_NODELIST:-}" \
  --env SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
  --env SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}" \
  "${IMAGE}" \
  /bin/bash "${INNER_SCRIPT}" "${OUTPUT_ROOT}" ${MODE_FLAG:+"${MODE_FLAG}"}

if [[ ! -r "${OUTPUT_ROOT}/manifest.json" ]]; then
  echo "FATAL: published data manifest is not readable by the submitting user: ${OUTPUT_ROOT}" >&2
  exit 101
fi
readonly FOREIGN_OWNER=$(find "${OUTPUT_ROOT}" -xdev \( ! -uid "${HOST_UID}" -o ! -gid "${HOST_GID}" \) -print -quit)
if [[ -n "${FOREIGN_OWNER}" ]]; then
  echo "FATAL: published data contains a foreign-owned path: ${FOREIGN_OWNER}" >&2
  exit 102
fi

# The bundle is create-once and all runtime consumers are read-only. Preserve
# existing execute bits while removing write bits from every entry, then prove
# the manifest remains readable from the host account.
find "${OUTPUT_ROOT}" -xdev -type f -exec chmod a-w -- {} +
find "${OUTPUT_ROOT}" -xdev -type d -exec chmod a-w -- {} +
[[ -r "${OUTPUT_ROOT}/manifest.json" ]] || {
  echo "FATAL: frozen data manifest became unreadable: ${OUTPUT_ROOT}" >&2
  exit 103
}
echo "DATA_PREP_OWNERSHIP=PASS uid=${HOST_UID} gid=${HOST_GID} immutable=1"

echo "DATA_PREP_HOST_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "DATA_PREP_WRAPPER_RESULT=PASS"
