#!/bin/bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--once] KEEP_LATEST DIR [DIR ...]" >&2
}

ONCE=0
if [ "${1:-}" = "--once" ]; then
  ONCE=1
  shift
fi

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

KEEP_LATEST="$1"
shift
MONITOR_DIRS=("$@")
SLEEP_INTERVAL="${CKPT_MONITOR_INTERVAL:-120}"
IDLE_INTERVAL="${CKPT_MONITOR_IDLE_INTERVAL:-30}"

if ! [[ "${KEEP_LATEST}" =~ ^-?[0-9]+$ ]]; then
  echo "KEEP_LATEST must be an integer, got: ${KEEP_LATEST}" >&2
  exit 1
fi

if [ "${KEEP_LATEST}" -eq -1 ]; then
  echo "KEEP_LATEST is -1 (infinite), exiting ckpt monitor."
  exit 0
fi

shopt -s nullglob

scan_once() {
  local monitor_dir
  local deleted_any=0

  for monitor_dir in "${MONITOR_DIRS[@]}"; do
    local iter_dirs=()
    local sorted_dirs=()
    local completed_dirs=()
    local target_dir
    local tracker_file
    local tracker_iter=""
    local num_to_delete=0
    local deleted_in_dir=0

    if [ ! -d "${monitor_dir}" ]; then
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] - Monitor dir does not exist yet: ${monitor_dir}"
      continue
    fi

    iter_dirs=("${monitor_dir}"/iter_[0-9]*)
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] - Found ${#iter_dirs[@]} iter dirs in: ${monitor_dir}"

    if [ "${#iter_dirs[@]}" -eq 0 ]; then
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] - No iter dirs to delete for: ${monitor_dir}"
      continue
    fi

    tracker_file="${monitor_dir}/latest_checkpointed_iteration.txt"
    if [ -f "${tracker_file}" ]; then
      tracker_iter="$(tr -d '[:space:]' < "${tracker_file}")"
      if ! [[ "${tracker_iter}" =~ ^[0-9]+$ ]]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] - Ignoring invalid tracker value in ${tracker_file}: ${tracker_iter}"
        tracker_iter=""
      fi
    fi

    mapfile -t sorted_dirs < <(printf "%s\n" "${iter_dirs[@]}" | sort)

    if [ -n "${tracker_iter}" ]; then
      local iter_name
      local iter_id
      for target_dir in "${sorted_dirs[@]}"; do
        iter_name="$(basename "${target_dir}")"
        iter_id="${iter_name#iter_}"
        if [[ "${iter_id}" =~ ^[0-9]+$ ]] && [ $((10#${iter_id})) -le $((10#${tracker_iter})) ]; then
          completed_dirs+=("${target_dir}")
        fi
      done
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] - Tracker latest completed iter: ${tracker_iter} in ${monitor_dir}"
    else
      completed_dirs=("${sorted_dirs[@]}")
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] - No valid tracker found in ${monitor_dir}; cleaning by directory order only."
    fi

    if [ "${#completed_dirs[@]}" -le "${KEEP_LATEST}" ]; then
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] - No completed iter dirs to delete for: ${monitor_dir}"
      continue
    fi

    num_to_delete=$((${#completed_dirs[@]} - KEEP_LATEST))
    for ((i = 0; i < num_to_delete; i++)); do
      target_dir="${completed_dirs[$i]}"
      if [ -d "${target_dir}" ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] - Deleting old iter dir: ${target_dir}"
        rm -rf -- "${target_dir}"
        deleted_in_dir=1
        deleted_any=1
      fi
    done

    if [ "${deleted_in_dir}" -eq 1 ]; then
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] - Deletion done for: ${monitor_dir}"
    else
      echo "[$(date +"%Y-%m-%d %H:%M:%S")] - Nothing deleted for: ${monitor_dir}"
    fi
  done

  if [ "${deleted_any}" -eq 1 ]; then
    return 0
  fi
  return 1
}

if [ "${ONCE}" = "1" ]; then
  scan_once || true
  exit 0
fi

while true; do
  if scan_once; then
    sleep "${SLEEP_INTERVAL}"
  else
    sleep "${IDLE_INTERVAL}"
  fi
done
