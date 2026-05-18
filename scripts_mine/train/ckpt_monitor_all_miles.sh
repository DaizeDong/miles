#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: ckpt_monitor_all_miles.sh [--once] [checkpoint_root]
EOF
}

ONCE=0
if [ "${1:-}" = "--once" ]; then
  ONCE=1
  shift
fi

if [ "$#" -gt 1 ]; then
  usage
  exit 2
fi

CHECKPOINT_ROOT="${1:-/mnt/weka/shrd/k2m/hongyi.wang/checkpoint_miles}"
KEEP_LATEST="${CKPT_KEEP_LATEST:-1}"
POLL_INTERVAL="${CKPT_MONITOR_ALL_INTERVAL:-120}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHILD_SCRIPT="${SCRIPT_DIR}/ckpt_monitor_miles.sh"

if [ ! -f "${CHILD_SCRIPT}" ]; then
  echo "[ckpt-monitor-all] child monitor script not found: ${CHILD_SCRIPT}" >&2
  exit 1
fi

if [ ! -d "${CHECKPOINT_ROOT}" ]; then
  echo "[ckpt-monitor-all] checkpoint root not found: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

scan_once() {
  local monitor_dirs=()

  mapfile -t monitor_dirs < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
  if [ "${#monitor_dirs[@]}" -eq 0 ]; then
    echo "[ckpt-monitor-all] no checkpoint directories found under ${CHECKPOINT_ROOT}"
    return 0
  fi

  echo "[ckpt-monitor-all] scanning ${#monitor_dirs[@]} checkpoint roots under ${CHECKPOINT_ROOT}"
  bash "${CHILD_SCRIPT}" --once "${KEEP_LATEST}" "${monitor_dirs[@]}"
}

echo "[ckpt-monitor-all] CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "[ckpt-monitor-all] KEEP_LATEST=${KEEP_LATEST}"
echo "[ckpt-monitor-all] POLL_INTERVAL=${POLL_INTERVAL}"

if [ "${ONCE}" = "1" ]; then
  scan_once
  exit 0
fi

while true; do
  scan_once || true
  sleep "${POLL_INTERVAL}"
done
