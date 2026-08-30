#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"

export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8001}"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
export LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-${WORKSPACE_ROOT}/output}"
export LOCAL_JOB_ROOT="${LOCAL_JOB_ROOT:-${WORKSPACE_ROOT}/jobs}"
export LOCAL_CONFIG_ROOT="${LOCAL_CONFIG_ROOT:-${WORKSPACE_ROOT}/config}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-${WORKSPACE_ROOT}/rclone/rclone.conf}"
export GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ "${AUTO_SETUP:-0}" == "1" ]]; then
    echo "[SDXL LoRA Factory] Running optional dependency setup..."
    "$PYTHON_BIN" backend/setup_check.py
fi

echo "[SDXL LoRA Factory] Listening on ${APP_HOST}:${APP_PORT}"
exec "$PYTHON_BIN" -m backend.main
