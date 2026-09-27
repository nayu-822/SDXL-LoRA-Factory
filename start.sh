#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"

export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8001}"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/models/checkpoints}"
export DATA_ROOT="${DATA_ROOT:-/data}"
export DATASET_DIR="${DATASET_DIR:-${LOCAL_DATA_ROOT:-${DATA_ROOT}/dataset}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_OUTPUT_ROOT:-${DATA_ROOT}/output}}"
export LOG_DIR="${LOG_DIR:-${LOCAL_JOB_ROOT:-${DATA_ROOT}/jobs}}"
export CONFIG_DIR="${CONFIG_DIR:-${LOCAL_CONFIG_ROOT:-${DATA_ROOT}/config}}"
export CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/cache}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
# Legacy aliases are still understood by Settings.  Do not override an
# explicitly supplied legacy value, which keeps custom Windows-like Linux
# deployments compatible with the previous RunPod branch.
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-${DATASET_DIR}}"
export LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-${OUTPUT_DIR}}"
export LOCAL_JOB_ROOT="${LOCAL_JOB_ROOT:-${LOG_DIR}}"
export LOCAL_CONFIG_ROOT="${LOCAL_CONFIG_ROOT:-${CONFIG_DIR}}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-${DATA_ROOT}/rclone/rclone.conf}"
export GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
export GDRIVE_ROOT="${GDRIVE_ROOT:-SDXL-LoRA-Factory/runs}"
export GDRIVE_ENABLED="${GDRIVE_ENABLED:-true}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

reject_network_volume_path() {
    local candidate
    candidate="$(realpath -m "$1")"
    local model_root
    model_root="$(realpath -m "$MODEL_DIR")"
    local workspace_root
    workspace_root="$(realpath -m "$WORKSPACE_ROOT")"
    if [[ "$candidate" == "$model_root" || "$candidate" == "$model_root"/* ]]; then
        echo "[ERROR] Factory data path is inside read-only MODEL_DIR: $candidate" >&2
        exit 1
    fi
    if [[ "$candidate" == "$workspace_root" || "$candidate" == "$workspace_root"/* ]]; then
        echo "[ERROR] Factory data must not be written under WORKSPACE_ROOT: $candidate" >&2
        exit 1
    fi
}

for generated_path in "$DATA_ROOT" "$DATASET_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$CONFIG_DIR" "$CACHE_DIR" "$(dirname "$RCLONE_CONFIG")"; do
    reject_network_volume_path "$generated_path"
done

mkdir -p "$DATASET_DIR" "$OUTPUT_DIR" "$LOG_DIR" "$CONFIG_DIR" "$CACHE_DIR"
mkdir -p "$DATA_ROOT/rclone" "$(dirname "$RCLONE_CONFIG")"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[WARN] MODEL_DIR not found: $MODEL_DIR" >&2
    echo "[WARN] /workspace/models/checkpoints が見つかりません。Network Volumeを確認してください。" >&2
else
    model_count="$(find "$MODEL_DIR" -type f \( -iname '*.safetensors' -o -iname '*.ckpt' \) -print -quit 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$model_count" == "0" ]]; then
        echo "[WARN] No .safetensors or .ckpt files found in MODEL_DIR: $MODEL_DIR" >&2
    else
        echo "[SDXL LoRA Factory] Base model directory: $MODEL_DIR"
    fi
fi

if [[ "${AUTO_SETUP:-0}" == "1" ]]; then
    if [[ "${RUNPOD_PRESERVE_TORCH:-0}" == "1" ]]; then
        echo "[SDXL LoRA Factory] AUTO_SETUP skipped because RUNPOD_PRESERVE_TORCH=1."
    else
        echo "[SDXL LoRA Factory] Running optional dependency setup..."
        "$PYTHON_BIN" backend/setup_check.py
    fi
fi

echo "[SDXL LoRA Factory] Listening on ${APP_HOST}:${APP_PORT}"
exec "$PYTHON_BIN" -m backend.main
