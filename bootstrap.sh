#!/usr/bin/env bash
set -euo pipefail

# RunPod bootstrap for an existing PyTorch/CUDA image.  It deliberately keeps
# the application below /app and never writes Factory data to /workspace.
REPO_URL="${REPO_URL:-https://github.com/nayu-822/SDXL-LoRA-Factory.git}"
GIT_BRANCH="${GIT_BRANCH:-feature/runpod-base}"
APP_DIR="${APP_DIR:-/app/SDXL-LoRA-Factory}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GDRIVE_ENABLED="${GDRIVE_ENABLED:-true}"

install_missing_system_tools() {
    local packages=(ca-certificates)
    command -v git >/dev/null 2>&1 || packages+=(git)
    if [[ "${GDRIVE_ENABLED,,}" != "false" ]]; then
        command -v rclone >/dev/null 2>&1 || packages+=(rclone)
    fi
    if command -v git >/dev/null 2>&1 && \
        ([[ "${GDRIVE_ENABLED,,}" == "false" ]] || command -v rclone >/dev/null 2>&1); then
        return
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "[ERROR] Required system tools are missing and apt-get is unavailable." >&2
        exit 1
    fi
    echo "[RunPod bootstrap] Installing missing system tools: ${packages[*]}"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends "${packages[@]}"
    rm -rf /var/lib/apt/lists/*
}

reject_app_path() {
    local candidate
    candidate="$(realpath -m "$APP_DIR")"
    local workspace_root
    workspace_root="$(realpath -m "$WORKSPACE_ROOT")"
    if [[ "$candidate" == "$workspace_root" || "$candidate" == "$workspace_root"/* ]]; then
        echo "[ERROR] APP_DIR must not be inside WORKSPACE_ROOT: $candidate" >&2
        exit 1
    fi
}

install_missing_system_tools
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
    exit 1
}

reject_app_path
mkdir -p /app
if [[ -d "$APP_DIR/.git" ]]; then
    echo "[RunPod bootstrap] Updating $APP_DIR from origin/$GIT_BRANCH"
    git -C "$APP_DIR" fetch --depth 1 origin "$GIT_BRANCH"
    git -C "$APP_DIR" reset --hard "origin/$GIT_BRANCH"
elif [[ -e "$APP_DIR" ]]; then
    echo "[ERROR] APP_DIR exists but is not a Git checkout: $APP_DIR" >&2
    exit 1
else
    echo "[RunPod bootstrap] Cloning $REPO_URL ($GIT_BRANCH) into $APP_DIR"
    git clone --depth 1 --branch "$GIT_BRANCH" "$REPO_URL" "$APP_DIR"
fi

export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8001}"
export WORKSPACE_ROOT
export MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/models/checkpoints}"
export DATA_ROOT="${DATA_ROOT:-/data}"
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-${DATA_ROOT}/dataset}"
export LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-${DATA_ROOT}/output}"
export LOCAL_JOB_ROOT="${LOCAL_JOB_ROOT:-${DATA_ROOT}/jobs}"
export LOCAL_CONFIG_ROOT="${LOCAL_CONFIG_ROOT:-${DATA_ROOT}/config}"
export CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/cache}"
export RCLONE_CONFIG="${RCLONE_CONFIG:-${DATA_ROOT}/rclone/rclone.conf}"
export GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
export GDRIVE_ROOT="${GDRIVE_ROOT:-SDXL-LoRA-Factory/runs}"
export GDRIVE_ENABLED
export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export RUNPOD_PRESERVE_TORCH=1
export AUTO_SETUP=0

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

for generated_path in "$DATA_ROOT" "$LOCAL_DATA_ROOT" "$LOCAL_OUTPUT_ROOT" "$LOCAL_JOB_ROOT" \
    "$LOCAL_CONFIG_ROOT" "$CACHE_DIR" "$(dirname "$RCLONE_CONFIG")"; do
    reject_network_volume_path "$generated_path"
done

mkdir -p "$DATA_ROOT" "$LOCAL_DATA_ROOT" "$LOCAL_OUTPUT_ROOT" "$LOCAL_JOB_ROOT" \
    "$LOCAL_CONFIG_ROOT" "$CACHE_DIR" "$(dirname "$RCLONE_CONFIG")"

echo "[RunPod bootstrap] Verifying the PyTorch/CUDA supplied by the base image"
"$PYTHON_BIN" - <<'PY'
try:
    import torch
except ImportError as error:
    raise SystemExit(f"PyTorch is missing from the selected RunPod image: {error}")
print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
PY

cd "$APP_DIR"
echo "[RunPod bootstrap] Installing Factory dependencies without replacing Torch/CUDA"
"$PYTHON_BIN" -m pip install --disable-pip-version-check --no-cache-dir -r requirements-runpod.txt
"$PYTHON_BIN" -m pip install --disable-pip-version-check --no-cache-dir --no-deps -e backend/sd-scripts

echo "[RunPod bootstrap] Starting SDXL LoRA Factory"
exec "$APP_DIR/start.sh"
