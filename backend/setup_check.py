"""Optional dependency/setup check for Windows and RunPod/Linux."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
SD_SCRIPTS_ROOT = BACKEND_ROOT / "sd-scripts"


def run_command(command: list[str], cwd: Path | None = None) -> bool:
    print("[SETUP]", " ".join(command))
    try:
        result = subprocess.run(command, cwd=str(cwd) if cwd else None, check=False)
        return result.returncode == 0
    except OSError as error:
        print(f"[ERROR] Could not execute command: {error}")
        return False


def check_sd_scripts() -> bool:
    key_file = SD_SCRIPTS_ROOT / "sdxl_train_network.py"
    if not key_file.is_file():
        print(f"[ERROR] Bundled sd-scripts entrypoint not found: {key_file}")
        return False
    print(f"[INFO] Bundled sd-scripts found: {SD_SCRIPTS_ROOT}")
    return True


def nvidia_gpu_name() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def pytorch_status() -> str:
    code = """
import sys
try:
    import torch
    import torchvision
    if not torch.cuda.is_available():
        print('NO_CUDA')
        sys.exit(0)
    major, minor = torch.cuda.get_device_capability()
    arch_list = torch.cuda.get_arch_list()
    if major * 10 + minor >= 120 and 'sm_120' not in arch_list:
        print('NEEDS_UPGRADE')
    else:
        print('OK')
except ImportError:
    print('MISSING')
"""
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return "MISSING"
        return result.stdout.strip().splitlines()[-1]
    except (OSError, IndexError):
        return "MISSING"


def check_pytorch() -> bool:
    gpu = nvidia_gpu_name()
    status = pytorch_status()
    is_nvidia = gpu is not None
    is_blackwell = bool(gpu and "RTX 50" in gpu)
    if status == "OK":
        print(f"[INFO] PyTorch is ready (GPU: {gpu or 'CPU'}).")
        return True
    if status == "NO_CUDA" and is_nvidia:
        print("[SETUP] NVIDIA GPU detected but installed PyTorch has no CUDA runtime; installing a CUDA build.")
    elif status == "NEEDS_UPGRADE" and is_blackwell:
        print("[SETUP] RTX 50 series detected; installing the PyTorch nightly CUDA build.")
    elif status == "MISSING":
        print("[SETUP] PyTorch is not installed; installing the appropriate build.")
    else:
        print(f"[WARN] PyTorch check returned {status}; installing the default build.")

    if is_blackwell:
        return run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--pre",
                "--upgrade",
                "torch",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/nightly/cu130",
            ]
        )
    if is_nvidia:
        return run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "torch",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )
    return run_command([sys.executable, "-m", "pip", "install", "torch", "torchvision"])


def check_requirements() -> bool:
    requirements = BACKEND_ROOT / "requirements.txt"
    sd_requirements = SD_SCRIPTS_ROOT / "requirements.txt"
    ok = True
    if requirements.is_file():
        ok = run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements)]) and ok
    if sd_requirements.is_file():
        ok = run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(sd_requirements)], cwd=SD_SCRIPTS_ROOT
        ) and ok
    return ok


def check_onnxruntime() -> bool:
    """Install the provider-appropriate ONNX Runtime distribution.

    The import name is ``onnxruntime`` for both distributions, so the choice
    belongs here rather than in the shared requirements file.  Windows keeps
    the CPU package for compatibility; Linux/NVIDIA Pods use the GPU wheel.
    """

    use_gpu = sys.platform.startswith("linux") and nvidia_gpu_name() is not None
    package = "onnxruntime-gpu" if use_gpu else "onnxruntime"
    opposite = "onnxruntime" if use_gpu else "onnxruntime-gpu"
    print(f"[SETUP] Selecting {package} for WD14 (Linux/NVIDIA GPU: {use_gpu}).")
    # Avoid leaving both distributions installed: both expose the same Python
    # import package and the last wheel installed would otherwise win silently.
    try:
        importlib.metadata.version(opposite)
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        if not run_command([sys.executable, "-m", "pip", "uninstall", "-y", opposite]):
            print(f"[WARN] Could not remove the alternate ONNX Runtime package: {opposite}")
    if run_command([sys.executable, "-m", "pip", "install", "--upgrade", package]):
        return True
    if use_gpu:
        print("[WARN] onnxruntime-gpu installation failed; falling back to CPU onnxruntime.")
        return run_command([sys.executable, "-m", "pip", "install", "--upgrade", "onnxruntime"])
    return False


def main() -> int:
    print("=" * 60)
    print("  SDXL LoRA Factory - Environment Setup Check")
    print("=" * 60)
    if not check_sd_scripts():
        return 1
    if not check_pytorch():
        return 1
    if not check_requirements():
        return 1
    if not check_onnxruntime():
        return 1
    print("[SETUP] Setup check completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
