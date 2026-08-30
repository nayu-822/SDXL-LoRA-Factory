from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path


def gpu_info() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            first = result.stdout.strip().splitlines()[0]
            name, _, memory = first.partition(",")
            return {"name": name.strip(), "memory": f"{memory.strip()} MiB" if memory else "N/A"}
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return {"name": "nvidia-smi not found (CPU only or no NVIDIA GPU)", "memory": "N/A"}


def torch_info() -> dict:
    info = {"python_version": sys.version.split()[0], "torch_version": "not installed", "cuda": False}
    try:
        import torch

        info["torch_version"] = str(torch.__version__)
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


def dependency_info() -> dict:
    names = [
        "fastapi",
        "uvicorn",
        "torch",
        "diffusers",
        "transformers",
        "onnxruntime",
        "onnxruntime-gpu",
        "tensorboard",
    ]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def sd_scripts_version(sd_scripts_root: Path) -> str:
    git_dir = sd_scripts_root / ".git"
    if git_dir.exists():
        try:
            result = subprocess.run(
                ["git", "-C", str(sd_scripts_root), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return "bundled sd-scripts"


def executable_available(executable: str) -> bool:
    path = Path(executable)
    return path.is_file() if path.is_absolute() else shutil.which(executable) is not None
