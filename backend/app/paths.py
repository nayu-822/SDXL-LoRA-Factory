from __future__ import annotations

import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Iterable, List

from .config import Settings


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
MODEL_EXTENSIONS = {".safetensors", ".ckpt"}
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_local_path(value: str, settings: Settings) -> Path:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("local path is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = settings.workspace_root / path
    return path.resolve()


def validate_output_path(value: str, settings: Settings) -> Path:
    """Resolve an output path and keep RunPod model storage read-only."""

    path = normalize_local_path(value, settings)
    return settings.validate_generated_path(path, "output path")


def is_within(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _allows_external_dataset_paths(settings: Settings) -> bool:
    """Keep the legacy external-folder behavior on Windows only.

    A RunPod process should never turn an arbitrary WORKSPACE_ROOT child into
    a dataset-editing target.  Windows keeps the opt-in compatibility switch
    because the original desktop application accepted folders anywhere on the
    local machine.
    """

    return os.name == "nt" and settings.allow_external_dataset_paths


def validate_local_dataset(value: str, settings: Settings) -> Path:
    path = normalize_local_path(value, settings)
    data_root = settings.data_root.resolve()
    if not _allows_external_dataset_paths(settings) and (
        path == data_root or not is_within(path, (data_root,))
    ):
        raise ValueError("dataset path must be a child of DATASET_DIR")
    if not path.exists():
        raise ValueError(f"dataset path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"dataset path is not a directory: {path}")
    return path


def image_files(dataset_path: Path) -> List[Path]:
    return sorted((p for p in dataset_path.rglob("*") if is_image(p)), key=lambda p: str(p).lower())


def validate_image_path(value: str, settings: Settings) -> Path:
    path = normalize_local_path(value, settings)
    if not _allows_external_dataset_paths(settings) and not is_within(path, (settings.data_root,)):
        raise ValueError("image path must be inside DATASET_DIR")
    if not path.exists() or not is_image(path):
        raise ValueError(f"image does not exist: {path}")
    return path


def validate_model_reference(value: str, settings: Settings) -> str:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("base model path or Hugging Face model id is required")
    candidate = Path(raw).expanduser()
    looks_local = candidate.is_absolute() or candidate.exists() or candidate.suffix.lower() in {
        ".safetensors",
        ".ckpt",
        ".pt",
        ".bin",
    }
    if looks_local:
        if os.name != "nt" and not settings.model_directory.is_dir():
            raise ValueError(
                f"{settings.model_directory} が見つかりません。RunPodのNetwork Volumeが正しく接続されているか確認してください。"
            )
        if not candidate.is_absolute():
            # Linux/RunPod model names are resolved against the read-only model
            # directory.  Windows keeps the old workspace-relative behavior.
            base = settings.model_directory if os.name != "nt" and candidate.suffix.lower() in MODEL_EXTENSIONS else settings.workspace_root
            candidate = base / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise ValueError(f"base model does not exist: {candidate}")
        if not candidate.is_file():
            raise ValueError(f"base model is not a file: {candidate}")
        if os.name != "nt" and not is_within(candidate, (settings.model_directory,)):
            raise ValueError(
                f"RunPod base models must be inside MODEL_DIR ({settings.model_directory}); "
                "the shared checkpoint directory is read-only"
            )
        return str(candidate)
    # A non-path value is treated as a diffusers/Hugging Face model id.
    if "\n" in raw or "\r" in raw or len(raw) > 512:
        raise ValueError("invalid base model reference")
    return raw


def validate_optional_model_path(value: str, settings: Settings) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = settings.workspace_root / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise ValueError(f"VAE does not exist: {candidate}")
    return str(candidate)


def list_base_models(settings: Settings) -> dict:
    """Return the files available for the RunPod Base Model selector.

    The directory is intentionally only read.  No download, rename, move, or
    cleanup operation in the application targets this path.
    """

    root = settings.model_directory.expanduser().resolve()
    result = {
        "model_dir": str(root),
        "exists": root.is_dir(),
        "models": [],
        "error": "",
    }
    if not root.exists():
        result["error"] = (
            f"{root} が見つかりません。RunPodのNetwork Volumeが正しく接続されているか確認してください。"
        )
        return result
    if not root.is_dir():
        result["error"] = f"MODEL_DIR is not a directory: {root}"
        return result

    models = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
            continue
        try:
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
        except OSError:
            continue
        models.append(
            {
                "name": path.name,
                "relative_path": relative,
                "path": str(path),
                "size_bytes": size,
            }
        )
    result["models"] = models
    if not models:
        result["error"] = f"{root} に .safetensors または .ckpt のモデルがありません。"
    return result


def validate_gdrive_path(value: str, remote_name: str) -> str:
    raw = (value or "").strip().replace("\\", "/")
    prefix = f"{remote_name}:"
    if raw.lower().startswith(prefix.lower()):
        raw = raw[len(prefix) :]
    if not raw or raw.startswith("/"):
        raise ValueError("Google Drive path must be a non-empty relative path")
    if ":" in raw.split("/", 1)[0]:
        raise ValueError("a different rclone remote is not allowed")
    parts = [part for part in PurePosixPath(raw).parts if part not in {""}]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("Google Drive path cannot contain . or ..")
    return "/".join(parts)


def gdrive_uri(value: str, settings: Settings) -> str:
    return f"{settings.gdrive_remote}:{validate_gdrive_path(value, settings.gdrive_remote)}"


def dataset_name_from_gdrive(value: str, settings: Settings) -> str:
    normalized = validate_gdrive_path(value, settings.gdrive_remote)
    name = normalized.rsplit("/", 1)[-1]
    return safe_slug(name, fallback="dataset")


def safe_slug(value: str, fallback: str = "item") -> str:
    cleaned = _SLUG_RE.sub("-", (value or "").strip()).strip(".-_")
    return cleaned[:96] or fallback


def _validated_dataset_cleanup_target(value: Path, settings: Settings) -> Path:
    """Resolve and validate a direct child of DATASET_DIR before deletion."""

    data_root = settings.data_root.expanduser().resolve()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = data_root / candidate
    if candidate.is_symlink():
        raise ValueError("dataset cleanup target cannot be a symlink")
    target = candidate.resolve()
    if target == data_root or target.parent != data_root:
        raise ValueError("dataset cleanup target must be a direct child of DATASET_DIR")
    if target.exists() and not target.is_dir():
        raise ValueError("dataset cleanup target is not a directory")
    return target


def dataset_directory_for_name(dataset_name: str, settings: Settings) -> Path:
    """Return the only directory that a Drive dataset sync may clear."""

    return _validated_dataset_cleanup_target(
        settings.data_root.expanduser().resolve() / safe_slug(dataset_name, "dataset"), settings
    )


def clear_dataset_directory(dataset_path: Path, settings: Settings) -> int:
    """Clear one dataset directory while preserving DATASET_DIR itself.

    The target is validated immediately before any filesystem mutation.  Child
    symlinks are unlinked as links, never traversed, and all other children are
    removed only from the validated target.
    """

    target = _validated_dataset_cleanup_target(Path(dataset_path), settings)
    settings.data_root.mkdir(parents=True, exist_ok=True)
    removed_count = 0
    if target.exists():
        for child in target.iterdir():
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)
            removed_count += 1
    target.mkdir(parents=True, exist_ok=True)
    return removed_count


def caption_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def dataset_counts(dataset_path: Path) -> dict:
    images = image_files(dataset_path)
    captions = [caption_path_for_image(path) for path in images if caption_path_for_image(path).is_file()]
    return {
        "image_count": len(images),
        "caption_count": len(captions),
        "missing_caption_count": len(images) - len(captions),
    }
