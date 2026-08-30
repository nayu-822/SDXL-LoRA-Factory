from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Iterable, List

from .config import Settings


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}
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


def is_within(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def validate_local_dataset(value: str, settings: Settings) -> Path:
    path = normalize_local_path(value, settings)
    if not settings.allow_external_dataset_paths and not is_within(
        path, (settings.data_root, settings.workspace_root)
    ):
        raise ValueError("dataset path must be inside WORKSPACE_ROOT")
    if not path.exists():
        raise ValueError(f"dataset path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"dataset path is not a directory: {path}")
    return path


def image_files(dataset_path: Path) -> List[Path]:
    return sorted((p for p in dataset_path.rglob("*") if is_image(p)), key=lambda p: str(p).lower())


def validate_image_path(value: str, settings: Settings) -> Path:
    path = normalize_local_path(value, settings)
    allowed_roots = (settings.data_root, settings.output_root, settings.job_root, settings.workspace_root)
    if not settings.allow_external_dataset_paths and not is_within(path, allowed_roots):
        raise ValueError("file path is outside the configured workspace")
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
        if not candidate.is_absolute():
            candidate = settings.workspace_root / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise ValueError(f"base model does not exist: {candidate}")
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
