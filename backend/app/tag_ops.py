from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .config import Settings
from .paths import caption_path_for_image, image_files, validate_image_path, validate_local_dataset


def get_tag_category(tag: str) -> str:
    value = tag.strip().lower()
    if value.startswith("@") or value in {"1girl", "1boy", "girl", "boy", "solo"}:
        return "tag-char"
    if any(item in value for item in ("hair", "eyes", "skin", "body")):
        return "tag-char"
    if any(item in value for item in ("shirt", "skirt", "pants", "dress", "uniform", "clothes", "wearing")):
        return "tag-clothes"
    if any(item in value for item in ("background", "outdoor", "indoor", "room", "sky", "tree", "nature")):
        return "tag-bg"
    if any(item in value for item in ("masterpiece", "best quality", "highres", "year", "score")):
        return "tag-meta"
    return "tag-general"


def normalize_tags(tags: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for raw in tags:
        value = str(raw).replace("\r", " ").replace("\n", " ").strip()
        if not value:
            continue
        if "," in value:
            chunks = value.split(",")
        else:
            chunks = [value]
        for chunk in chunks:
            tag = chunk.strip()
            if tag and tag.lower() not in seen:
                result.append(tag)
                seen.add(tag.lower())
    return result


def read_tags(image_path: Path) -> List[str]:
    caption_path = caption_path_for_image(image_path)
    if not caption_path.is_file():
        return []
    return normalize_tags(caption_path.read_text(encoding="utf-8", errors="replace").split(","))


def write_tags(image_path: Path, tags: Iterable[str]) -> None:
    caption_path_for_image(image_path).write_text(", ".join(normalize_tags(tags)), encoding="utf-8")


def list_image_records(dataset_value: str, settings: Settings) -> List[dict]:
    dataset_path = validate_local_dataset(dataset_value, settings)
    records = []
    for image_path in image_files(dataset_path):
        records.append(
            {
                "name": image_path.name,
                "relative_path": str(image_path.relative_to(dataset_path)).replace("\\", "/"),
                "path": str(image_path),
                "tags": [{"name": tag, "category": get_tag_category(tag)} for tag in read_tags(image_path)],
            }
        )
    return records


def update_image_tags(image_value: str, tags: Iterable[str], settings: Settings) -> List[str]:
    image_path = validate_image_path(image_value, settings)
    normalized = normalize_tags(tags)
    write_tags(image_path, normalized)
    return normalized


def batch_update_tags(dataset_value: str, tags: Iterable[str], position: str, settings: Settings) -> dict:
    dataset_path = validate_local_dataset(dataset_value, settings)
    normalized = normalize_tags(tags)
    if not normalized:
        raise ValueError("at least one tag is required")
    if position not in {"append", "prepend"}:
        raise ValueError("position must be append or prepend")
    changed = 0
    tags_changed = 0
    for image_path in image_files(dataset_path):
        current = read_tags(image_path)
        current_lower = {tag.lower() for tag in current}
        if position == "prepend":
            remaining = [tag for tag in current if tag.lower() not in {item.lower() for item in normalized}]
            updated = normalized + remaining
        else:
            updated = current + [tag for tag in normalized if tag.lower() not in current_lower]
        if updated != current:
            write_tags(image_path, updated)
            changed += 1
            tags_changed += abs(len(updated) - len(current))
    return {"images_changed": changed, "tags_changed": tags_changed, "image_count": len(image_files(dataset_path))}


def batch_remove_tags(dataset_value: str, tags: Iterable[str], settings: Settings) -> dict:
    dataset_path = validate_local_dataset(dataset_value, settings)
    normalized = normalize_tags(tags)
    if not normalized:
        raise ValueError("at least one tag is required")
    remove_set = {tag.lower() for tag in normalized}
    changed = 0
    removed = 0
    for image_path in image_files(dataset_path):
        current = read_tags(image_path)
        updated = [tag for tag in current if tag.lower() not in remove_set]
        if updated != current:
            write_tags(image_path, updated)
            changed += 1
            removed += len(current) - len(updated)
    return {"images_changed": changed, "tags_removed": removed, "image_count": len(image_files(dataset_path))}
