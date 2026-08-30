"""Minimal WD14 caption generator used by the web application.

The model is downloaded to the normal Hugging Face cache on first use and is
never written into the repository. ``--allow_mock`` exists only for local UI
smoke tests; RunPod defaults to a real WD14 run and fails loudly if its
dependencies are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import onnxruntime as rt
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
MODEL_FILENAME = "model.onnx"
CSV_FILENAME = "selected_tags.csv"
TAG_THRESHOLD = 0.35
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def download_model():
    print("Loading WD14 Model...", flush=True)
    try:
        model_path = hf_hub_download(MODEL_REPO, MODEL_FILENAME, local_files_only=True)
        csv_path = hf_hub_download(MODEL_REPO, CSV_FILENAME, local_files_only=True)
        print("Model loaded from local cache.", flush=True)
    except Exception:
        print("Model not found locally. Downloading from HuggingFace...", flush=True)
        model_path = hf_hub_download(MODEL_REPO, MODEL_FILENAME, local_files_only=False)
        csv_path = hf_hub_download(MODEL_REPO, CSV_FILENAME, local_files_only=False)
        print("Download complete.", flush=True)
    return model_path, csv_path


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def preprocess_image(image_path: Path, target_size: int = 448):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    max_dim = max(width, height)
    pad_width = (max_dim - width) // 2
    pad_height = (max_dim - height) // 2
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, (pad_width, pad_height))
    image = padded.resize((target_size, target_size), Image.Resampling.LANCZOS)
    image_np = np.array(image, dtype=np.float32)[:, :, ::-1]
    return np.expand_dims(image_np, axis=0)


def run_mock_tagger(image_paths: list[Path]) -> int:
    print("WARNING: WD14 dependencies are missing; running explicit mock mode.", flush=True)
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        image_path.with_suffix(".txt").write_text(
            "1girl, solo, masterpiece, best quality, mock_tag", encoding="utf-8"
        )
        print(f"[TAGGER_PROGRESS] {index}/{total}", flush=True)
    print("Mock tagging complete.", flush=True)
    return 0


def run_real_tagger(image_paths: list[Path]) -> int:
    model_path, csv_path = download_model()
    tag_names, _, general_indexes, character_indexes = load_labels(csv_path)
    print("Initializing ONNX Runtime...", flush=True)
    available = set(rt.get_available_providers())
    providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in available]
    session = rt.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    failures = 0
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        try:
            predictions = session.run(None, {input_name: preprocess_image(image_path)})[0][0]
            tags = [
                tag_names[tag_index].replace("_", " ")
                for tag_index in general_indexes + character_indexes
                if predictions[tag_index] > TAG_THRESHOLD
            ]
            image_path.with_suffix(".txt").write_text(", ".join(tags), encoding="utf-8")
        except Exception as error:
            failures += 1
            print(f"ERROR processing {image_path}: {error}", flush=True)
        print(f"[TAGGER_PROGRESS] {index}/{total}", flush=True)
    print(f"WD14 tagging complete: {total - failures}/{total} succeeded.", flush=True)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--allow_mock", action="store_true")
    args = parser.parse_args()
    dataset_path = Path(args.train_data_dir).expanduser()
    image_paths = (
        sorted(
            (
                path
                for path in dataset_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: str(path).lower(),
        )
        if dataset_path.is_dir()
        else []
    )
    if not image_paths:
        print(f"ERROR: no images found in dataset folder: {dataset_path}", flush=True)
        return 2
    print(f"Found {len(image_paths)} images. Starting WD14 Tagger...", flush=True)
    if HAS_DEPS:
        return run_real_tagger(image_paths)
    if args.allow_mock:
        return run_mock_tagger(image_paths)
    print(
        "ERROR: missing WD14 dependencies (onnxruntime, huggingface_hub, Pillow, pandas, numpy). "
        "Install backend/requirements.txt or set WD14_ALLOW_MOCK=true only for a smoke test.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
