from __future__ import annotations


def recommend_settings(image_count: int, training_type: str, vram: str) -> dict:
    """A transparent, opt-in heuristic; it never changes the form automatically."""
    count = max(1, int(image_count))
    if count <= 20:
        epochs, repeats = 15, 10
    elif count <= 60:
        epochs, repeats = 10, 6
    else:
        epochs, repeats = 8, 3

    if vram in {"very_low", "low"}:
        rank, alpha, batch_size = 16, 8, 1
    elif vram == "high":
        rank, alpha, batch_size = 32, 16, 2
    else:
        rank, alpha, batch_size = 32, 16, 1

    optimizer = "Prodigy" if training_type == "character" and vram not in {"very_low"} else "AdamW"
    return {
        "epochs": epochs,
        "repeats": repeats,
        "network_dim": rank,
        "network_alpha": alpha,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "learning_rate": 1.0 if optimizer == "Prodigy" else 1e-4,
        "min_snr_gamma": 5,
        "reason": f"rule-based suggestion for {count} images, {training_type} LoRA, {vram} VRAM",
    }
