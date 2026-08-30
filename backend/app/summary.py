from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_value(item) for item in value)
    return str(value)


def _write_pairs(lines: list[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        lines.append(f"{key}={_value(value)}")


def representative_step_records(steps: Iterable[dict], limit: int = 800) -> list[dict]:
    """Return a bounded, deterministic loss sample for the text summary."""

    records = list(steps)
    if limit <= 0 or len(records) <= limit:
        return records

    limit = max(2, limit)
    boundary_indices = {0, len(records) - 1}
    for index, record in enumerate(records):
        epoch = record.get("epoch")
        if epoch is None:
            continue
        previous = records[index - 1].get("epoch") if index else None
        following = records[index + 1].get("epoch") if index + 1 < len(records) else None
        if index == 0 or epoch != previous or epoch != following:
            boundary_indices.add(index)

    if len(boundary_indices) > limit:
        candidates = sorted(boundary_indices)
        selected = {candidates[0], candidates[-1]}
        for position in range(1, limit - 1):
            candidate_index = round(position * (len(candidates) - 1) / (limit - 1))
            selected.add(candidates[candidate_index])
        boundary_indices = selected

    selected_indices = set(boundary_indices)
    remaining = limit - len(selected_indices)
    if remaining > 0:
        for position in range(1, remaining + 1):
            candidate_index = round(position * (len(records) - 1) / (remaining + 1))
            selected_indices.add(candidate_index)
    return [records[index] for index in sorted(selected_indices)]


def write_training_summary(path: Path, data: dict) -> Path:
    """Write a stable, human/AI-readable summary even for failed jobs."""
    lines: list[str] = []
    lines.append("[JOB INFO]")
    _write_pairs(
        lines,
        {
            "job_id": data.get("job_id"),
            "job_name": data.get("job_name"),
            "start_time": data.get("start_time"),
            "end_time": data.get("end_time"),
            "status": data.get("status"),
            "duration": data.get("duration"),
        },
    )
    lines.extend(["", "[ENVIRONMENT]"])
    _write_pairs(lines, data.get("environment", {}))
    lines.extend(["", "[DATASET]"])
    _write_pairs(lines, data.get("dataset", {}))
    lines.extend(["", "[CAPTION]"])
    _write_pairs(lines, data.get("caption", {}))
    lines.extend(["", "[TRAINING CONFIG]"])
    _write_pairs(lines, data.get("training_config", {}))
    command = data.get("command")
    if command:
        lines.append(f"command={command}")
    lines.extend(["", "[LOSS DIAGNOSTICS]"])
    diagnostics = data.get("loss_diagnostics", {})
    _write_pairs(
        lines,
        {
            "total_training_steps": diagnostics.get("total_training_steps"),
            "total_training_steps_source": diagnostics.get("total_training_steps_source"),
            "estimated_total_steps": diagnostics.get("estimated_total_steps"),
            "effective_images_per_epoch": diagnostics.get("effective_images_per_epoch"),
            "images_x_repeats": diagnostics.get("images_x_repeats"),
            "lowest_epoch_loss": diagnostics.get("lowest_epoch_loss"),
            "lowest_loss_epoch": diagnostics.get("lowest_loss_epoch"),
            "final_loss": diagnostics.get("final_loss", data.get("loss_history", {}).get("final_loss")),
        },
    )
    lines.extend(["", "[LOSS HISTORY]"])
    loss_history = data.get("loss_history", {})
    for epoch in loss_history.get("epochs", []):
        number = epoch.get("epoch", "")
        lines.append(f"epoch_{number}_avg_loss={_value(epoch.get('average_loss'))}")
        if epoch.get("learning_rate") is not None:
            lines.append(f"epoch_{number}_learning_rate={_value(epoch.get('learning_rate'))}")
    all_steps = loss_history.get("steps", [])
    step_limit = int(data.get("step_loss_summary_limit", 800) or 800)
    steps = representative_step_records(all_steps, step_limit)
    lines.append(f"step_records_total={len(all_steps)}")
    lines.append(f"step_records_written={len(steps)}")
    lines.append(f"step_records_sampled={str(len(steps) < len(all_steps)).lower()}")
    for step in steps:
        lines.append(
            "step={step} epoch={epoch} loss={loss} average_loss={average_loss} learning_rate={learning_rate}".format(
                step=_value(step.get("step")),
                epoch=_value(step.get("epoch")),
                loss=_value(step.get("loss")),
                average_loss=_value(step.get("average_loss")),
                learning_rate=_value(step.get("learning_rate")),
            )
        )
    lines.append(f"final_loss={_value(loss_history.get('final_loss', diagnostics.get('final_loss')))}")
    lines.extend(["", "[SAMPLE SETTINGS]"])
    _write_pairs(lines, data.get("sample_settings", {}))
    lines.extend(["", "[OUTPUT FILES]"])
    output_files: Iterable[str] = data.get("output_files", [])
    lines.extend(str(item) for item in output_files)
    lines.extend(["", "[OUTPUT SYNC]"])
    output_sync = data.get("output_sync") or {"enabled": False, "status": "not_requested"}
    _write_pairs(lines, output_sync)
    lines.extend(["", "[WARNINGS]"])
    warnings = data.get("warnings", [])
    lines.extend(f"- {item}" for item in warnings)
    lines.extend(["", "[ERRORS]"])
    errors = data.get("errors", [])
    lines.extend(f"- {item}" for item in errors)
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path
