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
    lines.extend(["", "[LOSS HISTORY]"])
    loss_history = data.get("loss_history", {})
    for epoch in loss_history.get("epochs", []):
        number = epoch.get("epoch", "")
        lines.append(f"epoch_{number}_avg_loss={_value(epoch.get('average_loss'))}")
        if epoch.get("learning_rate") is not None:
            lines.append(f"epoch_{number}_learning_rate={_value(epoch.get('learning_rate'))}")
    for step in loss_history.get("steps", []):
        lines.append(
            "step={step} epoch={epoch} loss={loss} average_loss={average_loss} learning_rate={learning_rate}".format(
                step=_value(step.get("step")),
                epoch=_value(step.get("epoch")),
                loss=_value(step.get("loss")),
                average_loss=_value(step.get("average_loss")),
                learning_rate=_value(step.get("learning_rate")),
            )
        )
    lines.append(f"final_loss={_value(loss_history.get('final_loss'))}")
    lines.extend(["", "[SAMPLE SETTINGS]"])
    _write_pairs(lines, data.get("sample_settings", {}))
    lines.extend(["", "[OUTPUT FILES]"])
    output_files: Iterable[str] = data.get("output_files", [])
    lines.extend(str(item) for item in output_files)
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
