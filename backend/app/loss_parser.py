from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional


_NUMBER = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


class LossParser:
    """Parse sd-scripts console output and optionally merge TensorBoard data."""

    epoch_re = re.compile(r"\bepoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    progress_re = re.compile(r"(?:\b|\|\s*)(\d+)%\|.*?\|\s*(\d+)\s*/\s*(\d+)")
    current_loss_re = re.compile(r"(?:^|[\s,])(?:loss|loss/current)\s*[:=]\s*" + _NUMBER, re.IGNORECASE)
    average_loss_re = re.compile(
        r"(?:^|[\s,])(?:avr_loss|average[_ ]loss|loss/average|epoch_average)\s*[:=]\s*" + _NUMBER,
        re.IGNORECASE,
    )
    lr_re = re.compile(r"(?:^|[\s,])(?:lr(?:/[^\s,=]+)?|learning[_ ]rate)\s*[:=]\s*" + _NUMBER, re.IGNORECASE)

    def __init__(self):
        self.epoch: Optional[int] = None
        self.total_epochs: Optional[int] = None
        self.step: Optional[int] = None
        self.total_steps: Optional[int] = None
        self.loss: Optional[float] = None
        self.average_loss: Optional[float] = None
        self.learning_rate: Optional[float] = None
        self.steps: List[dict] = []
        self.epochs: Dict[int, dict] = {}

    @staticmethod
    def _float(match: Optional[re.Match]) -> Optional[float]:
        return float(match.group(1)) if match else None

    def on_line(self, line: str) -> dict:
        epoch_match = self.epoch_re.search(line)
        if epoch_match:
            self.epoch = int(epoch_match.group(1))
            self.total_epochs = int(epoch_match.group(2))

        progress_match = self.progress_re.search(line)
        if progress_match:
            self.step = int(progress_match.group(2))
            self.total_steps = int(progress_match.group(3))

        current = self._float(self.current_loss_re.search(line))
        average = self._float(self.average_loss_re.search(line))
        learning_rate = self._float(self.lr_re.search(line))
        if current is not None:
            self.loss = current
        if average is not None:
            self.average_loss = average
        if learning_rate is not None:
            self.learning_rate = learning_rate

        if self.step is not None and (current is not None or average is not None):
            record = {
                "step": self.step,
                "epoch": self.epoch,
                "loss": self.loss,
                "average_loss": self.average_loss,
                "learning_rate": self.learning_rate,
            }
            if self.steps and self.steps[-1].get("step") == self.step:
                self.steps[-1] = record
            else:
                self.steps.append(record)

        if self.epoch is not None and average is not None:
            self.epochs[self.epoch] = {
                "epoch": self.epoch,
                "average_loss": average,
                "step": self.step,
                "learning_rate": self.learning_rate,
            }
        return self.progress()

    def merge_tensorboard(self, logging_dir: Path) -> None:
        """Prefer structured TensorBoard values when the optional reader is installed."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return

        event_files = list(logging_dir.rglob("events.out.tfevents.*")) if logging_dir.exists() else []
        for event_file in event_files:
            try:
                accumulator = EventAccumulator(str(event_file))
                accumulator.Reload()
            except Exception:
                continue
            tags = accumulator.Tags().get("scalars", [])
            scalar_cache: Dict[int, dict] = {}
            for tag in tags:
                try:
                    values = accumulator.Scalars(tag)
                except Exception:
                    continue
                for item in values:
                    record = scalar_cache.setdefault(int(item.step), {"step": int(item.step)})
                    if tag in {"loss/current", "loss/current_step"}:
                        record["loss"] = float(item.value)
                    elif tag in {"loss/average", "loss/epoch_average"}:
                        record["average_loss"] = float(item.value)
                        if tag == "loss/epoch_average":
                            epoch = len(self.epochs) + 1
                            record["epoch"] = epoch
                            self.epochs[epoch] = {
                                "epoch": epoch,
                                "average_loss": float(item.value),
                                "step": int(item.step),
                            }
                    elif tag.startswith("lr/") and "learning_rate" not in record:
                        record["learning_rate"] = float(item.value)
            for step in sorted(scalar_cache):
                record = scalar_cache[step]
                if any(existing.get("step") == step for existing in self.steps):
                    for existing in self.steps:
                        if existing.get("step") == step:
                            existing.update(record)
                            break
                else:
                    self.steps.append(record)
        self.steps.sort(key=lambda item: item.get("step", 0))
        if self.steps:
            latest = self.steps[-1]
            self.step = latest.get("step", self.step)
            self.loss = latest.get("loss", self.loss)
            self.average_loss = latest.get("average_loss", self.average_loss)
            self.learning_rate = latest.get("learning_rate", self.learning_rate)

    def progress(self) -> dict:
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "step": self.step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "average_loss": self.average_loss,
            "learning_rate": self.learning_rate,
        }

    def as_dict(self) -> dict:
        return {**self.progress(), "steps": self.steps, "epochs": list(self.epochs.values())}
