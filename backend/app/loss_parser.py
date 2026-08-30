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
        self.structured_epoch_averages: List[dict] = []

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

    def merge_structured_records(self, records: List[dict]) -> None:
        """Merge scalar records using the bundled sd-scripts step semantics.

        ``train_network.py`` sends step logs with ``global_step`` as the
        TensorBoard step, but sends ``loss/epoch_average`` through
        ``epoch_logging`` where the TensorBoard step is the one-based epoch.
        Keeping those streams separate prevents an epoch average from being
        mistaken for a step-loss record.
        """

        scalar_cache: Dict[int, dict] = {}
        epoch_by_step: Dict[int, int] = {}
        epoch_records: List[dict] = []
        for item in records:
            try:
                tag = str(item["tag"])
                step = int(item["step"])
                value = float(item["value"])
            except (KeyError, TypeError, ValueError):
                continue

            if tag in {"epoch", "global/epoch", "training/epoch"}:
                epoch_by_step[step] = int(round(value))
                continue
            if tag == "loss/epoch_average":
                epoch_records.append({"step": step, "value": value})
                continue

            record = scalar_cache.setdefault(step, {"step": step})
            if tag in {"loss/current", "loss/current_step"}:
                record["loss"] = value
            elif tag == "loss/average":
                record["average_loss"] = value
            elif tag.startswith("lr/") and "learning_rate" not in record:
                record["learning_rate"] = value

        for item in epoch_records:
            step = item["step"]
            epoch = epoch_by_step.get(step, step if step > 0 else None)
            structured = {
                "epoch": epoch,
                "average_loss": item["value"],
                "step": step,
            }
            existing_structured = next(
                (
                    record
                    for record in self.structured_epoch_averages
                    if record.get("epoch") == epoch and record.get("step") == step
                ),
                None,
            )
            if existing_structured is None:
                self.structured_epoch_averages.append(structured)
            else:
                existing_structured.update(structured)
            if epoch is None:
                continue
            existing = self.epochs.get(epoch)
            if existing is None:
                self.epochs[epoch] = structured
            else:
                # Console output is the preferred source when it supplied an
                # epoch average.  TensorBoard fills only missing fields.
                if existing.get("average_loss") is None:
                    existing["average_loss"] = item["value"]
                if existing.get("step") is None:
                    existing["step"] = step

        for step in sorted(scalar_cache):
            record = scalar_cache[step]
            existing = next((item for item in self.steps if item.get("step") == step), None)
            if existing is None:
                self.steps.append(record)
            else:
                # Do not erase the epoch inferred from stdout when a
                # TensorBoard step record has no epoch field.
                existing.update({key: value for key, value in record.items() if value is not None})

        self.steps.sort(key=lambda item: item.get("step", 0))
        self.epochs = dict(sorted(self.epochs.items()))
        if epoch_records and self.average_loss is None:
            self.average_loss = max(epoch_records, key=lambda item: item["step"])["value"]
        if self.steps:
            latest = self.steps[-1]
            self.step = latest.get("step", self.step)
            if latest.get("loss") is not None:
                self.loss = latest["loss"]
            if latest.get("average_loss") is not None:
                self.average_loss = latest["average_loss"]
            if latest.get("learning_rate") is not None:
                self.learning_rate = latest["learning_rate"]

    def merge_tensorboard(self, logging_dir: Path) -> None:
        """Prefer structured TensorBoard values when the optional reader is installed."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            return

        event_files = list(logging_dir.rglob("events.out.tfevents.*")) if logging_dir.exists() else []
        records: List[dict] = []
        for event_file in event_files:
            try:
                accumulator = EventAccumulator(str(event_file))
                accumulator.Reload()
            except Exception:
                continue
            tags = accumulator.Tags().get("scalars", [])
            for tag in tags:
                try:
                    values = accumulator.Scalars(tag)
                except Exception:
                    continue
                for item in values:
                    records.append({"tag": tag, "step": int(item.step), "value": float(item.value)})
        self.merge_structured_records(records)

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
        return {
            **self.progress(),
            "steps": self.steps,
            "epochs": list(self.epochs.values()),
            "final_loss": self.loss,
        }
