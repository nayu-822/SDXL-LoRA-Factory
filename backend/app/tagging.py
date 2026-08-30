from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .config import Settings
from .events import EventHub
from .models import CaptionSyncRequest, TaggerConfig
from .paths import dataset_counts, image_files, validate_local_dataset
from .processes import ProcessManager, ProcessRun
from .sync import SyncService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_COUNT_RE = re.compile(r"\b(tagged_count|skipped_count|failed_count)=(\d+)")


class TaggerService:
    def __init__(self, settings: Settings, manager: ProcessManager, hub: EventHub, sync_service: SyncService):
        self.settings = settings
        self.manager = manager
        self.hub = hub
        self.sync_service = sync_service
        self.states: dict[str, dict] = {}

    async def start(self, config: TaggerConfig) -> Tuple[dict, ProcessRun]:
        dataset_path = validate_local_dataset(config.path, self.settings)
        images = image_files(dataset_path)
        if not images:
            raise ValueError("dataset is empty: no supported images were found")
        script = self.settings.backend_root / "bundled_tagger.py"
        if not script.is_file():
            raise ValueError(f"WD14 tagger script not found: {script}")

        tagger_id = f"tagger-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        command = [self._python(), str(script), f"--train_data_dir={dataset_path}"]
        if self.settings.allow_mock_tagger:
            command.append("--allow_mock")
        if config.overwrite_existing_captions:
            command.append("--overwrite_existing_captions")
        state = {
            "tagger_id": tagger_id,
            "status": "queued",
            "dataset_path": str(dataset_path),
            "image_count": len(images),
            "caption_count": 0,
            "started_at": _now(),
            "ended_at": "",
            "error": "",
            "auto_sync_captions": config.auto_sync_captions,
            "gdrive_path": config.gdrive_path,
            "overwrite_existing_captions": config.overwrite_existing_captions,
            "tagged_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "progress": {"current": 0, "total": len(images), "percent": 0},
            "warnings": [],
        }
        self.states[tagger_id] = state

        async def on_line(run: ProcessRun, line: str) -> None:
            state["run"] = run.public_dict()
            if "[TAGGER_PROGRESS]" in line:
                try:
                    pair = line.split("[TAGGER_PROGRESS]", 1)[1].strip().split("/", 1)
                    current, total = int(pair[0]), int(pair[1])
                    state["progress"] = {
                        "current": current,
                        "total": total,
                        "percent": round(current / total * 100) if total else 0,
                    }
                    await self.hub.progress("tagger", state["progress"], tagger_id)
                except (ValueError, IndexError):
                    pass
            if "[TAGGER_COUNTS]" in line:
                for key, value in _COUNT_RE.findall(line):
                    state[key] = int(value)
            if "warning" in line.lower():
                state["warnings"].append(line)
            await self.hub.log("tagger", line, tagger_id)

        async def on_done(run: ProcessRun) -> None:
            state["run"] = run.public_dict()
            state["ended_at"] = run.ended_at or _now()
            state["returncode"] = run.returncode
            state["caption_count"] = dataset_counts(dataset_path)["caption_count"]
            if run.status == "succeeded":
                state["status"] = "completed"
            elif run.status == "cancelled":
                state["status"] = "cancelled"
                state["error"] = "tagging cancelled"
            else:
                state["status"] = "failed"
                state["error"] = run.error or f"WD14 tagger exited with code {run.returncode}"

            if state["status"] == "completed" and config.auto_sync_captions:
                try:
                    if not config.gdrive_path.strip():
                        raise ValueError("gdrive_path is required for automatic caption sync")
                    sync_state, sync_run = await self.sync_service.start_captions(
                        CaptionSyncRequest(local_dataset_path=str(dataset_path), gdrive_path=config.gdrive_path)
                    )
                    await self.manager.wait(sync_run)
                    state["caption_sync"] = sync_state
                    if sync_state.get("status") != "completed":
                        state["warnings"].append("automatic caption sync failed")
                except Exception as error:
                    state["warnings"].append(f"automatic caption sync failed: {error}")
            await self.hub.status("tagger", state, tagger_id)

        run = await self.manager.start(
            kind="tagger",
            run_id=tagger_id,
            command=command,
            cwd=self.settings.backend_root,
            log_path=self.settings.job_root / "tagging" / f"{tagger_id}.log",
            on_line=on_line,
            on_done=on_done,
        )
        state["status"] = "running"
        state["run"] = run.public_dict()
        await self.hub.status("tagger", state, tagger_id)
        return state, run

    @staticmethod
    def _python() -> str:
        import sys

        return sys.executable

    def latest(self) -> Optional[dict]:
        return list(self.states.values())[-1] if self.states else None

    def public_status(self) -> dict:
        return {key: value for key, value in self.states.items()}
