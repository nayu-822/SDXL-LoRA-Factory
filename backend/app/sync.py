from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .config import Settings
from .events import EventHub
from .models import CaptionSyncRequest, DatasetSyncRequest, OutputSyncRequest
from .paths import (
    dataset_counts,
    dataset_name_from_gdrive,
    gdrive_uri,
    normalize_local_path,
    safe_slug,
    validate_gdrive_path,
    validate_local_dataset,
)
from .processes import ProcessManager, ProcessRun
from .rclone_sync import RcloneService


_PERCENT_RE = re.compile(r"(\d{1,3})%")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SyncService:
    """Safe, copy-only Google Drive operations backed by rclone."""

    def __init__(self, settings: Settings, rclone: RcloneService, manager: ProcessManager, hub: EventHub):
        self.settings = settings
        self.rclone = rclone
        self.manager = manager
        self.hub = hub
        self.states: dict[str, dict] = {}

    async def _start(self, operation: str, command: list[str], state: dict, log_name: str) -> Tuple[dict, ProcessRun]:
        sync_id = state["sync_id"]

        async def on_line(run: ProcessRun, line: str) -> None:
            state["run"] = run.public_dict()
            state["last_line"] = line
            match = _PERCENT_RE.search(line)
            if match:
                state["progress_percent"] = min(100, int(match.group(1)))
                await self.hub.progress("sync", {"operation": operation, **state}, sync_id)
            await self.hub.log("sync", f"[{operation}] {line}", sync_id)

        async def on_done(run: ProcessRun) -> None:
            state["run"] = run.public_dict()
            state["ended_at"] = run.ended_at or _now()
            state["returncode"] = run.returncode
            if run.status == "succeeded":
                if operation == "dataset":
                    try:
                        state.update(dataset_counts(Path(state["local_path"])))
                    except OSError as error:
                        state["error"] = f"could not inspect synced dataset: {error}"
                    if state.get("image_count", 0) == 0:
                        state["status"] = "failed"
                        state["error"] = "dataset is empty: no supported images were copied"
                    else:
                        state["status"] = "completed"
                else:
                    state["status"] = "completed"
            elif run.status == "cancelled":
                state["status"] = "cancelled"
                state["error"] = "sync cancelled"
            else:
                state["status"] = "failed"
                state["error"] = run.error or f"rclone exited with code {run.returncode}"
            await self.hub.status("sync", state, sync_id)

        run = await self.manager.start(
            kind="sync",
            run_id=sync_id,
            command=command,
            cwd=self.settings.workspace_root,
            log_path=self.settings.job_root / "sync" / f"{log_name}.log",
            on_line=on_line,
            on_done=on_done,
        )
        state["status"] = "running"
        state["run"] = run.public_dict()
        self.states[sync_id] = state
        await self.hub.status("sync", state, sync_id)
        return state, run

    async def _new_state(self, operation: str, local_path: Path, remote_path: str) -> dict:
        await self.rclone.check_remote()
        sync_id = f"{operation}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        return {
            "sync_id": sync_id,
            "operation": operation,
            "status": "queued",
            "local_path": str(local_path),
            "gdrive_path": remote_path,
            "started_at": _now(),
            "progress_percent": 0,
            "last_line": "",
            "error": "",
        }

    async def start_dataset(self, request: DatasetSyncRequest) -> Tuple[dict, ProcessRun]:
        remote_path = validate_gdrive_path(request.gdrive_path, self.settings.gdrive_remote)
        dataset_name = safe_slug(request.dataset_name or dataset_name_from_gdrive(remote_path, self.settings), "dataset")
        local_path = self.settings.data_root / dataset_name
        local_path.mkdir(parents=True, exist_ok=True)
        state = await self._new_state("dataset", local_path, remote_path)
        command = self.rclone.dataset_command(remote_path, local_path)
        return await self._start("dataset", command, state, state["sync_id"])

    async def start_captions(self, request: CaptionSyncRequest) -> Tuple[dict, ProcessRun]:
        local_path = validate_local_dataset(request.local_dataset_path, self.settings)
        remote_path = validate_gdrive_path(request.gdrive_path, self.settings.gdrive_remote)
        state = await self._new_state("captions", local_path, remote_path)
        state["caption_count"] = sum(
            1 for path in local_path.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"
        )
        command = self.rclone.captions_command(local_path, remote_path)
        return await self._start("captions", command, state, state["sync_id"])

    async def start_output(self, request: OutputSyncRequest) -> Tuple[dict, ProcessRun]:
        local_path = normalize_local_path(request.local_output_path, self.settings)
        if not local_path.exists() or not local_path.is_dir():
            raise ValueError(f"output path does not exist: {local_path}")
        remote_path = validate_gdrive_path(request.gdrive_path, self.settings.gdrive_remote)
        state = await self._new_state("output", local_path, remote_path)
        state["file_count"] = sum(1 for path in local_path.rglob("*") if path.is_file())
        command = self.rclone.output_command(local_path, remote_path)
        return await self._start("output", command, state, state["sync_id"])

    def latest(self) -> Optional[dict]:
        if not self.states:
            return None
        return list(self.states.values())[-1]

    def public_status(self) -> dict:
        return {key: value for key, value in self.states.items()}
