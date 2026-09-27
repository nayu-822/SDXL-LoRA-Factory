from __future__ import annotations

import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .config import Settings
from .events import EventHub
from .models import CaptionSyncRequest, DatasetSyncRequest, OutputSyncRequest
from .paths import (
    clear_dataset_directory,
    dataset_counts,
    dataset_directory_for_name,
    dataset_name_from_gdrive,
    is_within,
    validate_gdrive_path,
    validate_local_dataset,
    validate_output_path,
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
        dataset_name = request.dataset_name or dataset_name_from_gdrive(remote_path, self.settings)
        local_path = dataset_directory_for_name(dataset_name, self.settings)
        state = await self._new_state("dataset", local_path, remote_path)
        # rclone remains copy-only.  Clear only the validated dataset child
        # after the remote check, so deleted Drive files cannot linger locally.
        state["cleared_entry_count"] = clear_dataset_directory(local_path, self.settings)
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
        local_path = validate_output_path(request.local_output_path, self.settings)
        if not local_path.exists() or not local_path.is_dir():
            raise ValueError(f"output path does not exist: {local_path}")
        remote_path = validate_gdrive_path(request.gdrive_path, self.settings.gdrive_remote)
        state = await self._new_state("output", local_path, remote_path)
        state["file_count"] = sum(1 for path in local_path.rglob("*") if path.is_file())
        command = self.rclone.output_command(local_path, remote_path)
        return await self._start("output", command, state, state["sync_id"])

    async def start_output_file(self, local_file: Path, remote_path: str) -> Tuple[dict, ProcessRun]:
        """Copy one final output artifact without re-copying the whole run."""

        local_path = validate_output_path(str(local_file), self.settings)
        if not local_path.is_file():
            raise ValueError(f"output file does not exist: {local_path}")
        normalized = validate_gdrive_path(remote_path, self.settings.gdrive_remote)
        state = await self._new_state("output", local_path, normalized)
        state["file_count"] = 1
        command = self.rclone.build_copy_command(
            str(local_path), f"{self.settings.gdrive_remote}:{normalized}"
        )
        return await self._start("output", command, state, state["sync_id"])

    @staticmethod
    def _copy_for_backup(source: Path, destination: Path, predicate=None) -> int:
        """Materialize one category of a structured run backup."""

        if not source.exists() or not source.is_dir():
            return 0
        copied = 0
        for path in sorted(source.rglob("*"), key=lambda item: str(item).casefold()):
            if not path.is_file() or (predicate is not None and not predicate(path)):
                continue
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                try:
                    if target.samefile(path):
                        copied += 1
                        continue
                except OSError:
                    target.unlink()
            try:
                # Avoid temporarily doubling large LoRA files when /data is a
                # single filesystem.  Copy is the portable fallback.
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
            copied += 1
        return copied

    def prepare_run_backup(
        self,
        run_id: str,
        dataset_path: Path,
        config_path: Path,
        output_path: Path,
        logs_path: Path,
    ) -> Path:
        """Create the local tree that is copied to Drive for one run."""

        root = self.settings.run_backup_root / run_id
        if root.exists():
            # This path is generated application state below CONFIG_DIR.  It
            # is never derived from a user-selected or model path.
            shutil.rmtree(root)
        for name in ("lora", "captions", "config", "logs", "samples"):
            (root / name).mkdir(parents=True, exist_ok=True)
        self._copy_for_backup(
            dataset_path,
            root / "captions",
            predicate=lambda path: path.suffix.lower() == ".txt",
        )
        self._copy_for_backup(config_path, root / "config")
        self._copy_for_backup(output_path / "lora", root / "lora")
        self._copy_for_backup(output_path / "samples", root / "samples")
        self._copy_for_backup(logs_path, root / "logs")
        summary = output_path / "training_summary.txt"
        if summary.is_file():
            shutil.copy2(summary, root / "config" / summary.name)
            shutil.copy2(summary, root / "logs" / summary.name)
        return root

    async def start_run_backup(self, local_root: Path, remote_path: str) -> Tuple[dict, ProcessRun]:
        """Copy a structured run backup with rclone copy, never sync."""

        local_path = Path(local_root).expanduser().resolve()
        if not local_path.exists() or not local_path.is_dir():
            raise ValueError(f"backup path does not exist: {local_path}")
        if not is_within(local_path, (self.settings.run_backup_root,)):
            raise ValueError("backup path must be inside the generated run backup directory")
        normalized = validate_gdrive_path(remote_path, self.settings.gdrive_remote)
        state = await self._new_state("backup", local_path, normalized)
        state["file_count"] = sum(1 for path in local_path.rglob("*") if path.is_file())
        command = self.rclone.backup_command(local_path, normalized)
        return await self._start("backup", command, state, state["sync_id"])

    async def start_run_backup_file(self, local_file: Path, remote_path: str) -> Tuple[dict, ProcessRun]:
        """Copy one staged run artifact, used for the final summary refresh."""

        local_path = Path(local_file).expanduser().resolve()
        if not local_path.is_file():
            raise ValueError(f"backup file does not exist: {local_path}")
        if not is_within(local_path, (self.settings.run_backup_root,)):
            raise ValueError("backup file must be inside the generated run backup directory")
        normalized = validate_gdrive_path(remote_path, self.settings.gdrive_remote)
        state = await self._new_state("backup", local_path, normalized)
        state["file_count"] = 1
        command = self.rclone.build_copy_command(
            str(local_path), f"{self.settings.gdrive_remote}:{normalized}"
        )
        return await self._start("backup", command, state, state["sync_id"])

    def latest(self) -> Optional[dict]:
        if not self.states:
            return None
        return list(self.states.values())[-1]

    def public_status(self) -> dict:
        return {key: value for key, value in self.states.items()}
