from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

from .config import Settings
from .paths import gdrive_uri, validate_gdrive_path


class RcloneError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RcloneService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def executable(self) -> str:
        configured = Path(self.settings.rclone_bin)
        if configured.is_absolute():
            if configured.is_file():
                return str(configured)
        else:
            found = shutil.which(self.settings.rclone_bin)
            if found:
                return found
        raise RcloneError("rclone_missing", f"rclone executable not found: {self.settings.rclone_bin}")

    def validate_config(self) -> None:
        if not self.settings.rclone_config.is_file():
            raise RcloneError("rclone_config_missing", f"rclone.conf not found: {self.settings.rclone_config}")

    def build_listremotes_command(self) -> List[str]:
        return [self.settings.rclone_bin, "--config", str(self.settings.rclone_config), "listremotes"]

    def build_copy_command(
        self,
        source: str,
        destination: str,
        include: Optional[Iterable[str]] = None,
    ) -> List[str]:
        command = [
            self.settings.rclone_bin,
            "--config",
            str(self.settings.rclone_config),
            "copy",
            source,
            destination,
            "--create-empty-src-dirs",
            "--stats",
            "1s",
            "--stats-one-line",
        ]
        for pattern in include or ():
            command.extend(["--include", pattern])
        return command

    async def check_remote(self) -> dict:
        executable = self.executable()
        self.validate_config()
        command = [executable, "--config", str(self.settings.rclone_config), "listremotes"]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise RcloneError("rclone_missing", f"rclone executable not found: {self.settings.rclone_bin}") from error
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError as error:
            process.kill()
            await process.wait()
            raise RcloneError("gdrive_timeout", "rclone remote check timed out after 30 seconds") from error
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RcloneError("gdrive_access_failed", detail or "rclone could not read the config")
        remotes = [line.strip().rstrip(":") for line in stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]
        if self.settings.gdrive_remote not in remotes:
            raise RcloneError(
                "gdrive_remote_missing",
                f"rclone remote '{self.settings.gdrive_remote}' is not configured",
            )
        return {"available": True, "remotes": remotes, "remote": self.settings.gdrive_remote}

    def dataset_command(self, remote_path: str, local_path: Path) -> List[str]:
        normalized = validate_gdrive_path(remote_path, self.settings.gdrive_remote)
        return self.build_copy_command(
            f"{self.settings.gdrive_remote}:{normalized}", str(local_path)
        )

    def captions_command(self, local_path: Path, remote_path: str) -> List[str]:
        normalized = validate_gdrive_path(remote_path, self.settings.gdrive_remote)
        return self.build_copy_command(
            str(local_path), f"{self.settings.gdrive_remote}:{normalized}", include=["*.txt"]
        )

    def output_command(self, local_path: Path, remote_path: str) -> List[str]:
        normalized = validate_gdrive_path(remote_path, self.settings.gdrive_remote)
        return self.build_copy_command(
            str(local_path),
            f"{self.settings.gdrive_remote}:{normalized}",
            include=["*.safetensors", "*.png", "training_summary.txt", "*.log", "*.json"],
        )

    def public_status(self) -> dict:
        try:
            executable = self.executable()
            executable_state = {"available": True, "path": executable}
        except RcloneError as error:
            executable_state = {"available": False, "error_code": error.code, "message": str(error)}
        return {
            "executable": executable_state,
            "config": {"path": str(self.settings.rclone_config), "exists": self.settings.rclone_config.is_file()},
            "remote": self.settings.gdrive_remote,
        }
