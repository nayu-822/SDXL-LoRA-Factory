from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime paths and switches.

    The defaults are deliberately different on Windows and Linux.  RunPod uses
    ``/workspace`` by default, while a local Windows checkout gets an isolated
    ``.runtime`` directory so the original UI remains usable without extra
    configuration.
    """

    repo_root: Path
    backend_root: Path
    frontend_root: Path
    workspace_root: Path
    data_root: Path
    output_root: Path
    job_root: Path
    config_root: Path
    rclone_config: Path
    gdrive_remote: str
    rclone_bin: str
    app_port: int
    sd_scripts_root: Path
    allow_external_dataset_paths: bool
    allow_mock_tagger: bool

    @classmethod
    def from_env(cls) -> "Settings":
        backend_root = Path(__file__).resolve().parents[1]
        repo_root = backend_root.parent
        default_workspace = Path("/workspace") if os.name != "nt" else repo_root / ".runtime"

        workspace_root = Path(os.environ.get("WORKSPACE_ROOT", str(default_workspace))).expanduser()
        data_root = Path(os.environ.get("LOCAL_DATA_ROOT", str(workspace_root / "data"))).expanduser()
        output_root = Path(os.environ.get("LOCAL_OUTPUT_ROOT", str(workspace_root / "output"))).expanduser()
        job_root = Path(os.environ.get("LOCAL_JOB_ROOT", str(workspace_root / "jobs"))).expanduser()
        config_root = Path(os.environ.get("LOCAL_CONFIG_ROOT", str(workspace_root / "config"))).expanduser()
        rclone_config = Path(
            os.environ.get("RCLONE_CONFIG", str(workspace_root / "rclone" / "rclone.conf"))
        ).expanduser()

        return cls(
            repo_root=repo_root,
            backend_root=backend_root,
            frontend_root=repo_root / "frontend",
            workspace_root=workspace_root,
            data_root=data_root,
            output_root=output_root,
            job_root=job_root,
            config_root=config_root,
            rclone_config=rclone_config,
            gdrive_remote=os.environ.get("GDRIVE_REMOTE", "gdrive").strip() or "gdrive",
            rclone_bin=os.environ.get("RCLONE_BIN", "rclone").strip() or "rclone",
            app_port=int(os.environ.get("APP_PORT", "8001")),
            sd_scripts_root=Path(
                os.environ.get("SD_SCRIPTS_ROOT", str(backend_root / "sd-scripts"))
            ).expanduser(),
            allow_external_dataset_paths=_env_bool("ALLOW_EXTERNAL_DATASET_PATHS", os.name == "nt"),
            allow_mock_tagger=_env_bool("WD14_ALLOW_MOCK", False),
        )

    def ensure_runtime_directories(self) -> None:
        for path in (self.workspace_root, self.data_root, self.output_root, self.job_root, self.config_root):
            path.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        return {
            "platform": sys.platform,
            "app_port": self.app_port,
            "workspace_root": str(self.workspace_root),
            "data_root": str(self.data_root),
            "output_root": str(self.output_root),
            "job_root": str(self.job_root),
            "rclone_config": str(self.rclone_config),
            "gdrive_remote": self.gdrive_remote,
            "sd_scripts_root": str(self.sd_scripts_root),
        }
