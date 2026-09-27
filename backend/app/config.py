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


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class Settings:
    """Runtime paths and switches.

    The defaults are deliberately different on Windows and Linux.  RunPod uses
    model files live under ``/workspace`` and Factory-generated data defaults
    to ``/data`` on Linux/RunPod. A local Windows checkout gets an isolated
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
    # RunPod-specific locations are optional dataclass fields so the existing
    # Windows tests and callers that construct Settings directly keep working.
    model_dir: Path | None = None
    data_root_base: Path | None = None
    cache_root: Path | None = None
    gdrive_enabled: bool = True
    gdrive_root: str = "SDXL-LoRA-Factory/runs"

    @classmethod
    def from_env(cls) -> "Settings":
        backend_root = Path(__file__).resolve().parents[1]
        repo_root = backend_root.parent
        default_workspace = Path("/workspace") if os.name != "nt" else repo_root / ".runtime"

        workspace_root = Path(os.environ.get("WORKSPACE_ROOT", str(default_workspace))).expanduser()

        # The new Linux layout keeps the shared SDXL checkpoints under
        # /workspace and all Factory-generated data under /data.  Explicit
        # DATA_* variables win, while the legacy LOCAL_* variables remain
        # supported so the Windows launcher and custom deployments keep their
        # original behavior.
        data_root_base_value = os.environ.get("DATA_ROOT")
        if data_root_base_value:
            data_root_base = Path(data_root_base_value).expanduser()
        elif os.name != "nt":
            data_root_base = Path("/data")
        else:
            data_root_base = workspace_root

        legacy_data_root = os.environ.get("LOCAL_DATA_ROOT")
        dataset_root_value = os.environ.get("DATASET_DIR") or legacy_data_root
        if dataset_root_value:
            data_root = Path(dataset_root_value).expanduser()
        elif os.name != "nt" or data_root_base_value:
            data_root = data_root_base / "dataset"
        else:
            data_root = workspace_root / "data"

        output_root = Path(
            os.environ.get("OUTPUT_DIR")
            or os.environ.get("LOCAL_OUTPUT_ROOT")
            or str(data_root_base / "output")
        ).expanduser()
        default_job_root = data_root_base / "jobs" if os.name != "nt" else workspace_root / "jobs"
        job_root = Path(
            os.environ.get("LOG_DIR")
            or os.environ.get("LOCAL_JOB_ROOT")
            or str(default_job_root)
        ).expanduser()
        config_root = Path(
            os.environ.get("CONFIG_DIR")
            or os.environ.get("LOCAL_CONFIG_ROOT")
            or str(data_root_base / "config")
        ).expanduser()
        cache_root = Path(os.environ.get("CACHE_DIR", str(data_root_base / "cache"))).expanduser()
        default_rclone_config = (
            data_root_base / "rclone" / "rclone.conf" if os.name != "nt" else workspace_root / "rclone" / "rclone.conf"
        )
        rclone_config = Path(os.environ.get("RCLONE_CONFIG", str(default_rclone_config))).expanduser()
        model_dir = Path(
            os.environ.get("MODEL_DIR", str(workspace_root / "models" / "checkpoints"))
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
            model_dir=model_dir,
            data_root_base=data_root_base,
            cache_root=cache_root,
            gdrive_enabled=_env_bool("GDRIVE_ENABLED", True),
            gdrive_root=os.environ.get("GDRIVE_ROOT", "SDXL-LoRA-Factory/runs").strip()
            or "SDXL-LoRA-Factory/runs",
        )

    def ensure_runtime_directories(self) -> None:
        self.validate_generated_paths()
        paths = [
            self.data_root_base or self.data_root.parent,
            self.data_root,
            self.output_root,
            self.job_root,
            self.config_root,
            self.rclone_config.parent,
        ]
        # On RunPod, /workspace is an already-mounted shared volume.  Do not
        # create or initialize it; Factory only reads MODEL_DIR below it.
        if os.name == "nt":
            paths.append(self.workspace_root)
        if self.cache_root is not None:
            paths.append(self.cache_root)
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def validate_generated_paths(self) -> None:
        """Reject a Linux layout that would write Factory data into MODEL_DIR."""

        if os.name == "nt":
            return
        generated_paths = (
            self.data_root_base or self.data_root.parent,
            self.data_root,
            self.output_root,
            self.job_root,
            self.config_root,
            self.cache_directory,
            self.rclone_config.parent,
        )
        for path in generated_paths:
            self.validate_generated_path(path)

    def validate_generated_path(self, path: Path, label: str = "Factory data path") -> Path:
        """Validate one generated-data path before any caller creates it."""

        candidate = path.expanduser().resolve()
        if os.name == "nt":
            return candidate
        model_root = self.model_directory
        if _path_is_within(candidate, model_root):
            raise ValueError(
                f"{label} must not be inside MODEL_DIR ({model_root}); received {candidate}"
            )
        if _path_is_within(candidate, self.workspace_root):
            raise ValueError(
                f"{label} must not be inside WORKSPACE_ROOT ({self.workspace_root}); "
                f"RunPod Factory data belongs under DATA_ROOT: {candidate}"
            )
        return candidate

    @property
    def model_directory(self) -> Path:
        """Return the read-only base-model directory.

        Settings constructed by older callers do not provide ``model_dir``;
        retain the natural workspace fallback for those callers.
        """

        return (self.model_dir or (self.workspace_root / "models" / "checkpoints")).expanduser()

    @property
    def cache_directory(self) -> Path:
        return (self.cache_root or (self.workspace_root / "cache")).expanduser()

    @property
    def run_backup_root(self) -> Path:
        """Temporary/local staging area for a structured Google Drive run."""

        return self.config_root / "backups"

    def public_dict(self) -> dict:
        return {
            "platform": sys.platform,
            "app_port": self.app_port,
            "workspace_root": str(self.workspace_root),
            "data_root": str(self.data_root),
            "data_root_base": str(self.data_root_base or self.data_root.parent),
            "dataset_dir": str(self.data_root),
            "output_root": str(self.output_root),
            "job_root": str(self.job_root),
            "log_dir": str(self.job_root),
            "config_dir": str(self.config_root),
            "rclone_config": str(self.rclone_config),
            "gdrive_remote": self.gdrive_remote,
            "gdrive_enabled": self.gdrive_enabled,
            "gdrive_root": self.gdrive_root,
            "model_dir": str(self.model_directory),
            "cache_dir": str(self.cache_directory),
            "sd_scripts_root": str(self.sd_scripts_root),
        }
