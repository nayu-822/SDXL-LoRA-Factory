from __future__ import annotations

import asyncio
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .config import Settings
from .events import EventHub
from .loss_parser import LossParser
from .models import OutputSyncRequest, TrainingConfig
from .paths import (
    dataset_counts,
    image_files,
    safe_slug,
    validate_local_dataset,
    validate_model_reference,
    validate_optional_model_path,
    validate_output_path,
)
from .processes import ProcessManager, ProcessRun
from .summary import write_training_summary
from .system import gpu_info, sd_scripts_version, torch_info


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _number(value) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def build_gdrive_run_path(config: TrainingConfig, settings: Settings, run_id: str) -> str:
    """Resolve the Drive folder for one run without embedding credentials."""

    configured = config.gdrive_output_path.strip()
    if configured:
        if os.name == "nt":
            # Preserve the existing Windows output-sync destination.  RunPod
            # always receives a unique suffix below.
            return configured.replace("{RUN_ID}", run_id).replace("{run_id}", run_id)
        if "{RUN_ID}" in configured or "{run_id}" in configured:
            return configured.replace("{RUN_ID}", run_id).replace("{run_id}", run_id)
        return f"{configured.rstrip('/')}/{run_id}"
    return f"{settings.gdrive_root.rstrip('/')}/{run_id}"


def build_dataset_toml(image_dir: Path, repeats: int, width: int, height: int, min_bucket: int, max_bucket: int) -> str:
    """Create the exact dataset config consumed by bundled sd-scripts."""
    try:
        import toml

        return toml.dumps(
            {
                "general": {
                    "enable_bucket": True,
                    "resolution": [width, height],
                    "min_bucket_reso": min_bucket,
                    "max_bucket_reso": max_bucket,
                    "caption_extension": ".txt",
                },
                "datasets": [{"subsets": [{"image_dir": str(image_dir), "num_repeats": repeats}]}],
            }
        )
    except ImportError:
        # toml is in requirements, but keeping this fallback makes syntax/unit tests lightweight.
        quoted = json.dumps(str(image_dir), ensure_ascii=False)
        return (
            "[general]\n"
            "enable_bucket = true\n"
            f"resolution = [{width}, {height}]\n"
            f"min_bucket_reso = {min_bucket}\n"
            f"max_bucket_reso = {max_bucket}\n"
            'caption_extension = ".txt"\n\n'
            "[[datasets]]\n\n[[datasets.subsets]]\n"
            f"image_dir = {quoted}\n"
            f"num_repeats = {repeats}\n"
        )


def build_sample_prompts(config: TrainingConfig) -> list[dict]:
    prompts = [item.strip() for item in config.sample_prompts if item and item.strip()]
    if config.sample_prompt.strip() and config.sample_prompt.strip() not in prompts:
        prompts.insert(0, config.sample_prompt.strip())
    if not prompts:
        prompts = ["a high quality portrait of a character"]
    trigger_word = config.trigger_word.strip()
    if trigger_word:
        normalized_trigger = trigger_word.casefold()
        normalized_prompts = []
        for prompt in prompts:
            tags = [tag.strip().casefold() for tag in prompt.split(",")]
            if tags and tags[0] == normalized_trigger:
                normalized_prompts.append(prompt)
            elif normalized_trigger in tags:
                # Avoid duplicating a trigger that is already present later in
                # a manually authored prompt as well.
                normalized_prompts.append(prompt)
            else:
                normalized_prompts.append(f"{trigger_word}, {prompt}")
        prompts = normalized_prompts
    return [
        {
            "prompt": prompt,
            "negative_prompt": config.sample_negative_prompt,
            "seed": config.sample_seed,
            "sample_steps": config.sample_steps,
            "sample_sampler": config.sample_sampler,
            "width": config.sample_width,
            "height": config.sample_height,
            "scale": config.sample_scale,
        }
        for prompt in prompts
    ]


def build_training_command(
    config: TrainingConfig,
    script_path: Path,
    dataset_config_path: Path,
    output_dir: Path,
    sample_prompts_path: Path,
    logging_dir: Path,
    seed: int,
) -> list[str]:
    mixed_precision = config.mixed_precision.lower().strip()
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be no, fp16, or bf16")
    sampler_choices = {
        "ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver",
        "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a",
    }
    if config.sample_sampler not in sampler_choices:
        raise ValueError(f"unsupported sample sampler: {config.sample_sampler}")

    command = [
        sys.executable,
        str(script_path),
        f"--pretrained_model_name_or_path={config.model}",
        f"--output_dir={output_dir}",
        f"--output_name={safe_slug(config.name, 'sdxl_lora')}",
        f"--dataset_config={dataset_config_path}",
        "--network_module=networks.lora",
        f"--max_train_epochs={config.epochs}",
        f"--learning_rate={_number(config.lr)}",
        f"--network_dim={config.rank}",
        f"--network_alpha={_number(config.alpha)}",
        f"--train_batch_size={config.batch_size}",
        f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
        f"--mixed_precision={mixed_precision}",
        "--save_model_as=safetensors",
        "--save_every_n_epochs=1",
        "--sample_prompts=" + str(sample_prompts_path),
        f"--sample_sampler={config.sample_sampler}",
        f"--logging_dir={logging_dir}",
        "--log_with=tensorboard",
        f"--log_prefix={safe_slug(config.name, 'sdxl_lora')}-",
        f"--max_data_loader_n_workers={config.workers}",
        f"--seed={seed}",
        "--sdpa",
        "--cache_latents",
    ]
    if config.sample_at_first:
        command.append("--sample_at_first")
    if config.sample_every_n_epochs > 0:
        command.append(f"--sample_every_n_epochs={config.sample_every_n_epochs}")
    if config.vae.strip():
        command.append(f"--vae={config.vae}")
    if config.unet_lr is not None:
        command.append(f"--unet_lr={_number(config.unet_lr)}")
    if config.text_encoder_lr is not None:
        command.append(f"--text_encoder_lr={_number(config.text_encoder_lr)}")
    if config.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if config.flip_aug:
        command.append("--flip_aug")
    if config.shuffle_caption:
        command.extend(["--shuffle_caption", f"--keep_tokens={config.keep_tokens}"])
    if config.min_snr_gamma:
        command.append(f"--min_snr_gamma={_number(config.min_snr_gamma_value)}")
    if config.optimizer_type.strip() and config.optimizer_type.lower() != "adamw":
        command.append(f"--optimizer_type={config.optimizer_type.strip()}")
        if config.optimizer_args.strip():
            command.append("--optimizer_args")
            command.extend(shlex.split(config.optimizer_args.strip()))
    if config.vram == "high":
        command.append("--highvram")
    elif config.vram in {"low", "very_low"} or (
        config.optimizer_type.lower() != "adamw" and config.optimizer_low_vram
    ):
        command.extend(["--lowram", "--cache_latents_to_disk"])
    return command


class TrainingService:
    def __init__(self, settings: Settings, manager: ProcessManager, hub: EventHub, sync_service=None):
        self.settings = settings
        self.manager = manager
        self.hub = hub
        self.sync_service = sync_service
        self.states: dict[str, dict] = {}

    async def start(self, config: TrainingConfig) -> Tuple[dict, ProcessRun]:
        dataset_path = validate_local_dataset(config.path, self.settings)
        counts = dataset_counts(dataset_path)
        if counts["image_count"] == 0:
            raise ValueError("dataset is empty: no supported images were found")
        config.model = validate_model_reference(config.model, self.settings)
        config.vae = validate_optional_model_path(config.vae, self.settings)

        script_path = self.settings.sd_scripts_root / "sdxl_train_network.py"
        if not script_path.is_file():
            raise ValueError(f"sd-scripts entrypoint not found: {script_path}")

        name = safe_slug(config.name, "sdxl_lora")
        if config.seed is None:
            config.seed = random.randint(0, 2**32 - 1)
        job_id = f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        # Config/manifest files and process logs live in their dedicated
        # volatile /data trees.  Keep output files separate so the Drive
        # backup can reproduce the run as lora/captions/config/logs.
        job_dir = self.settings.config_root / job_id
        output_parent = (
            validate_output_path(config.output_dir, self.settings)
            if config.output_dir.strip()
            else validate_output_path(str(self.settings.output_root), self.settings)
        )
        output_dir = output_parent / job_id
        lora_dir = output_dir / "lora"
        samples_dir = output_dir / "samples"
        logs_dir = self.settings.job_root / job_id
        for path in (job_dir, lora_dir, samples_dir, logs_dir):
            path.mkdir(parents=True, exist_ok=True)

        dataset_config_path = job_dir / "dataset_config.toml"
        dataset_config_path.write_text(
            build_dataset_toml(
                dataset_path,
                config.repeats,
                config.resolution_width,
                config.resolution_height,
                config.min_bucket_reso,
                config.max_bucket_reso,
            ),
            encoding="utf-8",
        )
        sample_prompts_path = job_dir / "sample_prompts.json"
        sample_prompts_path.write_text(
            json.dumps(build_sample_prompts(config), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        command = build_training_command(
            config,
            script_path,
            dataset_config_path,
            lora_dir,
            sample_prompts_path,
            logs_dir / "tensorboard",
            config.seed,
        )
        (job_dir / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
        (job_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        (job_dir / "training_config.json").write_text(
            json.dumps(_model_dict(config), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

        gdrive_output_path = build_gdrive_run_path(config, self.settings, job_id)
        # The structured run backup is the RunPod/Linux addition.  Windows
        # keeps the existing auto_sync_output flow below, even though the
        # shared frontend sends both compatibility flags.
        backup_requested = bool(config.gdrive_backup_enabled) and os.name != "nt"
        state = {
            "job_id": job_id,
            "job_name": name,
            "status": "queued",
            "start_time": _now(),
            "end_time": "",
            "duration": "",
            "dataset_path": str(dataset_path),
            "gdrive_dataset_path": config.gdrive_dataset_path,
            "output_dir": str(output_dir),
            "lora_dir": str(lora_dir),
            "samples_dir": str(samples_dir),
            "logs_dir": str(logs_dir),
            "job_dir": str(job_dir),
            "gdrive_output_path": gdrive_output_path,
            "gdrive_run_path": gdrive_output_path,
            "gdrive_backup": {
                "enabled": backup_requested,
                "status": "pending" if backup_requested else "not_requested",
                "gdrive_path": gdrive_output_path,
            },
            "config": _model_dict(config),
            "command": command,
            "progress": LossParser().progress(),
            "loss_history": {"steps": [], "epochs": []},
            "warnings": [],
            "errors": [],
            "output_files": [],
            "_started_monotonic": asyncio.get_running_loop().time(),
        }
        self.states[job_id] = state
        self._persist(state)
        if backup_requested and self.sync_service is not None:
            try:
                backup_dir = self.sync_service.prepare_run_backup(
                    job_id,
                    dataset_path,
                    job_dir,
                    output_dir,
                    logs_dir,
                )
                state["backup_dir"] = str(backup_dir)
                self._persist(state)
                await self._run_structured_backup(state, "start")
            except Exception as error:
                # Drive is a backup destination, never a prerequisite for
                # starting the local training process.
                state["gdrive_backup"] = {
                    "enabled": True,
                    "status": "failed",
                    "phase": "start",
                    "gdrive_path": gdrive_output_path,
                    "error": str(error),
                }
                state["warnings"].append(f"Google Drive start backup failed: {error}")
                self._persist(state)
        parser = LossParser()

        async def on_line(run: ProcessRun, line: str) -> None:
            state["run"] = run.public_dict()
            progress = parser.on_line(line)
            state["progress"] = progress
            state["loss_history"] = parser.as_dict()
            low = line.lower()
            if "warning" in low and line not in state["warnings"]:
                state["warnings"].append(line)
            if any(
                marker in low
                for marker in ("traceback", "cuda out of memory", "modulenotfounderror", "runtimeerror", "exception", "error")
            ):
                if line not in state["errors"]:
                    state["errors"].append(line)
            await self.hub.log("train", line, job_id)
            await self.hub.progress("train", progress, job_id)
            self._persist(state)

        async def on_done(run: ProcessRun) -> None:
            state["run"] = run.public_dict()
            try:
                parser.merge_tensorboard(logs_dir / "tensorboard")
            except Exception as error:
                state["warnings"].append(f"TensorBoard loss parsing failed: {error}")
            state["loss_history"] = parser.as_dict()
            (logs_dir / "loss_history.json").write_text(
                json.dumps(state["loss_history"], ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            if run.error and run.error not in state["errors"]:
                state["errors"].append(run.error)
            if run.status == "succeeded":
                state["status"] = "completed"
            elif run.status == "cancelled":
                state["status"] = "cancelled"
                state["errors"].append("training cancelled")
            else:
                state["status"] = "failed"
                if run.returncode is not None:
                    state["errors"].append(f"sd-scripts exited with code {run.returncode}")

            try:
                postprocess = await asyncio.to_thread(self._postprocess, state, config, state["status"] == "completed")
                state["output_files"] = postprocess["output_files"]
                state["final_lora"] = postprocess.get("final_lora", "")
            except Exception as error:
                state["errors"].append(f"output post-processing failed: {error}")
                if state["status"] == "completed":
                    state["status"] = "failed"

            state["end_time"] = run.ended_at or _now()
            elapsed = asyncio.get_running_loop().time() - state["_started_monotonic"]
            state["duration"] = f"{elapsed:.1f}s"
            summary_path = Path(state["output_dir"]) / "training_summary.txt"
            state["output_sync"] = {
                "enabled": bool(config.auto_sync_output),
                "gdrive_output_path": gdrive_output_path,
                "status": "pending" if config.auto_sync_output else "not_requested",
            }
            if "training_summary.txt" not in state["output_files"]:
                state["output_files"].append("training_summary.txt")
            state["summary_path"] = str(summary_path)

            def write_summary() -> None:
                summary_data = self._summary_data(state, config, dataset_path, counts, parser)
                # Include the summary itself in the manifest written into the summary.
                if "training_summary.txt" not in summary_data["output_files"]:
                    summary_data["output_files"].append("training_summary.txt")
                write_training_summary(summary_path, summary_data)

            # Write a provisional summary before an optional output transfer.
            write_summary()
            self._persist(state)
            await self.hub.status("train", self._public_state(state), job_id)

            if backup_requested:
                try:
                    if self.sync_service is None:
                        raise RuntimeError("Google Drive backup service is unavailable")
                    backup_dir = self.sync_service.prepare_run_backup(
                        job_id,
                        dataset_path,
                        job_dir,
                        Path(state["output_dir"]),
                        logs_dir,
                    )
                    state["backup_dir"] = str(backup_dir)
                    await self._run_structured_backup(state, "finish")
                    if state["gdrive_backup"].get("status") != "completed":
                        state["warnings"].append("Google Drive run backup failed")
                except Exception as error:
                    state["gdrive_backup"] = {
                        "enabled": True,
                        "status": "failed",
                        "phase": "finish",
                        "gdrive_path": gdrive_output_path,
                        "error": str(error),
                    }
                    state["warnings"].append(f"Google Drive run backup failed: {error}")
                # The final summary records whether training and backup each
                # succeeded, so the two outcomes are never conflated. Write
                # it before copying the summary-only refresh to Drive.
                write_summary()
                backup_dir = Path(state.get("backup_dir", ""))
                if backup_dir.is_dir() and self.sync_service is not None:
                    try:
                        staged_summary = backup_dir / "logs" / "training_summary.txt"
                        staged_summary.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(summary_path, staged_summary)
                        summary_state, summary_run = await self.sync_service.start_run_backup_file(
                            staged_summary,
                            f"{gdrive_output_path.rstrip('/')}/logs",
                        )
                        await self.manager.wait(summary_run)
                        if summary_state.get("status") != "completed":
                            state["warnings"].append("Google Drive final summary copy failed")
                    except AttributeError:
                        # Third-party/test sync adapters from the old API may
                        # not provide the summary-only method.
                        pass
                    except Exception as error:
                        state["warnings"].append(f"Google Drive final summary copy failed: {error}")
                self._persist(state)
                await self.hub.status("train", self._public_state(state), job_id)
            elif config.auto_sync_output:
                try:
                    if self.sync_service is None:
                        raise RuntimeError("output sync service is unavailable")
                    sync_state, sync_run = await self.sync_service.start_output(
                        OutputSyncRequest(local_output_path=state["output_dir"], gdrive_path=gdrive_output_path)
                    )
                    await self.manager.wait(sync_run)
                    state["output_sync"] = {
                        "enabled": True,
                        "gdrive_output_path": gdrive_output_path,
                        **sync_state,
                    }
                    if sync_state.get("status") != "completed":
                        state["warnings"].append("automatic output sync failed")
                except Exception as error:
                    state["output_sync"] = {
                        "enabled": True,
                        "gdrive_output_path": gdrive_output_path,
                        "status": "failed",
                        "error": str(error),
                    }
                    state["warnings"].append(f"automatic output sync failed: {error}")
                # Replace the provisional summary so the final sync result is
                # available to both the UI and downstream diagnostics, then
                # copy that final summary once more to Drive.
                write_summary()
                if self.sync_service is not None:
                    try:
                        summary_state, summary_run = await self.sync_service.start_output_file(
                            summary_path,
                            gdrive_output_path,
                        )
                        await self.manager.wait(summary_run)
                        if summary_state.get("status") != "completed":
                            state["warnings"].append("Google Drive final summary copy failed")
                    except AttributeError:
                        pass
                    except Exception as error:
                        state["warnings"].append(f"Google Drive final summary copy failed: {error}")
                write_summary()
                self._persist(state)
                await self.hub.status("train", self._public_state(state), job_id)

            if config.shutdown:
                if os.name == "nt":
                    await self.hub.log("train", "shutdown requested; scheduling Windows shutdown in 60 seconds", job_id)
                    subprocess.run(["shutdown", "/s", "/t", "60"], check=False)
                else:
                    await self.hub.log("train", "shutdown option ignored on Linux/RunPod", job_id)

        env = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(self.settings.sd_scripts_root)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        run = await self.manager.start(
            kind="training",
            run_id=job_id,
            command=command,
            cwd=self.settings.sd_scripts_root,
            log_path=logs_dir / "training.log",
            on_line=on_line,
            on_done=on_done,
            env=env,
        )
        state["status"] = "running"
        state["run"] = run.public_dict()
        self._persist(state)
        await self.hub.status("train", self._public_state(state), job_id)
        return state, run

    async def _run_structured_backup(self, state: dict, phase: str) -> dict:
        """Run one non-destructive Drive copy and update its separate status."""

        if self.sync_service is None:
            raise RuntimeError("Google Drive backup service is unavailable")
        backup_dir = Path(state.get("backup_dir", ""))
        if not backup_dir.is_dir():
            raise RuntimeError(f"run backup staging directory does not exist: {backup_dir}")
        sync_state, sync_run = await self.sync_service.start_run_backup(
            backup_dir,
            state["gdrive_run_path"],
        )
        await self.manager.wait(sync_run)
        state["gdrive_backup"] = {
            "enabled": True,
            "phase": phase,
            "gdrive_path": state["gdrive_run_path"],
            **sync_state,
        }
        self._persist(state)
        await self.hub.status("train", self._public_state(state), state["job_id"])
        return state["gdrive_backup"]

    def _postprocess(self, state: dict, config: TrainingConfig, success: bool) -> dict:
        output_dir = Path(state["output_dir"])
        lora_dir = Path(state["lora_dir"])
        samples_dir = Path(state["samples_dir"])
        sample_source = lora_dir / "sample"
        if sample_source.is_dir():
            samples_dir.mkdir(parents=True, exist_ok=True)
            for item in list(sample_source.iterdir()):
                destination = samples_dir / item.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                shutil.move(str(item), str(destination))
            try:
                sample_source.rmdir()
            except OSError:
                pass

        if success:
            final_lora = lora_dir / f"{safe_slug(config.name, 'sdxl_lora')}.safetensors"
            last_epoch = lora_dir / f"{safe_slug(config.name, 'sdxl_lora')}-{config.epochs:06d}.safetensors"
            if final_lora.is_file() and not last_epoch.exists():
                shutil.copy2(final_lora, last_epoch)
        else:
            final_lora = lora_dir / f"{safe_slug(config.name, 'sdxl_lora')}.safetensors"

        output_files = []
        for path in sorted(output_dir.rglob("*"), key=lambda item: str(item).lower()):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".safetensors", ".png", ".log", ".json"} or path.name == "training_summary.txt":
                output_files.append(str(path.relative_to(output_dir)).replace("\\", "/"))
        return {
            "output_files": output_files,
            "final_lora": str(final_lora) if final_lora.is_file() else "",
        }

    @staticmethod
    def _loss_diagnostics(state: dict, config: TrainingConfig, counts: dict, parser: LossParser) -> dict:
        image_count = int(counts.get("image_count", 0) or 0)
        images_x_repeats = image_count * config.repeats if image_count else None
        effective_images_per_epoch = images_x_repeats
        steps_per_epoch_estimated = (
            math.ceil(images_x_repeats / (config.batch_size * config.gradient_accumulation_steps))
            if images_x_repeats is not None
            else None
        )

        total_training_steps = parser.total_steps if parser.total_steps and parser.total_steps > 0 else None
        total_training_steps_source = "observed" if total_training_steps is not None else ""
        if total_training_steps is None and state.get("status") == "completed" and parser.step:
            total_training_steps = parser.step
            total_training_steps_source = "observed"

        epoch_losses = []
        for epoch, record in parser.epochs.items():
            value = record.get("average_loss")
            if value is None:
                continue
            try:
                epoch_losses.append((int(epoch), float(value)))
            except (TypeError, ValueError):
                continue
        lowest_epoch_loss = min((value for _, value in epoch_losses), default=None)
        lowest_loss_epoch = next(
            (epoch for epoch, value in sorted(epoch_losses) if value == lowest_epoch_loss),
            None,
        )
        final_loss = parser.loss if parser.loss is not None else parser.average_loss
        return {
            "total_training_steps": total_training_steps,
            "total_training_steps_source": total_training_steps_source,
            "effective_images_per_epoch": effective_images_per_epoch,
            "images_x_repeats": images_x_repeats,
            "steps_per_epoch_estimated": steps_per_epoch_estimated,
            "estimated_total_steps": (
                steps_per_epoch_estimated * config.epochs if steps_per_epoch_estimated is not None else None
            ),
            "lowest_epoch_loss": lowest_epoch_loss,
            "lowest_loss_epoch": lowest_loss_epoch,
            "final_loss": final_loss,
        }

    def _summary_data(self, state: dict, config: TrainingConfig, dataset_path: Path, counts: dict, parser: LossParser) -> dict:
        actual_config = _model_dict(config)
        hardware = gpu_info()
        training_config = {
            "base_model": config.model,
            "vae": config.vae,
            "training_type": config.training_type,
            "trigger_word": config.trigger_word,
            "resolution": f"{config.resolution_width}x{config.resolution_height}",
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "repeats": config.repeats,
            "optimizer": config.optimizer_type,
            "learning_rate": config.lr,
            "unet_lr": config.unet_lr,
            "text_encoder_lr": config.text_encoder_lr,
            "network_dim": config.rank,
            "network_alpha": config.alpha,
            "min_snr_gamma": config.min_snr_gamma_value if config.min_snr_gamma else "disabled",
            "gradient_checkpointing": config.gradient_checkpointing,
            "mixed_precision": config.mixed_precision,
            "seed": config.seed,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "workers": config.workers,
            "save_every_n_epochs": 1,
            "actual_config": json.dumps(actual_config, ensure_ascii=False, sort_keys=True),
        }
        loss_diagnostics = self._loss_diagnostics(state, config, counts, parser)
        return {
            "job_id": state["job_id"],
            "job_name": state["job_name"],
            "start_time": state["start_time"],
            "end_time": state["end_time"],
            "status": state["status"],
            "duration": state["duration"],
            "environment": {
                "gpu": hardware.get("name", ""),
                "vram": hardware.get("memory", ""),
                **hardware,
                **torch_info(),
                "python_version": sys.version.split()[0],
                "sd_scripts_version_or_commit": sd_scripts_version(self.settings.sd_scripts_root),
            },
            "dataset": {
                "gdrive_dataset_path": config.gdrive_dataset_path,
                "local_dataset_path": str(dataset_path),
                **counts,
            },
            "caption": {
                "trigger_words": config.trigger_word,
                "shuffle_caption": config.shuffle_caption,
                "keep_tokens": config.keep_tokens,
            },
            "training_config": training_config,
            "command": shlex.join(state["command"]),
            "loss_history": parser.as_dict(),
            "loss_diagnostics": loss_diagnostics,
            "sample_settings": {
                "sample_prompt": " | ".join(item["prompt"] for item in build_sample_prompts(config)),
                "negative_prompt": config.sample_negative_prompt,
                "seed": config.sample_seed,
                "sampler": config.sample_sampler,
                "steps": config.sample_steps,
                "width": config.sample_width,
                "height": config.sample_height,
                "every_epoch": config.sample_every_n_epochs,
                "sample_at_first": config.sample_at_first,
            },
            "output_files": list(state.get("output_files", [])),
            "output_sync": dict(state.get("output_sync", {})),
            "gdrive_backup": dict(state.get("gdrive_backup", {})),
            "paths": {
                "dataset_dir": str(self.settings.data_root),
                "config_dir": str(state.get("job_dir", "")),
                "output_dir": str(state.get("output_dir", "")),
                "log_dir": str(state.get("logs_dir", "")),
                "cache_dir": str(self.settings.cache_directory),
                "model_dir": str(self.settings.model_directory),
            },
            "step_loss_summary_limit": 800,
            "warnings": list(dict.fromkeys(state.get("warnings", []))),
            "errors": list(dict.fromkeys(state.get("errors", []))),
        }

    @staticmethod
    def _public_state(state: dict) -> dict:
        return {key: value for key, value in state.items() if not key.startswith("_")}

    def _persist(self, state: dict) -> None:
        path = Path(state["job_dir"]) / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._public_state(state), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def public_status(self) -> dict:
        return {key: self._public_state(value) for key, value in self.states.items()}

    def get(self, job_id: str) -> Optional[dict]:
        return self._public_state(self.states[job_id]) if job_id in self.states else None
