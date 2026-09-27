from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:  # supports both `python backend/main.py` and `python -m backend.main`
    from .app.config import Settings
    from .app.events import EventHub
    from .app.models import (
        BatchTagUpdate,
        CaptionSyncRequest,
        DatasetSyncRequest,
        OutputSyncRequest,
        RecommendationRequest,
        TagUpdate,
        TaggerConfig,
        TrainingConfig,
        TriggerWordUpdate,
    )
    from .app.paths import (
        dataset_counts,
        list_base_models,
        validate_image_path,
        validate_local_dataset,
    )
    from .app.processes import ProcessManager, ProcessRun
    from .app.rclone_sync import RcloneError, RcloneService
    from .app.recommendations import recommend_settings
    from .app.sync import SyncService
    from .app.system import dependency_info, executable_available, gpu_info, torch_info
    from .app.tag_ops import (
        batch_remove_tags,
        batch_update_tags,
        list_image_records,
        update_image_tags,
    )
    from .app.tagging import TaggerService
    from .app.training import TrainingService
except ImportError:  # pragma: no cover - used when the file is run directly
    from app.config import Settings
    from app.events import EventHub
    from app.models import (
        BatchTagUpdate,
        CaptionSyncRequest,
        DatasetSyncRequest,
        OutputSyncRequest,
        RecommendationRequest,
        TagUpdate,
        TaggerConfig,
        TrainingConfig,
        TriggerWordUpdate,
    )
    from app.paths import dataset_counts, list_base_models, validate_image_path, validate_local_dataset
    from app.processes import ProcessManager, ProcessRun
    from app.rclone_sync import RcloneError, RcloneService
    from app.recommendations import recommend_settings
    from app.sync import SyncService
    from app.system import dependency_info, executable_available, gpu_info, torch_info
    from app.tag_ops import batch_remove_tags, batch_update_tags, list_image_records, update_image_tags
    from app.tagging import TaggerService
    from app.training import TrainingService


settings = Settings.from_env()
hub = EventHub()
processes = ProcessManager()
rclone = RcloneService(settings)
sync_service = SyncService(settings, rclone, processes, hub)
tagger_service = TaggerService(settings, processes, hub, sync_service)
training_service = TrainingService(settings, processes, hub, sync_service)

app = FastAPI(title="SDXL LoRA Factory", version="5.2-runpod")


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, RcloneError):
        return HTTPException(status_code=400, detail={"code": error.code, "message": str(error)})
    return HTTPException(status_code=400, detail={"code": "invalid_request", "message": str(error)})


@app.on_event("startup")
async def startup() -> None:
    settings.ensure_runtime_directories()
    # Missing Drive credentials must not prevent the web UI from starting.
    # When a RunPod Secret is provided, materialize it into the private config
    # path before the first status check without ever logging its contents.
    if settings.gdrive_enabled:
        try:
            rclone.ensure_config_from_environment()
        except Exception:
            # The detailed error is reported by /api/system/status when the user
            # explicitly checks the Drive connection.
            pass
    await hub.status("system", {"status": "ready", **settings.public_dict()}, "")


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await hub.connect(websocket)
    try:
        while True:
            # The browser only listens, but accepting pings/close frames keeps
            # the connection alive through RunPod Proxy.
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        await hub.disconnect(websocket)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "sdxl-lora-factory", "platform": sys.platform}


@app.get("/api/system/status")
async def system_status():
    remote = None
    remote_error = None
    if settings.gdrive_enabled:
        try:
            remote = await rclone.check_remote()
        except Exception as error:
            remote_error = {"code": getattr(error, "code", "gdrive_unavailable"), "message": str(error)}
    else:
        remote = {"available": False, "disabled": True}
    return {
        "settings": settings.public_dict(),
        "rclone": rclone.public_status(),
        "gdrive": {"remote": remote, "error": remote_error},
        "models": list_base_models(settings),
        "gpu": await asyncio.to_thread(gpu_info),
        "torch": await asyncio.to_thread(torch_info),
        "dependencies": await asyncio.to_thread(dependency_info),
        "processes": processes.public_status(),
    }


@app.get("/api/gpu-info")
async def get_gpu_info():
    return await asyncio.to_thread(gpu_info)


@app.get("/api/check-scripts")
async def check_scripts():
    entrypoint = settings.sd_scripts_root / "sdxl_train_network.py"
    tagger = settings.backend_root / "bundled_tagger.py"
    return {
        "exists": entrypoint.is_file(),
        "path": str(settings.sd_scripts_root) if entrypoint.is_file() else "",
        "entrypoint": str(entrypoint),
        "tagger_exists": tagger.is_file(),
    }


@app.get("/api/base-models")
@app.get("/api/models")
async def base_models():
    """List read-only SDXL checkpoints available on the attached volume."""

    return await asyncio.to_thread(list_base_models, settings)


def _dialog(kind: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        if kind == "file":
            return filedialog.askopenfilename(
                title="Select file",
                filetypes=[("Safetensors / checkpoints", "*.safetensors *.ckpt *.pt"), ("All files", "*.*")],
            )
        return filedialog.askdirectory(title="Select folder")
    finally:
        root.destroy()


@app.get("/api/browse-folder")
async def browse_folder():
    if os.name != "nt":
        raise HTTPException(status_code=501, detail="RunPod/Linux has no desktop picker; enter the path manually")
    try:
        return {"path": await asyncio.to_thread(_dialog, "folder")}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/browse-file")
async def browse_file():
    if os.name != "nt":
        raise HTTPException(status_code=501, detail="RunPod/Linux has no desktop picker; enter the path manually")
    try:
        return {"path": await asyncio.to_thread(_dialog, "file")}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/validate-dataset")
async def validate_dataset(path: str):
    try:
        local_path = validate_local_dataset(path, settings)
        return {"valid": True, "path": str(local_path), **dataset_counts(local_path)}
    except Exception as error:
        raise _http_error(error)


@app.post("/api/dataset/sync")
@app.post("/api/sync-dataset")
async def sync_dataset(request: DatasetSyncRequest):
    try:
        state, _ = await sync_service.start_dataset(request)
        return {"status": "started", **state}
    except Exception as error:
        raise _http_error(error)


@app.post("/api/dataset/sync-captions")
@app.post("/api/sync-captions")
async def sync_captions(request: CaptionSyncRequest):
    try:
        state, _ = await sync_service.start_captions(request)
        return {"status": "started", **state}
    except Exception as error:
        raise _http_error(error)


@app.get("/api/dataset/images")
async def list_images(path: str):
    try:
        local_path = validate_local_dataset(path, settings)
        return {"path": str(local_path), "files": list_image_records(str(local_path), settings), **dataset_counts(local_path)}
    except Exception as error:
        raise _http_error(error)


@app.post("/api/dataset/update-tags")
async def update_tags(update: TagUpdate):
    try:
        normalized = update_image_tags(update.path, update.tags, settings)
        return {"status": "success", "tags": normalized}
    except Exception as error:
        raise _http_error(error)


@app.post("/api/dataset/batch-tags")
async def batch_tags(update: BatchTagUpdate):
    try:
        if update.position == "remove":
            result = batch_remove_tags(update.path, update.tags, settings)
        else:
            result = batch_update_tags(update.path, update.tags, update.position, settings)
        return {"status": "success", **result}
    except Exception as error:
        raise _http_error(error)


@app.post("/api/dataset/trigger-word")
async def trigger_word(update: TriggerWordUpdate):
    try:
        result = batch_update_tags(update.path, [update.trigger_word], "prepend", settings)
        return {"status": "success", **result}
    except Exception as error:
        raise _http_error(error)


@app.get("/api/image")
async def get_image(path: str):
    try:
        image_path = validate_image_path(path, settings)
        return FileResponse(str(image_path))
    except Exception as error:
        raise _http_error(error)


@app.post("/api/run-tagger")
async def run_tagger(config: TaggerConfig):
    try:
        state, _ = await tagger_service.start(config)
        return {"status": "started", **state}
    except Exception as error:
        raise _http_error(error)


@app.post("/api/setup-scripts")
async def setup_scripts():
    if processes.running("setup") is not None:
        return {"status": "error", "message": "setup is already running"}
    setup_script = settings.backend_root / "setup_check.py"
    if not setup_script.is_file():
        raise HTTPException(status_code=404, detail=f"setup script not found: {setup_script}")
    run_id = f"setup-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    async def on_line(run: ProcessRun, line: str) -> None:
        await hub.log("setup", line, run_id)

    async def on_done(run: ProcessRun) -> None:
        await hub.status("setup", run.public_dict(), run_id)

    try:
        run = await processes.start(
            kind="setup",
            run_id=run_id,
            command=[sys.executable, str(setup_script)],
            cwd=settings.backend_root,
            log_path=settings.job_root / "setup" / f"{run_id}.log",
            on_line=on_line,
            on_done=on_done,
            env={"PYTHONUNBUFFERED": "1"},
        )
        return {"status": "started", "run_id": run_id}
    except Exception as error:
        raise _http_error(error)


@app.post("/api/start-training")
async def start_training(config: TrainingConfig):
    try:
        state, _ = await training_service.start(config)
        return {"status": "started", **training_service._public_state(state)}
    except Exception as error:
        raise _http_error(error)


@app.post("/api/sync-output")
async def sync_output(request: OutputSyncRequest):
    try:
        state, _ = await sync_service.start_output(request)
        return {"status": "started", **state}
    except Exception as error:
        raise _http_error(error)


@app.get("/api/status")
async def status(job_id: str = ""):
    training = training_service.get(job_id) if job_id else None
    if training is None:
        # ``dict_values`` is not reversible on every supported Python version.
        latest = list(training_service.states.values())[-1] if training_service.states else None
        training = training_service._public_state(latest) if latest is not None else None
    return {
        "training": training,
        "tagger": tagger_service.latest(),
        "sync": sync_service.latest(),
        "processes": processes.public_status(),
    }


@app.get("/api/jobs")
async def jobs():
    return {"jobs": training_service.public_status()}


@app.get("/api/jobs/{job_id}")
async def job(job_id: str):
    state = training_service.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    return state


@app.post("/api/recommendations")
async def recommendations(request: RecommendationRequest):
    return recommend_settings(
        request.image_count,
        request.training_type,
        request.vram,
        request.gradient_accumulation_steps,
    )


@app.post("/api/cancel/{kind}")
async def cancel(kind: str):
    if kind not in {"training", "tagger", "sync", "backup", "setup"}:
        raise HTTPException(status_code=400, detail="unsupported process kind")
    result = await processes.cancel(kind)
    if result is None:
        raise HTTPException(status_code=404, detail=f"no running {kind} process")
    return {"status": "cancellation_requested", **result}


# The API routes are registered before the catch-all static mount.
app.mount("/", StaticFiles(directory=str(settings.frontend_root), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("APP_HOST", "0.0.0.0"),
        port=settings.app_port,
        ws_ping_interval=60,
        ws_ping_timeout=60,
        timeout_keep_alive=120,
    )
