from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional


LineHandler = Callable[["ProcessRun", str], Awaitable[None]]
DoneHandler = Callable[["ProcessRun"], Awaitable[None]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProcessRun:
    run_id: str
    kind: str
    command: List[str]
    log_path: Path
    started_at: str = field(default_factory=_now)
    ended_at: str = ""
    status: str = "starting"
    returncode: Optional[int] = None
    error: str = ""
    cancel_requested: bool = False
    process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    task: Optional[asyncio.Task] = field(default=None, repr=False)

    def running(self) -> bool:
        return self.status in {"starting", "running"}

    def public_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "command": self.command,
            "log_path": str(self.log_path),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "returncode": self.returncode,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "pid": self.process.pid if self.process is not None else None,
        }


class ProcessManager:
    """Runs external commands without invoking a shell."""

    def __init__(self):
        self.runs: Dict[str, ProcessRun] = {}

    def running(self, kind: str) -> Optional[ProcessRun]:
        run = self.runs.get(kind)
        return run if run and run.running() else None

    async def start(
        self,
        kind: str,
        run_id: str,
        command: List[str],
        cwd: Path,
        log_path: Path,
        on_line: LineHandler,
        on_done: DoneHandler,
        env: Optional[dict] = None,
    ) -> ProcessRun:
        if self.running(kind) is not None:
            raise RuntimeError(f"{kind} is already running")

        run = ProcessRun(run_id=run_id, kind=kind, command=list(command), log_path=log_path)
        self.runs[kind] = run
        run.task = asyncio.create_task(
            self._execute(run, cwd, on_line, on_done, env), name=f"sdxl-{kind}-{run_id}"
        )
        return run

    async def _execute(
        self,
        run: ProcessRun,
        cwd: Path,
        on_line: LineHandler,
        on_done: DoneHandler,
        env: Optional[dict],
    ) -> None:
        run.log_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = None
        try:
            if run.cancel_requested:
                run.status = "cancelled"
                return
            command_env = os.environ.copy()
            if env:
                command_env.update({str(k): str(v) for k, v in env.items()})
            run.process = await asyncio.create_subprocess_exec(
                *run.command,
                cwd=str(cwd),
                env=command_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            run.status = "running"
            output_file = run.log_path.open("w", encoding="utf-8", errors="replace")
            assert run.process.stdout is not None
            while True:
                raw_line = await run.process.stdout.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                output_file.write(line + "\n")
                output_file.flush()
                try:
                    await on_line(run, line)
                except Exception as callback_error:  # keep the child process alive
                    run.error = f"line handler error: {callback_error}"
            run.returncode = await run.process.wait()
            if run.cancel_requested:
                run.status = "cancelled"
            else:
                run.status = "succeeded" if run.returncode == 0 else "failed"
        except Exception as error:
            run.status = "failed"
            run.returncode = -1
            run.error = str(error)
        finally:
            if output_file is not None:
                output_file.close()
            run.ended_at = _now()
            try:
                await on_done(run)
            except Exception as callback_error:
                run.error = f"completion handler error: {callback_error}"

    async def cancel(self, kind: str) -> Optional[dict]:
        run = self.running(kind)
        if run is None:
            return None
        run.cancel_requested = True
        process = run.process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        return run.public_dict()

    async def wait(self, run: Optional[ProcessRun]) -> Optional[ProcessRun]:
        if run is not None and run.task is not None:
            await run.task
        return run

    def public_status(self) -> dict:
        return {kind: run.public_dict() for kind, run in self.runs.items()}
