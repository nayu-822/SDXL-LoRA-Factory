from __future__ import annotations

import asyncio
import copy
import json
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Set

try:
    from fastapi import WebSocketDisconnect
except ImportError:  # FastAPI is only required by the HTTP entrypoint.
    class WebSocketDisconnect(Exception):
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventHub:
    """Small in-process event hub for logs and progress over WebSocket."""

    def __init__(self, history_size: int = 500):
        self.connections: Set[Any] = set()
        self.locks: Dict[Any, asyncio.Lock] = {}
        self.history: Deque[dict] = deque(maxlen=history_size)
        self.state: dict = {}

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        self.connections.add(websocket)
        self.locks[websocket] = asyncio.Lock()
        await self._send(websocket, {"type": "connected", "time": _now()})
        for event in list(self.history):
            await self._send(websocket, event)
        if self.state:
            await self._send(websocket, {"type": "state", "state": self.state, "time": _now()})

    async def disconnect(self, websocket: Any) -> None:
        self.connections.discard(websocket)
        self.locks.pop(websocket, None)

    async def _send(self, websocket: Any, event: dict) -> None:
        lock = self.locks.get(websocket)
        if lock is None:
            return
        try:
            async with lock:
                await websocket.send_text(json.dumps(event, ensure_ascii=False))
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            await self.disconnect(websocket)

    async def publish(self, event: dict, remember: bool = True) -> None:
        # Event payloads often contain mutable job state. Snapshot them so a
        # reconnecting browser receives the state that existed at each event.
        event = copy.deepcopy(event)
        event.setdefault("time", _now())
        if remember:
            self.history.append(event)
        for websocket in list(self.connections):
            await self._send(websocket, event)

    async def log(self, channel: str, line: str, task_id: str = "") -> None:
        await self.publish(
            {
                "type": "log",
                "channel": channel,
                "task_id": task_id,
                "line": line.rstrip("\r\n"),
            }
        )

    async def progress(self, channel: str, progress: dict, task_id: str = "") -> None:
        await self.publish(
            {
                "type": "progress",
                "channel": channel,
                "task_id": task_id,
                "progress": progress,
            }
        )

    async def status(self, channel: str, status: dict, task_id: str = "") -> None:
        event = {"type": "status", "channel": channel, "task_id": task_id, "status": status}
        self.state[channel] = copy.deepcopy(status)
        await self.publish(event)
