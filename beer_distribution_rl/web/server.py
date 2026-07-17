"""FastAPI app: static UI, WebSocket stream, REST controls."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from beer_distribution_rl.web.runner import EpisodeRunner

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ControlRequest(BaseModel):
    action: Literal["play", "pause", "step", "reset"]
    seed: int | None = None
    speed_ms: int | None = Field(default=None, ge=50, le=5000)


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def broadcast_threadsafe(self, message: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)


def create_app(runner: EpisodeRunner | None = None) -> FastAPI:
    manager = ConnectionManager()
    episode = runner or EpisodeRunner()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.bind_loop(asyncio.get_running_loop())
        episode.add_listener(manager.broadcast_threadsafe)
        app.state.runner = episode
        app.state.manager = manager
        yield
        episode.remove_listener(manager.broadcast_threadsafe)
        episode.shutdown()

    app = FastAPI(title="Beer Game Spectator", lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return episode.snapshot()

    @app.post("/api/control")
    async def control(body: ControlRequest) -> dict[str, Any]:
        if body.speed_ms is not None:
            episode.set_speed(body.speed_ms)

        if body.action == "play":
            episode.play()
        elif body.action == "pause":
            episode.pause()
        elif body.action == "step":
            episode.step_once()
        elif body.action == "reset":
            episode.reset(seed=body.seed)

        return episode.snapshot()

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            await ws.send_json(episode.snapshot())
            while True:
                # Keep the socket alive; control goes through REST.
                # Clients may send pings as JSON `{"type":"ping"}`.
                data = await ws.receive_json()
                if isinstance(data, dict) and data.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await manager.disconnect(ws)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
