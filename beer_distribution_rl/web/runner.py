"""Thread-safe Sterman-driven episode runner for the live spectator."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from beer_distribution_rl.agents.baselines import StermanAgent
from beer_distribution_rl.env.core import BeerGameCore, y_topology_env_config
from beer_distribution_rl.env.core_types import Y_ROLE_NAMES, Y_ROLES, Role
from beer_distribution_rl.web.frames import SpectatorFrame, frame_from_step, initial_frame

Listener = Callable[[dict[str, Any]], None]

ROLE_ORDER: tuple[str, ...] = tuple(Y_ROLE_NAMES[r] for r in Y_ROLES)


class EpisodeRunner:
    """Runs Y-topology Beer Game episodes with Sterman agents.

    Thread-safe: play loop runs in a background thread; control methods
    may be called from the FastAPI / WebSocket event loop.
    """

    def __init__(self, speed_ms: int = 500, seed: int = 0) -> None:
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        self._speed_ms = max(50, int(speed_ms))
        self._seed = int(seed)
        self._playing = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()

        # Y-topology: two retailers under one wholesaler, correlated demand.
        self._core = BeerGameCore(y_topology_env_config())
        self._agents: dict[Role, StermanAgent] = {r: StermanAgent() for r in Y_ROLES}
        self._cumulative_cost = 0.0
        self._cumulative_local_costs: dict[Role, float] = {r: 0.0 for r in Y_ROLES}
        self._last_frame: SpectatorFrame | None = None
        self._history: list[dict[str, Any]] = []

        self.reset(seed=self._seed)

    # --- listeners ---------------------------------------------------------

    def add_listener(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _emit(self, message: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(message)
            except Exception:
                pass

    def _status_payload(self) -> dict[str, Any]:
        return {
            "type": "status",
            "playing": self._playing,
            "speed_ms": self._speed_ms,
            "seed": self._seed,
            "roles": list(ROLE_ORDER),
            "horizon": self._core.config.horizon,
            "topology": self._core.topology.name,
            "demand_model": getattr(
                self._core.config.demand,
                "name",
                type(self._core.config.demand).__name__,
            ),
        }

    def _emit_status(self) -> None:
        self._emit(self._status_payload())

    def _emit_frame(self, frame: SpectatorFrame) -> None:
        payload = {"type": "frame", **frame.to_dict()}
        self._emit(payload)

    # --- public state ------------------------------------------------------

    @property
    def last_frame(self) -> SpectatorFrame | None:
        with self._lock:
            return self._last_frame

    @property
    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def snapshot(self) -> dict[str, Any]:
        """Full sync payload for a newly connected client."""
        with self._lock:
            status = self._status_payload()
            frame = self._last_frame.to_dict() if self._last_frame else None
            history = list(self._history)
        out: dict[str, Any] = {**status, "type": "snapshot", "history": history}
        if frame is not None:
            out["frame"] = frame
        return out

    # --- controls ----------------------------------------------------------

    def set_speed(self, speed_ms: int) -> None:
        with self._lock:
            self._speed_ms = max(50, int(speed_ms))
        self._emit_status()
        self._wake.set()

    def play(self) -> None:
        with self._lock:
            if self._core._terminated:
                return
            self._playing = True
            self._ensure_thread()
        self._emit_status()
        self._wake.set()

    def pause(self) -> None:
        with self._lock:
            self._playing = False
        self._emit_status()

    def step_once(self) -> SpectatorFrame | None:
        with self._lock:
            was_playing = self._playing
            self._playing = False
            frame = self._advance()
        if was_playing:
            self._emit_status()
        return frame

    def reset(self, seed: int | None = None) -> SpectatorFrame:
        with self._lock:
            self._playing = False
            if seed is not None:
                self._seed = int(seed)
            for agent in self._agents.values():
                agent.reset()
            states = self._core.reset(seed=self._seed)
            self._cumulative_cost = 0.0
            self._cumulative_local_costs = {r: 0.0 for r in Y_ROLES}
            self._history = []
            frame = initial_frame(states, horizon=self._core.config.horizon)
            self._last_frame = frame
            self._history.append(frame.to_dict())
        self._emit_status()
        self._emit_frame(frame)
        self._wake.set()
        return frame

    def shutdown(self) -> None:
        self._stop.set()
        self._playing = False
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    # --- internals ---------------------------------------------------------

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="spectator-runner", daemon=True)
        self._thread.start()

    def _advance(self) -> SpectatorFrame | None:
        """Take one env step. Caller must hold ``_lock``."""
        if self._core._terminated:
            self._playing = False
            return None

        orders = {r: self._agents[r].order(self._core.states[r]) for r in Y_ROLES}
        states, _rewards, terminated, info = self._core.step(orders)
        self._cumulative_cost += float(info.system_cost)
        for role in Y_ROLES:
            self._cumulative_local_costs[role] += float(info.local_costs[role])
        frame = frame_from_step(
            states,
            info,
            t=self._core.t,
            cumulative_cost=self._cumulative_cost,
            cumulative_local_costs=self._cumulative_local_costs,
            terminated=terminated,
            horizon=self._core.config.horizon,
        )
        self._last_frame = frame
        self._history.append(frame.to_dict())
        if terminated:
            self._playing = False
        self._emit_frame(frame)
        if terminated:
            self._emit_status()
        return frame

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                playing = self._playing and not self._core._terminated
                speed = self._speed_ms
            if not playing:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            with self._lock:
                if self._playing and not self._core._terminated:
                    self._advance()
            # Sleep outside the lock so controls stay responsive.
            deadline = time.monotonic() + speed / 1000.0
            while time.monotonic() < deadline and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if self._wake.wait(timeout=min(0.05, remaining)):
                    self._wake.clear()
                    with self._lock:
                        if not self._playing:
                            break
