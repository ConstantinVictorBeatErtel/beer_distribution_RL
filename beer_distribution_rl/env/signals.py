"""Optional, unverified cheap-talk signaling channel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from beer_distribution_rl.env.core_types import ROLES, Role


@dataclass(frozen=True)
class Signal:
    claimed_demand: int | None = None
    claimed_inventory: int | None = None


@dataclass(frozen=True)
class HonestyMetrics:
    abs_demand_error: float | None
    abs_inventory_error: float | None
    mean_abs_error: float | None  # mean over provided (non-null) claims


@dataclass
class SignalChannel:
    """Broadcasts are free, optional, delayed, and never verified."""

    delay: int = 1
    _buffer: list[dict[Role, Signal | None]] = field(default_factory=list)

    def reset(self) -> None:
        # Seed delay buffer with empty broadcasts so week-1 receive is all-None.
        self._buffer = [{r: None for r in ROLES} for _ in range(self.delay)]

    def send(self, signals: Mapping[Role, Signal | None]) -> None:
        frame = {r: signals.get(r) for r in ROLES}
        self._buffer.append(frame)

    def receive(self) -> dict[Role, dict[Role, Signal | None]]:
        """What every role observes this week (same delayed global board)."""
        if not self._buffer:
            board = {r: None for r in ROLES}
        else:
            board = self._buffer.pop(0)
        # Each role sees the full delayed board (spec: all agents observe all broadcasts).
        return {observer: dict(board) for observer in ROLES}

    def measure_honesty(
        self,
        signals: Mapping[Role, Signal | None],
        truths: Mapping[Role, Mapping[str, int]],
    ) -> dict[Role, HonestyMetrics]:
        out: dict[Role, HonestyMetrics] = {}
        for role in ROLES:
            sig = signals.get(role)
            truth = truths.get(role, {})
            if sig is None:
                out[role] = HonestyMetrics(None, None, None)
                continue
            errs: list[float] = []
            dem_err: float | None = None
            inv_err: float | None = None
            if sig.claimed_demand is not None and "demand" in truth:
                dem_err = float(abs(sig.claimed_demand - truth["demand"]))
                errs.append(dem_err)
            if sig.claimed_inventory is not None and "inventory" in truth:
                inv_err = float(abs(sig.claimed_inventory - truth["inventory"]))
                errs.append(inv_err)
            mae = sum(errs) / len(errs) if errs else None
            out[role] = HonestyMetrics(dem_err, inv_err, mae)
        return out
