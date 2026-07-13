"""Optional, unverified cheap-talk signaling channel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

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
    """Broadcasts are free, optional, delayed, and never verified.

    Every role in ``roles`` hears the full delayed board — including a rival
    retailer in the Y-topology. Truthfulness is never enforced.
    """

    delay: int = 1
    roles: tuple[Role, ...] = ROLES
    _buffer: list[dict[Role, Signal | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("SignalChannel requires a non-empty roles tuple")

    def reset(self) -> None:
        # Seed delay buffer with empty broadcasts so week-1 receive is all-None.
        self._buffer = [{r: None for r in self.roles} for _ in range(self.delay)]

    def send(self, signals: Mapping[Role, Signal | None]) -> None:
        frame = {r: signals.get(r) for r in self.roles}
        self._buffer.append(frame)

    def receive(self) -> dict[Role, dict[Role, Signal | None]]:
        """What every role observes this week (same delayed global board)."""
        if not self._buffer:
            board = {r: None for r in self.roles}
        else:
            board = self._buffer.pop(0)
        # Each role sees the full delayed board (spec: all agents observe all broadcasts).
        return {observer: dict(board) for observer in self.roles}

    def listeners_of(self, sender: Role) -> tuple[Role, ...]:
        """Who hears ``sender``'s broadcast (everyone, including rivals)."""
        return self.roles

    def measure_honesty(
        self,
        signals: Mapping[Role, Signal | None],
        truths: Mapping[Role, Mapping[str, int]],
        roles: Sequence[Role] | None = None,
    ) -> dict[Role, HonestyMetrics]:
        active = tuple(roles) if roles is not None else self.roles
        out: dict[Role, HonestyMetrics] = {}
        for role in active:
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
