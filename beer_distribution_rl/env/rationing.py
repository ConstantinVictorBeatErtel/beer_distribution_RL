"""Allocation / rationing mechanisms for capacity-constrained nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from beer_distribution_rl.env.core_types import Role


@dataclass
class RationContext:
    """Optional context for honesty-weighted allocation."""

    honesty_ema: Mapping[Role, float] = field(default_factory=dict)
    temperature: float = 1.0


class RationingPolicy(Protocol):
    def allocate(
        self,
        requested: Mapping[Role, int],
        available: int,
        ctx: RationContext | None = None,
    ) -> dict[Role, int]:
        """Return allocations with sum ≤ available and 0 ≤ alloc[r] ≤ requested[r]."""
        ...


def _identity_or_empty(requested: Mapping[Role, int], available: int) -> dict[Role, int] | None:
    total = sum(int(v) for v in requested.values())
    if available >= total:
        return {r: int(v) for r, v in requested.items()}
    if available <= 0 or total <= 0:
        return {r: 0 for r in requested}
    return None


def _largest_remainder(raw: Mapping[Role, float], available: int, caps: Mapping[Role, int]) -> dict[Role, int]:
    """Hamilton largest-remainder to convert fractional shares to ints."""
    floors = {r: min(caps[r], int(math.floor(raw[r]))) for r in raw}
    used = sum(floors.values())
    remaining = available - used
    order = sorted(
        raw.keys(),
        key=lambda r: (raw[r] - math.floor(raw[r]), -int(r)),
        reverse=True,
    )
    out = dict(floors)
    for r in order:
        if remaining <= 0:
            break
        if out[r] < caps[r]:
            out[r] += 1
            remaining -= 1
    # If still remaining but all at cap (shouldn't happen if sum(caps)>=available path), drop.
    return out


@dataclass
class ProportionalRationing:
    """Allocate in proportion to requested amounts (classic shortage-gaming rule)."""

    def allocate(
        self,
        requested: Mapping[Role, int],
        available: int,
        ctx: RationContext | None = None,
    ) -> dict[Role, int]:
        early = _identity_or_empty(requested, available)
        if early is not None:
            return early
        total = sum(int(v) for v in requested.values())
        raw = {r: available * (int(v) / total) for r, v in requested.items()}
        caps = {r: int(v) for r, v in requested.items()}
        return _largest_remainder(raw, available, caps)


@dataclass
class UniformRationing:
    """Split available as evenly as possible (capped by request)."""

    def allocate(
        self,
        requested: Mapping[Role, int],
        available: int,
        ctx: RationContext | None = None,
    ) -> dict[Role, int]:
        early = _identity_or_empty(requested, available)
        if early is not None:
            return early
        roles = list(requested.keys())
        out = {r: 0 for r in roles}
        remaining = available
        # Iteratively give one unit in round-robin to roles still below request.
        while remaining > 0:
            progressed = False
            for r in roles:
                if remaining <= 0:
                    break
                if out[r] < int(requested[r]):
                    out[r] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return out


@dataclass
class HonestyWeightedRationing:
    """Weight ∝ exp(EMA / temperature); EMA is measured honesty (higher = more honest).

    Does NOT enter the RL reward — only physical allocation.
    ``honesty_ema`` values are expected on a scale where larger ⇒ more honest
    (core passes ``-mean_abs_error`` EMA, so less lying ⇒ larger / less-negative).
    """

    def allocate(
        self,
        requested: Mapping[Role, int],
        available: int,
        ctx: RationContext | None = None,
    ) -> dict[Role, int]:
        early = _identity_or_empty(requested, available)
        if early is not None:
            return early
        ctx = ctx or RationContext()
        temp = max(ctx.temperature, 1e-6)
        weights: dict[Role, float] = {}
        for r, req in requested.items():
            ema = float(ctx.honesty_ema.get(r, 0.0))
            weights[r] = math.exp(ema / temp) * max(int(req), 0)
        wsum = sum(weights.values())
        if wsum <= 0:
            # Fall back to proportional on requests.
            return ProportionalRationing().allocate(requested, available, ctx)
        raw = {r: available * (weights[r] / wsum) for r in requested}
        caps = {r: int(v) for r, v in requested.items()}
        return _largest_remainder(raw, available, caps)
