"""Demand processes for the beer distribution game."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


class DemandProcess(Protocol):
    def reset(self, rng: random.Random) -> None: ...

    def __call__(self, t: int, rng: random.Random) -> int:
        """Return non-negative integer demand for week ``t`` (1-indexed)."""
        ...


@dataclass
class ClassicStepDemand:
    """Classic MIT step: demand ``pre`` for weeks < switch_week, else ``post``."""

    pre: int = 4
    post: int = 8
    switch_week: int = 5

    def reset(self, rng: random.Random) -> None:
        return None

    def __call__(self, t: int, rng: random.Random) -> int:
        return self.pre if t < self.switch_week else self.post


@dataclass
class UniformDemand:
    """Stationary uniform integer demand on [low, high] inclusive."""

    low: int = 0
    high: int = 15

    def reset(self, rng: random.Random) -> None:
        return None

    def __call__(self, t: int, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)


@dataclass
class AR1Demand:
    """Gaussian AR(1) demand rounded to non-negative ints, optional mean shift."""

    mu: float = 8.0
    phi: float = 0.5
    sigma: float = 2.0
    regime_shift_week: int | None = None
    mu_after: float | None = None
    x0: float | None = None

    def __post_init__(self) -> None:
        self._x: float = self.mu if self.x0 is None else self.x0

    def reset(self, rng: random.Random) -> None:
        self._x = self.mu if self.x0 is None else self.x0

    def __call__(self, t: int, rng: random.Random) -> int:
        mu = self.mu
        if (
            self.regime_shift_week is not None
            and self.mu_after is not None
            and t >= self.regime_shift_week
        ):
            mu = self.mu_after
        eps = rng.gauss(0.0, self.sigma)
        self._x = mu + self.phi * (self._x - mu) + eps
        return max(0, int(round(self._x)))


def mean_demand(process: DemandProcess, horizon: int, seed: int = 0) -> float:
    """Empirical mean demand over ``horizon`` weeks (for capacity sweeps)."""
    rng = random.Random(seed)
    process.reset(rng)
    total = 0
    for t in range(1, horizon + 1):
        total += int(process(t, rng))
    return total / horizon
