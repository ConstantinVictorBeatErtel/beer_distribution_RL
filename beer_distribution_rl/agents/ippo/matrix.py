"""Tier-1 matrix cell enumeration with documented pruning (Agent R1).

Full design space:
  {A,B,C} × {serial,y} × {∞,1.2μ,1.0μ,0.8μ} × {prop,uniform,honesty_weighted}
  × {ar1,regime_switch} × ≥10 seeds

Pruning (see DECISIONS.md § Tier-1 v1.1 matrix):
  1. At C=∞: only proportional rationing (mechanisms never bind).
  2. On serial: only proportional (single claimant ⇒ prop≡uniform≡honesty_weighted).
  3. Dropped 1.5μ from the M3 sweep (new grid is {∞,1.2,1.0,0.8}).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from beer_distribution_rl.agents.ippo.trainer import IPPOConfig, capacity_tag, default_run_name


# Axes (post-prune defaults)
REGIMES = ("A", "B", "C")
TOPOLOGIES = ("serial", "y")
CAPS: tuple[float | None, ...] = (None, 1.2, 1.0, 0.8)
RATIONING_ALL = ("proportional", "uniform", "honesty_weighted")
DEMANDS = ("ar1", "regime_switch")
DEFAULT_SEEDS = tuple(range(10))


@dataclass(frozen=True)
class MatrixCell:
    regime: str
    topology: str
    capacity_mult: float | None
    rationing: str
    demand: str
    seed: int

    def to_config(self, **overrides: Any) -> IPPOConfig:
        base = dict(
            regime=self.regime,
            topology=self.topology,
            capacity_mult=self.capacity_mult,
            rationing=self.rationing,
            demand=self.demand,
            seed=self.seed,
            # Vec defaults: n_envs×rollout = large PPO batch; timesteps set so
            # update count ≈ M3 (50k/1024 ≈ 49). 400k/(64×128) ≈ 48.8 updates.
            n_envs=64,
            rollout_steps=128,
            total_timesteps=400_000,
            minibatch_size=2048,
            eval_every=10,
            eval_episodes=8,
            log_every=10,
            device="cpu",
            out_dir="artifacts/runs/ippo/tier1_v11",
        )
        base.update(overrides)
        return IPPOConfig(**base)

    @property
    def run_name(self) -> str:
        return default_run_name(self.to_config())

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "topology": self.topology,
            "capacity_mult": self.capacity_mult,
            "capacity_tag": capacity_tag(self.to_config()),
            "rationing": self.rationing,
            "demand": self.demand,
            "seed": self.seed,
            "run_name": self.run_name,
        }


def should_keep_cell(
    regime: str,
    topology: str,
    capacity_mult: float | None,
    rationing: str,
    demand: str,
) -> tuple[bool, str | None]:
    """Return (keep, prune_reason). Prune reasons are logged in DECISIONS.md."""
    _ = (regime, demand)  # axes kept fully; pruning is topology/cap/rationing
    topo = topology.lower()
    if capacity_mult is None and rationing != "proportional":
        return False, "rationing_at_infinite_capacity"
    if topo in ("serial", "chain") and rationing != "proportional":
        return False, "serial_single_claimant_equivalence"
    return True, None


def iter_pruned_cells(
    *,
    regimes: tuple[str, ...] = REGIMES,
    topologies: tuple[str, ...] = TOPOLOGIES,
    caps: tuple[float | None, ...] = CAPS,
    rationing: tuple[str, ...] = RATIONING_ALL,
    demands: tuple[str, ...] = DEMANDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> Iterator[MatrixCell]:
    for regime, topo, cap, rat, dem, seed in itertools.product(
        regimes, topologies, caps, rationing, demands, seeds
    ):
        keep, _ = should_keep_cell(regime, topo, cap, rat, dem)
        if keep:
            yield MatrixCell(
                regime=regime,
                topology=topo,
                capacity_mult=cap,
                rationing=rat,
                demand=dem,
                seed=seed,
            )


def enumerate_cells(**kwargs: Any) -> list[MatrixCell]:
    return list(iter_pruned_cells(**kwargs))


def prune_summary(
    *,
    regimes: tuple[str, ...] = REGIMES,
    topologies: tuple[str, ...] = TOPOLOGIES,
    caps: tuple[float | None, ...] = CAPS,
    rationing: tuple[str, ...] = RATIONING_ALL,
    demands: tuple[str, ...] = DEMANDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Counts before/after prune for DECISIONS / CLI."""
    full = 0
    reasons: dict[str, int] = {}
    kept = 0
    for regime, topo, cap, rat, dem, _seed in itertools.product(
        regimes, topologies, caps, rationing, demands, seeds
    ):
        full += 1
        ok, reason = should_keep_cell(regime, topo, cap, rat, dem)
        if ok:
            kept += 1
        else:
            assert reason is not None
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "full_cartesian": full,
        "kept": kept,
        "pruned": full - kept,
        "prune_reasons": reasons,
        "axes": {
            "regimes": list(regimes),
            "topologies": list(topologies),
            "caps": ["inf" if c is None else c for c in caps],
            "rationing": list(rationing),
            "demands": list(demands),
            "n_seeds": len(seeds),
        },
    }


def cell_config_payload(cell: MatrixCell, cfg: IPPOConfig) -> dict[str, Any]:
    """YAML-serializable config + identity for every run artifact."""
    return {
        "cell": cell.as_dict(),
        "config": asdict(cfg),
    }
