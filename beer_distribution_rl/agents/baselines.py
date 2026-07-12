"""Heuristic and optimal-form baselines for the beer game."""

from __future__ import annotations

import random
from dataclasses import dataclass

from beer_distribution_rl.env.core import RoleState


@dataclass(frozen=True)
class StermanParams:
    """Anchoring-and-adjustment parameters (Sterman 1989 style).

    Documented in DECISIONS.md for reproducibility.
    """

    theta: float = 0.36  # demand expectation smoothing
    alpha_s: float = 0.5  # stock adjustment
    alpha_sl: float = 0.5  # supply-line adjustment
    ship_delay: int = 2


@dataclass
class StermanAgent:
    """Stateful Sterman heuristic (maintains smoothed demand expectation)."""

    params: StermanParams = StermanParams()
    expected_demand: float = 4.0

    def reset(self, expected_demand: float = 4.0) -> None:
        self.expected_demand = expected_demand

    def order(self, state: RoleState) -> int:
        p = self.params
        d = float(state.last_demand_or_order)
        self.expected_demand = p.theta * self.expected_demand + (1.0 - p.theta) * d
        desired_stock = self.expected_demand
        net_stock = float(state.inventory - state.backlog)
        supply_line = float(sum(state.ship_pipeline))
        desired_sl = self.expected_demand * float(p.ship_delay)
        adj_stock = p.alpha_s * (desired_stock - net_stock)
        adj_sl = p.alpha_sl * (desired_sl - supply_line)
        raw = self.expected_demand + adj_stock + adj_sl
        return max(0, int(round(raw)))


def sterman_order(
    state: RoleState, params: StermanParams, expected_demand: float
) -> tuple[int, float]:
    """One Sterman step; returns (order, updated_expected_demand)."""
    agent = StermanAgent(params=params, expected_demand=expected_demand)
    qty = agent.order(state)
    return qty, agent.expected_demand


def base_stock_order(state: RoleState, S: int, order_cap: int = 64) -> int:
    """Order-up-to base-stock policy on inventory position."""
    ip = state.inventory_position()
    return max(0, min(order_cap, int(S - ip)))


def random_order(rng: random.Random, cap: int = 64) -> int:
    return rng.randint(0, cap)


# Classic Clark–Scarf levels from Oroojlooy et al. (arXiv:1708.05924)
CLASSIC_BASE_STOCK_LEVELS: dict[str, int] = {
    "retailer": 9,
    "wholesaler": 5,
    "distributor": 3,
    "factory": 1,
}

CLASSIC_BASE_STOCK_VECTOR: tuple[int, int, int, int] = (9, 5, 3, 1)
