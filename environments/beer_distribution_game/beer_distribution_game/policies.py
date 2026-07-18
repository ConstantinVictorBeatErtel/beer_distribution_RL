"""Deterministic scripted counterparties and cheap baseline policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Protocol

from .scenario import Role, ScenarioSpec, derive_seed


class Policy(Protocol):
    policy_id: str
    policy_version: str

    def act(self, observation: dict) -> int: ...


def initial_forecast(spec: ScenarioSpec, role: Role) -> float:
    if spec.tier == 1:
        return 8.0
    if spec.topology == "y" and role not in ("retailer_a", "retailer_b"):
        return 15.0
    return 7.5


@dataclass
class AdaptiveBaseStockPolicy:
    forecast: float
    alpha: float = 0.25
    order_cap: int = 128
    replenishment_delay: int = 3
    policy_id: str = "adaptive_base_stock"
    policy_version: str = "2"

    def act(self, observation: dict) -> int:
        demand = float(observation["state"]["incoming_demand_or_order"])
        self.forecast = (1.0 - self.alpha) * self.forecast + self.alpha * demand
        # The observation is emitted after this week's demand has been fulfilled.
        # An order placed now arrives after `replenishment_delay` further decision
        # intervals, so adding another review period double-counts one week.
        target = math.ceil(self.replenishment_delay * self.forecast)
        inventory_position = int(observation["state"]["inventory_position"])
        return min(self.order_cap, max(0, target - inventory_position))


@dataclass
class ScarcityAggressivePolicy:
    base: AdaptiveBaseStockPolicy
    increment: int = 8
    policy_id: str = "scarcity_aggressive"
    policy_version: str = "1"

    def act(self, observation: dict) -> int:
        return min(self.base.order_cap, self.base.act(observation) + self.increment)


@dataclass
class RandomPolicy:
    rng: random.Random
    order_cap: int = 128
    policy_id: str = "uniform_random"
    policy_version: str = "1"

    def act(self, observation: dict) -> int:
        del observation
        return self.rng.randint(0, self.order_cap)


def adaptive_policy(spec: ScenarioSpec, role: Role) -> AdaptiveBaseStockPolicy:
    return AdaptiveBaseStockPolicy(
        forecast=initial_forecast(spec, role),
        order_cap=spec.order_cap,
        replenishment_delay=spec.order_delay + spec.shipment_delay,
    )


def counterparty_policies(
    spec: ScenarioSpec, controlled_role: Role
) -> dict[Role, Policy]:
    policies: dict[Role, Policy] = {}
    for role in spec.roles:
        if role == controlled_role:
            continue
        base = adaptive_policy(spec, role)
        if spec.aggressive_retailers and role in ("retailer_a", "retailer_b"):
            policies[role] = ScarcityAggressivePolicy(base)
        else:
            policies[role] = base
    return policies


def random_policy(spec: ScenarioSpec, role: Role) -> RandomPolicy:
    return RandomPolicy(
        rng=random.Random(
            derive_seed(spec.master_seed_hex, f"baseline/random/{role}")
        ),
        order_cap=spec.order_cap,
    )
