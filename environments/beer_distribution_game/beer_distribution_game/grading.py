"""Deterministic programmatic grading over canonical transition records."""

from __future__ import annotations

from statistics import mean, median, variance
from typing import Any

from .scenario import Role, ScenarioSpec


def _sum_role_cost(transitions: list[dict], role: Role) -> float:
    return float(sum(float(row["local_costs"][role]) for row in transitions))


def _terminal_cost(spec: ScenarioSpec, inventory_position: int) -> float:
    if inventory_position >= 0:
        return spec.holding_cost * inventory_position
    return spec.backlog_cost * -inventory_position


def _service_metrics(
    transitions: list[dict], role: Role
) -> dict[str, float | int | None]:
    total_new = 0
    total_immediate = 0
    positive_cycles = 0
    successful_cycles = 0
    for row in transitions:
        incoming = row["incoming_by_claimant"][role]
        backlog_before = row["backlog_before_by_claimant"][role]
        allocation = row["allocations"][role]
        for claimant, new_demand in incoming.items():
            new_demand = int(new_demand)
            prior = int(backlog_before.get(claimant, 0))
            shipped = int(allocation.get(claimant, 0))
            immediate = min(new_demand, max(0, shipped - prior))
            total_new += new_demand
            total_immediate += immediate
            if new_demand > 0:
                positive_cycles += 1
                successful_cycles += int(immediate == new_demand)
    ending_backlog = int(
        transitions[-1]["states_after_fulfillment"][role]["backlog"]
    )
    return {
        "immediate_fill_rate": (
            total_immediate / total_new if total_new else None
        ),
        "cycle_service_level": (
            successful_cycles / positive_cycles if positive_cycles else None
        ),
        "horizon_fulfillment": (
            (total_new - ending_backlog) / total_new if total_new else None
        ),
        "ending_backlog": ending_backlog,
        "new_demand_units": total_new,
    }


def _stability_metrics(
    spec: ScenarioSpec, transitions: list[dict], role: Role
) -> dict[str, float | int | None]:
    warmup = spec.order_delay + spec.shipment_delay
    scored = transitions[warmup:]
    orders = [int(row["orders"][role]) for row in scored]
    demands = [
        int(sum(row["incoming_by_claimant"][role].values())) for row in scored
    ]
    demand_variance = variance(demands) if len(demands) >= 2 else None
    order_variance = variance(orders) if len(orders) >= 2 else None
    bullwhip = (
        order_variance / demand_variance
        if demand_variance is not None
        and demand_variance > 1e-12
        and order_variance is not None
        else None
    )
    volatility = None
    if len(orders) >= 2:
        volatility = mean(abs(b - a) for a, b in zip(orders, orders[1:])) / max(
            mean(demands), 1
        )
    total_demand = sum(demands)
    return {
        "metric_warmup_weeks": warmup,
        "bullwhip_ratio": bullwhip,
        "normalized_order_volatility": volatility,
        "order_mean": mean(orders) if orders else None,
        "order_median": median(orders) if orders else None,
        "order_variance": order_variance,
        "demand_variance": demand_variance,
        "maximum_order": max(orders) if orders else None,
        "order_cap_hit_rate": (
            sum(order == spec.order_cap for order in orders) / len(orders)
            if orders
            else None
        ),
        "order_to_demand_ratio": sum(orders) / total_demand if total_demand else None,
    }


def grade_episode(
    *,
    spec: ScenarioSpec,
    controlled_role: Role,
    operational: list[dict],
    settlement: list[dict],
    terminal_inventory_positions: dict[Role, int],
    protocol_clean: bool,
    base_reference: dict[str, float] | None = None,
) -> dict[str, Any]:
    operational_local = _sum_role_cost(operational, controlled_role)
    settlement_local = _sum_role_cost(settlement, controlled_role)
    terminal_by_role = {
        role: _terminal_cost(spec, terminal_inventory_positions[role])
        for role in spec.roles
    }
    terminal_local = terminal_by_role[controlled_role]
    local_total = operational_local + settlement_local + terminal_local
    operational_system = sum(
        _sum_role_cost(operational, role) for role in spec.roles
    )
    settlement_system = sum(
        _sum_role_cost(settlement, role) for role in spec.roles
    )
    terminal_system = sum(terminal_by_role.values())
    system_total = operational_system + settlement_system + terminal_system

    base_local = None if base_reference is None else base_reference["local_total_cost"]
    base_system = None if base_reference is None else base_reference["system_total_cost"]
    cost_score = None
    if base_local is not None:
        if base_local > 0:
            cost_score = base_local / (base_local + local_total)
        else:
            cost_score = 1.0 if local_total == 0 else 0.0
    reward = None if cost_score is None else float(protocol_clean) * cost_score

    return {
        "grader_version": "1.0.0",
        "status": "scored" if cost_score is not None else "reference",
        "episode_reward": reward,
        "protocol_clean": protocol_clean,
        "primary": {
            "local_total_cost": local_total,
            "paired_base_stock_local_total_cost": base_local,
            "cost_score": cost_score,
        },
        "costs": {
            "operational_local_cost": operational_local,
            "settlement_local_cost": settlement_local,
            "terminal_exposure_cost": terminal_local,
            "mean_weekly_local_cost": operational_local / spec.horizon,
            "system_total_cost": system_total,
            "operational_system_cost": operational_system,
            "settlement_system_cost": settlement_system,
            "terminal_system_exposure_cost": terminal_system,
            "other_roles_total_cost": system_total - local_total,
            "paired_base_stock_system_total_cost": base_system,
            "local_cost_ratio": (
                local_total / base_local if base_local not in (None, 0) else None
            ),
            "system_cost_ratio": (
                system_total / base_system if base_system not in (None, 0) else None
            ),
        },
        "service": _service_metrics(operational, controlled_role),
        "stability": _stability_metrics(spec, operational, controlled_role),
        "termination_reason": "horizon_completed",
    }
