"""Obs → prompt serializer (E1 no-leak, memory-matched information set)."""

from __future__ import annotations

import re
from typing import Any

from beer_distribution_rl.agents.llm.memory import AgentMemory
from beer_distribution_rl.env.core import BeerGameCore, Role

# Check-3 / recurrent-IPPO own-history fields that MUST appear in text prompts.
OWN_HISTORY_FIELDS: tuple[str, ...] = (
    "inventory",
    "backlog",
    "on_order",
    "last_demand_or_order",  # demand / incoming observed
    "last_shipment_received",  # ship_in / alloc_recv at decision
    "last_order_placed",  # own past order
    "ship_pipeline",
    "order_pipeline",
)

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "customer_demand",
    "true_demand",
    "consumer_demand",
    "end_customer_demand",
    "rival_inventory",
    "other_agent_inventory",
)


def observe_local(core: BeerGameCore, role: Role) -> dict[str, Any]:
    """Local dict obs for LLM agents — never includes signals (order-only design)."""
    obs = core.observe(role)
    obs.pop("signals", None)
    return obs


def serialize_prompt(
    memory: AgentMemory,
    obs: dict[str, Any],
    *,
    order_cap: int,
    holding: float,
    backlog_cost: float,
    delta_max: int = 8,
) -> str:
    """Render an agent's own observation (+ retained own history) as text.

    Emits exactly the memory-matched information set (own past orders, demand
    observed, allocations received, backlog, pipelines). Never emits other
    agents' private state or privileged demand keys (E1).
    """
    for key in OWN_HISTORY_FIELDS:
        if key not in obs:
            raise KeyError(f"obs missing required field {key!r}")

    lines = [
        f"You are the {memory.role_name} in a beer distribution supply chain.",
        "Goal: minimize YOUR local cost = holding*inventory + backlog*backlog_units.",
        f"Your costs: holding={holding}, backlog={backlog_cost}.",
        f"Action space: choose integer delta in [{-delta_max}, {delta_max}].",
        f"Your order will be clip(last_demand_or_order + delta, 0, {order_cap}).",
        "You do NOT see other agents' private inventories or true consumer demand "
        "under a privileged key (retailers see customer demand only as "
        "last_demand_or_order).",
        "Your only action is the order delta. Do not broadcast signals.",
        "",
        "Output grammar (strict): emit ONLY a JSON object matching:",
        f'{{"delta": <integer in [{-delta_max}, {delta_max}]>}}',
        "Example valid reply:",
        '{"delta": 0}',
        "No other text.",
        "",
        f"Current week t={obs['t']} (0-indexed; upcoming decision).",
        f"inventory={obs['inventory']}",
        f"backlog={obs['backlog']}",
        f"on_order={obs['on_order']}",
        f"inventory_position={obs['inventory_position']}",
        f"last_demand_or_order={obs['last_demand_or_order']}",
        f"last_shipment_received={obs['last_shipment_received']}",
        f"last_order_placed={obs['last_order_placed']}",
        f"ship_pipeline={obs['ship_pipeline']}",
        f"order_pipeline={obs['order_pipeline']}",
        "",
        "Own history (prior weeks):",
    ]
    if not memory.history:
        lines.append("(none — first week)")
    else:
        for rec in memory.history:
            lines.append(
                f"  week={rec.week}: demand_or_incoming={rec.demand_or_incoming}, "
                f"ship_in={rec.ship_in}, ordered={rec.ordered}, "
                f"alloc_recv={rec.alloc_recv}, "
                f"inv={rec.inventory}, backlog={rec.backlog}, "
                f"on_order={rec.on_order}, "
                f"ship_pipeline={rec.ship_pipeline}, "
                f"order_pipeline={rec.order_pipeline}, "
                f"cost={rec.local_cost:.2f}"
            )
    lines.append("")
    lines.append('{"delta":')
    return "\n".join(lines)


def prompt_leak_report(prompt: str, role: Role, core: BeerGameCore) -> list[str]:
    """E1-style leak checks on rendered text (no other-role private state)."""
    issues: list[str] = []
    low = prompt.lower()
    for s in FORBIDDEN_SUBSTRINGS:
        if s in low:
            issues.append(f"forbidden substring: {s}")
    for r, name in core.role_names.items():
        if r == role:
            continue
        pat = re.compile(rf"\b{re.escape(name)}\b.*\binventory\s*=\s*\d+", re.I | re.S)
        if pat.search(prompt):
            issues.append(f"other-role inventory leak pattern for {name}")
    if role not in (Role.RETAILER, Role.RETAILER_B):
        if re.search(r"customer_demand\s*=\s*\d+", prompt, re.I):
            issues.append("customer_demand numeric field")
        if re.search(r"true_demand\s*=\s*\d+", prompt, re.I):
            issues.append("true_demand numeric field")
    return issues
