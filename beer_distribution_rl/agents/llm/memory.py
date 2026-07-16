"""Structured own-history memory for LLM agents (Check 3 information set)."""

from __future__ import annotations

from dataclasses import dataclass, field

from beer_distribution_rl.env.core import Role


@dataclass
class WeekRecord:
    """One week of own-visible outcomes (memory-matched vs recurrent IPPO).

    Fields align with Check 3 structured history / ``OWN_HISTORY_CORE_FIELDS``
    in ``agents/ippo/obs.py`` — never other agents' private state.
    """

    week: int
    demand_or_incoming: int  # last_demand_or_order observed
    ship_in: int  # last_shipment_received at decision time
    ordered: int  # order placed this week
    alloc_recv: int  # allocation / shipment received this week (post-step)
    inventory: int
    backlog: int
    on_order: int
    ship_pipeline: list[int]
    order_pipeline: list[int]
    local_cost: float


@dataclass
class AgentMemory:
    """Per-role retained context across weeks (own history only)."""

    role: Role
    role_name: str
    history: list[WeekRecord] = field(default_factory=list)

    def append(self, record: WeekRecord) -> None:
        self.history.append(record)
