"""Structured own-history memory for LLM agents (Check 3 information set).

Productized rolling window: each week's prompt includes only the last ``W``
weeks of the agent's OWN structured history (orders, demand observed,
allocations, backlog). Default ``W=8`` clears readiness blockers 3/7;
``W`` is a one-line config knob for later sensitivity ablations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from beer_distribution_rl.env.core import Role

# Ablation-ready default — change once for W-sensitivity (mirrors recurrent
# baseline "does more memory help?" question).
DEFAULT_ROLLING_WINDOW: int = 8


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
    """Per-role retained context across weeks (own history only).

    Full episode history is retained for diagnostics; prompts use
    :meth:`windowed_history` (last ``window`` weeks only). E1: never stores
    other agents' private state.
    """

    role: Role
    role_name: str
    history: list[WeekRecord] = field(default_factory=list)
    window: int = DEFAULT_ROLLING_WINDOW

    def append(self, record: WeekRecord) -> None:
        self.history.append(record)

    def windowed_history(self, window: int | None = None) -> list[WeekRecord]:
        """Last ``W`` own-week records for the prompt (rolling context)."""
        w = self.window if window is None else window
        if w < 0:
            raise ValueError(f"window must be >= 0, got {w}")
        if w == 0:
            return []
        return self.history[-w:]

    def reset(self) -> None:
        self.history.clear()
