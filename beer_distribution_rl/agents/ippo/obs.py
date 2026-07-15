"""Observation featurization for IPPO (local info only + delayed signals).

Memory / information-set notes (Check 3 + Check 4, llm_tier_readiness):
The flat local obs is the *per-week* content both the Markovian MLP and the
recurrent GRU consume. Structured LLM history serializes the same own-only
fields over weeks:

  demand_or_incoming ← last_demand_or_order
  ship_in / alloc_recv ← last_shipment_received
  ordered ← last_order_placed
  inv / backlog ← inventory, backlog
  cost ← h·inv + b·backlog (coeffs also in obs)

E1 no-leak: never includes other roles' true inventories or privileged
customer_demand / true_demand keys for upstream agents.
"""

from __future__ import annotations

import numpy as np

from beer_distribution_rl.env.core import BeerGameCore, EnvConfig, Role, RoleState
from beer_distribution_rl.env.signals import Signal
from beer_distribution_rl.env.topology import get_topology

# Check-3-aligned own-history fields present in every local obs (indices 0–5).
OWN_HISTORY_CORE_FIELDS: tuple[str, ...] = (
    "inventory",
    "backlog",
    "on_order",
    "last_demand_or_order",  # demand / incoming observed this week
    "last_shipment_received",  # ship_in / allocation received
    "last_order_placed",  # own past order
)


def _signal_roles(config: EnvConfig) -> tuple[Role, ...]:
    """Roles that appear on the cheap-talk board (topology-dependent)."""
    topo = config.topology
    if isinstance(topo, str):
        return get_topology(topo).roles
    return topo.roles


def obs_dim(config: EnvConfig) -> int:
    # inventory, backlog, on_order, last_demand, last_ship, last_order,
    # ship_pipeline..., order_pipeline..., t_frac, holding, backlog_cost
    base = 6 + config.ship_delay + config.order_delay + 3
    if config.signaling_enabled:
        # per role on delayed board: present, claimed_demand, claimed_inventory
        base += 3 * len(_signal_roles(config))
    return base


def _signal_feats(sig: Signal | None, scale: float = 20.0) -> list[float]:
    if sig is None:
        return [0.0, 0.0, 0.0]
    dem = 0.0 if sig.claimed_demand is None else float(sig.claimed_demand) / scale
    inv = 0.0 if sig.claimed_inventory is None else float(sig.claimed_inventory) / scale
    return [1.0, dem, inv]


def state_to_obs(state: RoleState, role: Role, core: BeerGameCore) -> np.ndarray:
    """Local observation — never includes other roles' true inventories."""
    cfg = core.config
    costs = cfg.costs[int(role)]
    t_frac = core.t / max(cfg.horizon, 1)
    scale = 20.0
    feats = [
        float(state.inventory) / scale,
        float(state.backlog) / scale,
        float(state.on_order) / scale,
        float(state.last_demand_or_order) / scale,
        float(state.last_shipment_received) / scale,
        float(state.last_order_placed) / scale,
        *[float(x) / scale for x in state.ship_pipeline],
        *[float(x) / scale for x in state.order_pipeline],
        float(t_frac),
        float(costs.holding),
        float(costs.backlog),
    ]
    if cfg.signaling_enabled:
        board_roles = core.roles
        board = getattr(core, "_last_signal_board", None) or {r: None for r in board_roles}
        for r in board_roles:
            feats.extend(_signal_feats(board.get(r)))
    expected = obs_dim(cfg)
    if len(feats) < expected:
        feats.extend([0.0] * (expected - len(feats)))
    return np.asarray(feats[:expected], dtype=np.float32)
