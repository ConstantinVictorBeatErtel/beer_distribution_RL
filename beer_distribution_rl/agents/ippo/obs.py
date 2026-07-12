"""Observation featurization for IPPO (local info only + delayed signals)."""

from __future__ import annotations

import numpy as np

from beer_distribution_rl.env.core import BeerGameCore, EnvConfig, Role, RoleState
from beer_distribution_rl.env.core_types import ROLES
from beer_distribution_rl.env.signals import Signal


def obs_dim(config: EnvConfig) -> int:
    # inventory, backlog, on_order, last_demand, last_ship, last_order,
    # ship_pipeline..., order_pipeline..., t_frac, holding, backlog_cost
    base = 6 + config.ship_delay + config.order_delay + 3
    if config.signaling_enabled:
        # per role on delayed board: present, claimed_demand, claimed_inventory
        base += 3 * len(ROLES)
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
        board = getattr(core, "_last_signal_board", None) or {r: None for r in ROLES}
        for r in ROLES:
            feats.extend(_signal_feats(board.get(r)))
    expected = obs_dim(cfg)
    if len(feats) < expected:
        feats.extend([0.0] * (expected - len(feats)))
    return np.asarray(feats[:expected], dtype=np.float32)
