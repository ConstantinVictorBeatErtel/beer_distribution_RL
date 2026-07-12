"""Observation featurization for IPPO (local info only)."""

from __future__ import annotations

import numpy as np

from beer_distribution_rl.env.core import BeerGameCore, EnvConfig, Role, RoleState


def obs_dim(config: EnvConfig) -> int:
    # inventory, backlog, on_order, last_demand, last_ship, last_order,
    # ship_pipeline..., order_pipeline..., t_frac, holding, backlog_cost
    return 6 + config.ship_delay + config.order_delay + 3


def state_to_obs(state: RoleState, role: Role, core: BeerGameCore) -> np.ndarray:
    """Local observation — never includes other roles' inventories."""
    cfg = core.config
    costs = cfg.costs[int(role)]
    t_frac = core.t / max(cfg.horizon, 1)
    scale = 20.0  # rough classic inventory scale for numerical stability
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
    expected = obs_dim(cfg)
    if len(feats) < expected:
        feats.extend([0.0] * (expected - len(feats)))
    return np.asarray(feats[:expected], dtype=np.float32)
