"""Episode runner with optional signal-observation ablations (eval-only)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from beer_distribution_rl.env.core import ROLES, Role
from beer_distribution_rl.env.signals import Signal

from analysis.diag.common import signal_feature_slice


def _apply_signal_ablation(
    obs_by_role: dict[Role, np.ndarray],
    mode: str,
    rng: np.random.Generator,
    sig_slice: slice,
) -> dict[Role, np.ndarray]:
    """Corrupt only the delayed signal-board features listeners see."""
    if mode == "intact":
        return obs_by_role

    out = {r: o.copy() for r, o in obs_by_role.items()}
    roles = list(ROLES)

    if mode == "zero":
        for r in roles:
            out[r][sig_slice] = 0.0
        return out

    if mode == "shuffle":
        # Same permutation of role-blocks applied to every listener's board.
        blocks = []
        # Use retailer board as canonical (all listeners see the same delayed board).
        board = out[Role.RETAILER][sig_slice].reshape(len(ROLES), 3).copy()
        perm = rng.permutation(len(ROLES))
        shuffled = board[perm]
        flat = shuffled.reshape(-1)
        for r in roles:
            out[r][sig_slice] = flat
        return out

    if mode == "random":
        # Valid-ish values: present ∈ {0,1}, claims ~ U[0, scale] with scale=1 (obs already /20).
        for r in roles:
            feats = []
            for _ in ROLES:
                present = float(rng.integers(0, 2))
                dem = float(rng.uniform(0.0, 15.0 / 20.0)) if present else 0.0
                inv = float(rng.uniform(0.0, 40.0 / 20.0)) if present else 0.0
                feats.extend([present, dem, inv])
            out[r][sig_slice] = np.asarray(feats, dtype=np.float32)
        return out

    raise ValueError(f"unknown ablation mode: {mode}")


def evaluate_with_ablation(
    trainer,
    *,
    n_episodes: int,
    seed: int,
    ablation_mode: str = "intact",
    collect_steps: bool = False,
    step_hook: Callable | None = None,
) -> dict:
    """Mirror IPPOTrainer.evaluate but optionally ablate signal features in obs.

    Agents still *emit* signals via the policy; only listener observations are corrupted.
    """
    cfg = trainer.cfg
    core = trainer.core
    signaling = trainer.signaling
    sig_slice = signal_feature_slice(trainer.obs_dim)
    rng = np.random.default_rng(seed + 17)

    costs: list[float] = []
    local = {r: [] for r in ROLES}
    share_rates: list[float] = []
    honesty_scores: list[float] = []
    order_series = {r: [] for r in ROLES}
    demand_series: list[float] = []
    inflation_flags: list[float] = []
    step_records: list[dict] = []

    for ep in range(n_episodes):
        states = core.reset(seed + ep)
        done = False
        sys_acc = 0.0
        loc_acc = {r: 0.0 for r in ROLES}
        broadcasts = 0
        broadcast_opps = 0
        mae_sum = 0.0
        mae_n = 0
        steps = 0
        while not done:
            orders: dict[Role, int] = {}
            signals: dict[Role, Signal | None] | None = {} if signaling else None
            raw_obs = {r: trainer._obs(states, r) for r in ROLES}
            obs = _apply_signal_ablation(raw_obs, ablation_mode, rng, sig_slice)

            with torch.no_grad():
                for r in ROLES:
                    o = torch.as_tensor(obs[r], device=trainer.device).unsqueeze(0)
                    a, _, _ = trainer._policy_act(r, o, greedy=not signaling)
                    if signaling:
                        row = a.squeeze(0).cpu().numpy().astype(int)
                        orders[r] = trainer._decode_order(int(row[0]), states[r])
                        assert signals is not None
                        signals[r] = trainer._decode_signal(
                            states[r], int(row[1]), int(row[2]), int(row[3])
                        )
                    else:
                        orders[r] = trainer._decode_order(int(a.item()), states[r])

            states, rewards, done, info = core.step(orders, signals)
            sys_acc += info.system_cost
            for r in ROLES:
                loc_acc[r] += info.local_costs[r]
                order_series[r].append(info.orders_placed[r])
            demand_series.append(info.incoming_orders[Role.RETAILER])

            if signaling:
                for r in ROLES:
                    broadcast_opps += 1
                    if info.signals_sent.get(r) is not None:
                        broadcasts += 1
                    h = info.honesty.get(r, {})
                    mae = h.get("mean_abs_error", float("nan"))
                    if mae == mae:
                        mae_sum += float(mae)
                        mae_n += 1
                if info.rationed:
                    need = info.incoming_orders[Role.FACTORY]
                    if need > 0 and info.orders_placed[Role.FACTORY] > 1.5 * need:
                        inflation_flags.append(1.0)
                    else:
                        inflation_flags.append(0.0)

            if collect_steps or step_hook is not None:
                rec = {
                    "ep": ep,
                    "t": core.t,
                    "rationed": bool(info.rationed),
                    "factory_production": int(info.factory_production),
                    "capacity": core.config.capacity,
                    "orders_placed": {r.name.lower(): int(info.orders_placed[r]) for r in ROLES},
                    "incoming_orders": {
                        r.name.lower(): int(info.incoming_orders[r]) for r in ROLES
                    },
                    "shipments": {r.name.lower(): int(info.shipments[r]) for r in ROLES},
                    "shipments_received": {
                        r.name.lower(): int(info.shipments_received[r]) for r in ROLES
                    },
                    "inventories": {r.name.lower(): int(states[r].inventory) for r in ROLES},
                    "backlogs": {r.name.lower(): int(states[r].backlog) for r in ROLES},
                    "system_cost": float(info.system_cost),
                }
                # Capacity bind: production order clipped by factory capacity.
                cap = core.config.capacity
                factory_order = int(info.orders_placed[Role.FACTORY])
                if cap is None:
                    rec["capacity_binds"] = False
                else:
                    rec["capacity_binds"] = factory_order > int(cap)
                # After fill: backlog > 0 iff available < need this week (allocation shortfall).
                # On a serial chain the "proportional" rule is identity for the single claimant;
                # "triggers" means a physical shortfall occurred (fill < need).
                rec["allocation_triggers"] = any(int(states[r].backlog) > 0 for r in ROLES)
                need_f = int(info.incoming_orders[Role.FACTORY])
                rec["factory_order_inflation"] = (
                    float(factory_order) / float(need_f) if need_f > 0 else float("nan")
                )
                if collect_steps:
                    step_records.append(rec)
                if step_hook is not None:
                    step_hook(rec, info, states)

            steps += 1

        costs.append(sys_acc / max(steps, 1))
        for r in ROLES:
            local[r].append(loc_acc[r] / max(steps, 1))
        if signaling and broadcast_opps:
            share_rates.append(broadcasts / broadcast_opps)
        if mae_n:
            honesty_scores.append(-(mae_sum / mae_n) / max(cfg.order_cap, 1))

    out: dict = {
        "eval/mean_system_cost": float(np.mean(costs)),
        "eval/std_system_cost": float(np.std(costs)),
        "eval/episode_costs": [float(x) for x in costs],
    }
    for r in ROLES:
        out[f"eval/{r.name.lower()}_cost"] = float(np.mean(local[r]))
    dvar = float(np.var(demand_series)) if len(demand_series) > 1 else 0.0
    for r in ROLES:
        ovar = float(np.var(order_series[r])) if len(order_series[r]) > 1 else 0.0
        out[f"eval/bullwhip_{r.name.lower()}"] = ovar / dvar if dvar > 1e-12 else float("inf")
    # Aggregate bullwhip (factory / demand) as primary ratio for plots.
    out["eval/bullwhip_ratio"] = out["eval/bullwhip_factory"]
    if signaling:
        out["eval/sharing_rate"] = float(np.mean(share_rates)) if share_rates else 0.0
        out["eval/honesty_score"] = float(np.mean(honesty_scores)) if honesty_scores else 0.0
        out["eval/inflation_rate"] = float(np.mean(inflation_flags)) if inflation_flags else 0.0
    if collect_steps:
        out["steps"] = step_records
    return out
