"""Independent PPO trainer: one ActorCritic per role, no shared parameters."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

from beer_distribution_rl.agents.ippo.buffer import RoleBuffer
from beer_distribution_rl.agents.ippo.networks import ActorCritic, SignalingActorCritic
from beer_distribution_rl.agents.ippo.obs import obs_dim, state_to_obs
from beer_distribution_rl.agents.ippo.vec_env import SyncBeerGameVecEnv
from beer_distribution_rl.env.core import (
    BeerGameCore,
    EnvConfig,
    Role,
    classic_env_config,
    y_topology_env_config,
)
from beer_distribution_rl.env.demand import (
    DEFAULT_ORDER_CAP,
    info_value_table,
    mean_demand,
    recommend_order_cap,
    resolve_matrix_demand,
    warn_if_boundary_saturated,
)
from beer_distribution_rl.env.rationing import (
    HonestyWeightedRationing,
    ProportionalRationing,
    UniformRationing,
)
from beer_distribution_rl.env.signals import Signal


@dataclass
class IPPOConfig:
    regime: str = "A"  # A, B, or C
    horizon: int = 52
    total_timesteps: int = 200_000
    # Vectorized envs per cell — large N makes GPU useful for 2×256 MLPs.
    n_envs: int = 1
    rollout_steps: int = 1024  # steps *per env*; batch = rollout_steps * n_envs
    update_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    hidden: int = 256
    order_cap: int = DEFAULT_ORDER_CAP
    action_mode: str = "relative"  # relative | absolute
    action_delta_max: int = 8
    claim_delta_max: int = 8
    reward_scale: float = 0.1
    # v1.1 training default: AR(1); uniform retained for backward comparison.
    demand: str = "ar1"  # classic_step | uniform | ar1 | regime_switch | correlated_y
    topology: str = "serial"  # serial | y
    # Capacity: None = ∞; else multiplier of mean demand (1.2, 1.0, 0.8)
    capacity_mult: float | None = None
    rationing: str = "proportional"  # proportional | uniform | honesty_weighted
    seed: int = 0
    device: str = "cpu"
    log_every: int = 5
    eval_every: int = 10
    eval_episodes: int = 10
    out_dir: str = "artifacts/runs/ippo"
    run_name: str | None = None
    boundary_warn_frac: float = 0.05


def _make_rationing(name: str):
    if name == "proportional":
        return ProportionalRationing()
    if name == "uniform":
        return UniformRationing()
    if name == "honesty_weighted":
        return HonestyWeightedRationing()
    raise ValueError(f"unknown rationing: {name}")


def build_env_config(cfg: IPPOConfig) -> EnvConfig:
    if cfg.regime not in ("A", "B", "C"):
        raise ValueError(f"unknown regime: {cfg.regime}")
    topo = cfg.topology.lower().strip()
    if topo in ("y", "y_topology", "ytopology"):
        factory = y_topology_env_config
        topo_name = "y"
    elif topo in ("serial", "chain"):
        factory = classic_env_config
        topo_name = "serial"
    else:
        raise ValueError(f"unknown topology: {cfg.topology}")

    demand = resolve_matrix_demand(cfg.demand, topo_name)

    capacity = None
    if cfg.capacity_mult is not None:
        mu = mean_demand(demand, cfg.horizon, seed=cfg.seed)
        capacity = float(cfg.capacity_mult) * mu

    return factory(
        horizon=cfg.horizon,
        demand=demand,
        regime=cfg.regime,  # type: ignore[arg-type]
        signaling_enabled=(cfg.regime == "B"),
        order_cap=cfg.order_cap,
        seed=cfg.seed,
        capacity=capacity,
        rationing=_make_rationing(cfg.rationing),
        topology=topo_name,
    )


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def capacity_tag(cfg: IPPOConfig) -> str:
    if cfg.capacity_mult is None:
        return "inf"
    return f"{cfg.capacity_mult:.1f}mu".replace(".", "p")


def default_run_name(cfg: IPPOConfig) -> str:
    topo = cfg.topology.lower().replace("y_topology", "y")
    if topo in ("y", "ytopology"):
        topo = "y"
    return (
        f"regime{cfg.regime}_topo{topo}_cap{capacity_tag(cfg)}_"
        f"rat{cfg.rationing}_dem{cfg.demand}_seed{cfg.seed}"
    )


def _bind_flags(info) -> tuple[bool, bool]:
    """Prefer StepInfo fields; fall back for older cores."""
    cap = bool(getattr(info, "capacity_binds", False))
    alloc = bool(getattr(info, "allocation_triggers", False))
    return cap, alloc


class IPPOTrainer:
    """CleanRL-style IPPO with strictly independent per-role policies/critics."""

    def __init__(self, cfg: IPPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env_config = build_env_config(cfg)
        self.core = BeerGameCore(self.env_config)
        self.roles = self.core.roles
        self.vec = SyncBeerGameVecEnv(self.env_config, max(1, cfg.n_envs))
        if cfg.action_mode == "relative":
            self.n_order = 2 * cfg.action_delta_max + 1
        elif cfg.action_mode == "absolute":
            self.n_order = cfg.order_cap + 1
        else:
            raise ValueError(f"unknown action_mode: {cfg.action_mode}")
        self.n_claim = 2 * cfg.claim_delta_max + 1
        self.obs_dim = obs_dim(self.env_config)
        self.signaling = cfg.regime == "B"

        self.policies: dict[Role, nn.Module] = {}
        for r in self.roles:
            if self.signaling:
                self.policies[r] = SignalingActorCritic(
                    self.obs_dim, self.n_order, self.n_claim, cfg.hidden
                ).to(self.device)
            else:
                self.policies[r] = ActorCritic(self.obs_dim, self.n_order, cfg.hidden).to(
                    self.device
                )
        self.optimizers = {
            r: torch.optim.Adam(self.policies[r].parameters(), lr=cfg.learning_rate, eps=1e-5)
            for r in self.roles
        }
        self.buffers = {r: RoleBuffer() for r in self.roles}
        self._assert_independent_params()
        self.global_step = 0
        self.update_idx = 0
        self.history: list[dict[str, Any]] = []
        self._bootstrap: dict[Role, float] = {}

    def _assert_independent_params(self) -> None:
        ids = []
        for r in self.roles:
            for p in self.policies[r].parameters():
                ids.append(id(p))
        if len(ids) != len(set(ids)):
            raise RuntimeError("Parameter sharing detected across roles")
        critic_ids = [id(self.policies[r].critic_head) for r in self.roles]
        if len(set(critic_ids)) != len(self.roles):
            raise RuntimeError("Shared critic detected across roles")

    def _obs(self, states, role: Role, core: BeerGameCore | None = None) -> np.ndarray:
        return state_to_obs(states[role], role, core or self.core)

    def _decode_order(self, action_idx: int, state) -> int:
        if self.cfg.action_mode == "absolute":
            return int(action_idx)
        delta = int(action_idx) - self.cfg.action_delta_max
        raw = int(state.last_demand_or_order) + delta
        return max(0, min(self.cfg.order_cap, raw))

    def _decode_signal(self, state, broadcast: int, cd_idx: int, ci_idx: int) -> Signal | None:
        if int(broadcast) == 0:
            return None
        d_delta = int(cd_idx) - self.cfg.claim_delta_max
        i_delta = int(ci_idx) - self.cfg.claim_delta_max
        claimed_d = max(0, min(self.cfg.order_cap, int(state.last_demand_or_order) + d_delta))
        claimed_i = max(0, min(self.cfg.order_cap * 2, int(state.inventory) + i_delta))
        return Signal(claimed_demand=claimed_d, claimed_inventory=claimed_i)

    def _policy_act(self, role: Role, ot: torch.Tensor, greedy: bool = False):
        pol = self.policies[role]
        if self.signaling:
            if greedy:
                actions = pol.greedy(ot)  # type: ignore[attr-defined]
                _, value = None, pol._dists(ot)[4]  # type: ignore[attr-defined]
                return actions, torch.zeros(ot.shape[0], device=self.device), value
            return pol.act(ot)
        if greedy:
            dist, value = pol.forward(ot)  # type: ignore[attr-defined]
            a = dist.probs.argmax(dim=-1)
            return a, dist.log_prob(a), value
        return pol.act(ot)

    def collect_rollout(self) -> dict[str, float]:
        cfg = self.cfg
        n_envs = self.vec.n_envs
        for b in self.buffers.values():
            b.clear()

        states_list = self.vec.reset(cfg.seed + self.global_step)
        ep_system_costs: list[float] = []
        ep_cost_acc = np.zeros(n_envs, dtype=np.float64)
        ep_len = np.zeros(n_envs, dtype=np.int32)
        steps = 0

        while steps < cfg.rollout_steps:
            orders_batch: list[dict[Role, int]] = [{} for _ in range(n_envs)]
            signals_batch: list[dict[Role, Signal | None] | None] = (
                [{} for _ in range(n_envs)] if self.signaling else [None] * n_envs
            )
            stored: dict[Role, list[Any]] = {r: [] for r in self.roles}
            logprobs: dict[Role, list[float]] = {r: [] for r in self.roles}
            values: dict[Role, list[float]] = {r: [] for r in self.roles}
            obs_np: dict[Role, list[np.ndarray]] = {r: [] for r in self.roles}

            with torch.no_grad():
                for r in self.roles:
                    obs_stack = np.stack(
                        [
                            self._obs(states_list[i], r, self.vec.cores[i])
                            for i in range(n_envs)
                        ]
                    )
                    ot = torch.as_tensor(obs_stack, device=self.device)
                    a, lp, val = self._policy_act(r, ot, greedy=False)
                    lp_np = lp.detach().cpu().numpy()
                    val_np = val.detach().cpu().numpy()
                    if self.signaling:
                        rows = a.detach().cpu().numpy().astype(int)
                        for i in range(n_envs):
                            row = rows[i]
                            obs_np[r].append(obs_stack[i])
                            stored[r].append(row.tolist())
                            logprobs[r].append(float(lp_np[i]))
                            values[r].append(float(val_np[i]))
                            orders_batch[i][r] = self._decode_order(
                                int(row[0]), states_list[i][r]
                            )
                            assert signals_batch[i] is not None
                            signals_batch[i][r] = self._decode_signal(  # type: ignore[index]
                                states_list[i][r], int(row[1]), int(row[2]), int(row[3])
                            )
                    else:
                        idxs = a.detach().cpu().numpy().astype(int)
                        for i in range(n_envs):
                            idx = int(idxs[i])
                            obs_np[r].append(obs_stack[i])
                            stored[r].append(idx)
                            logprobs[r].append(float(lp_np[i]))
                            values[r].append(float(val_np[i]))
                            orders_batch[i][r] = self._decode_order(idx, states_list[i][r])

            states_list, rewards, dones, infos = self.vec.step(orders_batch, signals_batch)
            self.global_step += n_envs
            steps += 1

            for i in range(n_envs):
                ep_cost_acc[i] += infos[i].system_cost
                ep_len[i] += 1
                for r in self.roles:
                    self.buffers[r].obs.append(obs_np[r][i])
                    self.buffers[r].actions.append(stored[r][i])
                    self.buffers[r].logprobs.append(logprobs[r][i])
                    self.buffers[r].rewards.append(float(rewards[i][r]) * cfg.reward_scale)
                    self.buffers[r].dones.append(bool(dones[i]))
                    self.buffers[r].values.append(values[r][i])
                if dones[i]:
                    ep_system_costs.append(float(ep_cost_acc[i] / max(int(ep_len[i]), 1)))
                    ep_cost_acc[i] = 0.0
                    ep_len[i] = 0
                    states_list[i] = self.vec.reset_one(i, cfg.seed + self.global_step + i)

        self._bootstrap = {}
        with torch.no_grad():
            for r in self.roles:
                obs_stack = np.stack(
                    [
                        self._obs(states_list[i], r, self.vec.cores[i])
                        for i in range(n_envs)
                    ]
                )
                ot = torch.as_tensor(obs_stack, device=self.device)
                if self.signaling:
                    v = self.policies[r]._dists(ot)[4]  # type: ignore
                else:
                    _, v = self.policies[r].forward(ot)  # type: ignore
                self._bootstrap[r] = v.detach().cpu().numpy().astype(np.float32)

        mean_ep = float(np.mean(ep_system_costs)) if ep_system_costs else float("nan")
        return {
            "rollout_mean_system_cost": mean_ep,
            "episodes": float(len(ep_system_costs)),
            "n_envs": float(n_envs),
        }

    def update(self) -> dict[str, float]:
        cfg = self.cfg
        metrics: dict[str, float] = {}

        n_envs = self.vec.n_envs
        for r in self.roles:
            buf = self.buffers[r]
            boot = self._bootstrap.get(r, 0.0)
            if n_envs == 1:
                # Scalar bootstrap; zero if last transition terminated.
                last_v: float | np.ndarray = (
                    0.0 if (buf.dones and buf.dones[-1]) else float(np.asarray(boot).reshape(-1)[0])
                )
            else:
                boot_arr = np.asarray(boot, dtype=np.float32).reshape(-1)
                if boot_arr.shape != (n_envs,):
                    boot_arr = np.full(n_envs, float(boot_arr.reshape(-1)[0]), dtype=np.float32)
                # Zero bootstrap for envs that ended on the last stored step.
                last_dones = np.asarray(buf.dones[-n_envs:], dtype=np.float32)
                last_v = boot_arr * (1.0 - last_dones)
            advantages, returns = buf.compute_gae(
                last_v, cfg.gamma, cfg.gae_lambda, n_envs=n_envs
            )
            data = buf.as_tensors()
            obs = data["obs"].to(self.device)
            actions = data["actions"].to(self.device)
            old_logprobs = data["logprobs"].to(self.device)
            advantages = advantages.to(self.device)
            returns = returns.to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            n = len(buf)
            inds = np.arange(n)
            policy_loss_acc = value_loss_acc = entropy_acc = 0.0
            n_updates = 0

            for _ in range(cfg.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, n, cfg.minibatch_size):
                    mb = inds[start : start + cfg.minibatch_size]
                    mb_t = torch.as_tensor(mb, device=self.device)
                    new_logprob, value, entropy = self.policies[r].evaluate(
                        obs[mb_t], actions[mb_t]
                    )
                    ratio = (new_logprob - old_logprobs[mb_t]).exp()
                    adv = advantages[mb_t]
                    pg1 = ratio * adv
                    pg2 = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv
                    policy_loss = -torch.min(pg1, pg2).mean()
                    value_loss = 0.5 * ((value - returns[mb_t]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                    self.optimizers[r].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policies[r].parameters(), cfg.max_grad_norm)
                    self.optimizers[r].step()

                    policy_loss_acc += float(policy_loss.item())
                    value_loss_acc += float(value_loss.item())
                    entropy_acc += float(entropy_loss.item())
                    n_updates += 1

            prefix = self.core.role_names.get(r, r.name.lower())
            metrics[f"{prefix}/policy_loss"] = policy_loss_acc / max(n_updates, 1)
            metrics[f"{prefix}/value_loss"] = value_loss_acc / max(n_updates, 1)
            metrics[f"{prefix}/entropy"] = entropy_acc / max(n_updates, 1)

        self.update_idx += 1
        return metrics

    def evaluate(self, n_episodes: int | None = None, seed: int | None = None) -> dict[str, float]:
        n_episodes = n_episodes or self.cfg.eval_episodes
        seed = self.cfg.seed + 10_000 if seed is None else seed
        costs = []
        local = {r: [] for r in self.roles}
        share_rates = []
        honesty_scores = []
        order_series = {r: [] for r in self.roles}
        demand_series = []
        inflation_flags = []
        boundary_hits = 0
        boundary_orders = 0
        cap_bind_flags: list[float] = []
        alloc_trigger_flags: list[float] = []
        week_events: list[dict[str, Any]] = []

        for ep in range(n_episodes):
            states = self.core.reset(seed + ep)
            done = False
            sys_acc = 0.0
            loc_acc = {r: 0.0 for r in self.roles}
            broadcasts = 0
            broadcast_opps = 0
            mae_sum = 0.0
            mae_n = 0
            steps = 0
            while not done:
                orders: dict[Role, int] = {}
                signals: dict[Role, Signal | None] | None = {} if self.signaling else None
                with torch.no_grad():
                    for r in self.roles:
                        o = torch.as_tensor(
                            self._obs(states, r, self.core), device=self.device
                        ).unsqueeze(0)
                        a, _, _ = self._policy_act(r, o, greedy=not self.signaling)
                        if self.signaling:
                            row = a.squeeze(0).cpu().numpy().astype(int)
                            orders[r] = self._decode_order(int(row[0]), states[r])
                            assert signals is not None
                            signals[r] = self._decode_signal(
                                states[r], int(row[1]), int(row[2]), int(row[3])
                            )
                        else:
                            orders[r] = self._decode_order(int(a.item()), states[r])
                states, rewards, done, info = self.core.step(orders, signals)
                sys_acc += info.system_cost
                cap_b, alloc_b = _bind_flags(info)
                cap_bind_flags.append(1.0 if cap_b else 0.0)
                alloc_trigger_flags.append(1.0 if alloc_b else 0.0)
                week_events.append(
                    {
                        "ep": ep,
                        "t": self.core.t,
                        "capacity_binds": cap_b,
                        "allocation_triggers": alloc_b,
                        "rationed": bool(info.rationed),
                        "factory_production": int(info.factory_production),
                    }
                )
                for r in self.roles:
                    loc_acc[r] += info.local_costs[r]
                    order_series[r].append(info.orders_placed[r])
                    boundary_orders += 1
                    if info.orders_placed[r] == self.cfg.order_cap:
                        boundary_hits += 1
                # Consumer demand proxy: sum over customer-facing roles.
                cust = info.customer_demand
                if cust is None and info.customer_demands:
                    cust = int(sum(info.customer_demands.values()))
                if cust is None:
                    # serial fallback
                    from beer_distribution_rl.env.core_types import Role as R

                    cust = info.incoming_orders.get(R.RETAILER, 0)
                demand_series.append(int(cust or 0))
                if self.signaling:
                    for r in self.roles:
                        broadcast_opps += 1
                        if info.signals_sent.get(r) is not None:
                            broadcasts += 1
                        h = info.honesty.get(r, {})
                        mae = h.get("mean_abs_error", float("nan"))
                        if mae == mae:
                            mae_sum += float(mae)
                            mae_n += 1
                    if info.rationed:
                        # Inflation at factory (or first factory role)
                        frole = self.core.topology.factories[0]
                        need = info.incoming_orders.get(frole, 0)
                        if need > 0 and info.orders_placed[frole] > 1.5 * need:
                            inflation_flags.append(1.0)
                        else:
                            inflation_flags.append(0.0)
                steps += 1
            costs.append(sys_acc / steps)
            for r in self.roles:
                local[r].append(loc_acc[r] / steps)
            if self.signaling and broadcast_opps:
                share_rates.append(broadcasts / broadcast_opps)
            if mae_n:
                honesty_scores.append(-(mae_sum / mae_n) / max(self.cfg.order_cap, 1))

        frac_at_cap = boundary_hits / max(boundary_orders, 1)
        warn_if_boundary_saturated(
            frac_at_cap,
            threshold=self.cfg.boundary_warn_frac,
            context=f"IPPO eval regime={self.cfg.regime}",
        )

        out: dict[str, float] = {
            "eval/mean_system_cost": float(np.mean(costs)),
            "eval/std_system_cost": float(np.std(costs)),
            "eval/frac_actions_at_cap": float(frac_at_cap),
            "eval/frac_capacity_binds": float(np.mean(cap_bind_flags)) if cap_bind_flags else 0.0,
            "eval/frac_allocation_triggers": (
                float(np.mean(alloc_trigger_flags)) if alloc_trigger_flags else 0.0
            ),
        }
        for r in self.roles:
            name = self.core.role_names.get(r, r.name.lower())
            out[f"eval/{name}_cost"] = float(np.mean(local[r]))
        dvar = float(np.var(demand_series)) if len(demand_series) > 1 else 0.0
        for r in self.roles:
            name = self.core.role_names.get(r, r.name.lower())
            ovar = float(np.var(order_series[r])) if len(order_series[r]) > 1 else 0.0
            out[f"eval/bullwhip_{name}"] = ovar / dvar if dvar > 1e-12 else float("inf")
        if self.signaling:
            out["eval/sharing_rate"] = float(np.mean(share_rates)) if share_rates else 0.0
            out["eval/honesty_score"] = (
                float(np.mean(honesty_scores)) if honesty_scores else 0.0
            )
            out["eval/inflation_rate"] = (
                float(np.mean(inflation_flags)) if inflation_flags else 0.0
            )
        # Stash week events for train() to persist (closes D5 log gap).
        self._last_week_events = week_events
        return out

    def train(self) -> Path:
        cfg = self.cfg
        name = cfg.run_name or default_run_name(cfg)
        out = Path(cfg.out_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        demand_info = info_value_table(horizon=cfg.horizon, n_traj=500, seed=cfg.seed)
        cap_rec = recommend_order_cap(
            self.env_config.demand,
            delta_max=cfg.action_delta_max,
            horizon=cfg.horizon,
        )
        meta = {
            "config": asdict(cfg),
            "git_sha": git_sha(),
            "obs_dim": self.obs_dim,
            "n_order": self.n_order,
            "signaling": self.signaling,
            "capacity": self.env_config.capacity,
            "topology": self.core.topology.name,
            "roles": [self.core.role_names[r] for r in self.roles],
            "independent_policies": True,
            "shared_critic": False,
            "demand_info_value": demand_info,
            "order_cap_recommendation": cap_rec,
        }
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
        (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
        (out / "demand_info_value.json").write_text(json.dumps(demand_info, indent=2))

        t0 = time.time()
        while self.global_step < cfg.total_timesteps:
            roll = self.collect_rollout()
            upd = self.update()
            row = {"step": self.global_step, "update": self.update_idx, **roll, **upd}
            if self.update_idx % cfg.eval_every == 0:
                row.update(self.evaluate())
            self.history.append(row)
            if self.update_idx % cfg.log_every == 0:
                cost = row.get("eval/mean_system_cost", row.get("rollout_mean_system_cost"))
                extra = ""
                if "eval/honesty_score" in row:
                    extra = (
                        f" honesty={row['eval/honesty_score']:.3f} "
                        f"share={row['eval/sharing_rate']:.2f}"
                    )
                if "eval/frac_actions_at_cap" in row:
                    extra += f" at_cap={row['eval/frac_actions_at_cap']:.3f}"
                if "eval/frac_capacity_binds" in row:
                    extra += (
                        f" cap_bind={row['eval/frac_capacity_binds']:.2f} "
                        f"alloc={row['eval/frac_allocation_triggers']:.2f}"
                    )
                print(
                    f"[IPPO R{cfg.regime} {cfg.topology} cap={capacity_tag(cfg)} "
                    f"{cfg.rationing} {cfg.demand}] "
                    f"u={self.update_idx} step={self.global_step} cost={cost:.3f}{extra} "
                    f"t={time.time()-t0:.0f}s"
                )

        self.save(out / "checkpoints")
        (out / "history.json").write_text(json.dumps(self.history, indent=2))
        final_eval = self.evaluate(n_episodes=max(20, cfg.eval_episodes))
        (out / "final_eval.json").write_text(json.dumps(final_eval, indent=2))
        # Per-week bind/allocation events for D5 (no checkpoint recompute).
        week_events = getattr(self, "_last_week_events", [])
        (out / "week_events.json").write_text(json.dumps(week_events, indent=2))
        print("Final eval:", final_eval)
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for r in self.roles:
            name = self.core.role_names.get(r, r.name.lower())
            torch.save(self.policies[r].state_dict(), path / f"policy_{name}.pt")

    def load(self, path: Path) -> None:
        for r in self.roles:
            name = self.core.role_names.get(r, r.name.lower())
            pt = path / f"policy_{name}.pt"
            # Backward-compat with serial retailer naming.
            if not pt.exists() and r.name == "RETAILER":
                pt = path / "policy_retailer.pt"
            self.policies[r].load_state_dict(
                torch.load(pt, map_location=self.device, weights_only=True)
            )
