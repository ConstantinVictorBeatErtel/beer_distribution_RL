"""Independent PPO trainer: one ActorCritic per role, no shared parameters."""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from beer_distribution_rl.agents.ippo.buffer import RoleBuffer
from beer_distribution_rl.agents.ippo.networks import ActorCritic, SignalingActorCritic
from beer_distribution_rl.agents.ippo.obs import obs_dim, state_to_obs
from beer_distribution_rl.env.core import BeerGameCore, EnvConfig, Role, ROLES, classic_env_config
from beer_distribution_rl.env.demand import ClassicStepDemand, UniformDemand, mean_demand
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
    n_envs: int = 1
    rollout_steps: int = 1024
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
    order_cap: int = 64
    action_mode: str = "relative"  # relative | absolute
    action_delta_max: int = 8
    claim_delta_max: int = 8
    reward_scale: float = 0.1
    demand: str = "uniform"  # classic_step | uniform
    # Capacity: None = ∞; else multiplier of mean demand (1.5, 1.2, 1.0, 0.8)
    capacity_mult: float | None = None
    rationing: str = "proportional"  # proportional | uniform | honesty_weighted
    seed: int = 0
    device: str = "cpu"
    log_every: int = 5
    eval_every: int = 10
    eval_episodes: int = 10
    out_dir: str = "artifacts/runs/ippo"
    run_name: str | None = None


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
    if cfg.demand == "uniform":
        demand: Any = UniformDemand(0, 15)
    elif cfg.demand == "classic_step":
        demand = ClassicStepDemand()
    else:
        raise ValueError(f"unknown demand: {cfg.demand}")

    capacity = None
    if cfg.capacity_mult is not None:
        mu = mean_demand(demand, cfg.horizon, seed=cfg.seed)
        capacity = float(cfg.capacity_mult) * mu

    return classic_env_config(
        horizon=cfg.horizon,
        demand=demand,
        regime=cfg.regime,  # type: ignore[arg-type]
        signaling_enabled=(cfg.regime == "B"),
        order_cap=cfg.order_cap,
        seed=cfg.seed,
        capacity=capacity,
        rationing=_make_rationing(cfg.rationing),
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
    return f"regime{cfg.regime}_cap{capacity_tag(cfg)}_rat{cfg.rationing}_seed{cfg.seed}"


class IPPOTrainer:
    """CleanRL-style IPPO with strictly independent per-role policies/critics."""

    def __init__(self, cfg: IPPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env_config = build_env_config(cfg)
        self.core = BeerGameCore(self.env_config)
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
        for r in ROLES:
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
            for r in ROLES
        }
        self.buffers = {r: RoleBuffer() for r in ROLES}
        self._assert_independent_params()
        self.global_step = 0
        self.update_idx = 0
        self.history: list[dict[str, Any]] = []
        self._bootstrap: dict[Role, float] = {}

    def _assert_independent_params(self) -> None:
        ids = []
        for r in ROLES:
            for p in self.policies[r].parameters():
                ids.append(id(p))
        if len(ids) != len(set(ids)):
            raise RuntimeError("Parameter sharing detected across roles")
        critic_ids = [id(self.policies[r].critic_head) for r in ROLES]
        if len(set(critic_ids)) != 4:
            raise RuntimeError("Shared critic detected across roles")

    def _obs(self, states, role: Role) -> np.ndarray:
        return state_to_obs(states[role], role, self.core)

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
                # logprob unused
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
        for b in self.buffers.values():
            b.clear()

        states = self.core.reset(cfg.seed + self.global_step)
        ep_system_costs: list[float] = []
        ep_cost_acc = 0.0
        ep_len = 0
        steps = 0

        while steps < cfg.rollout_steps:
            orders: dict[Role, int] = {}
            signals: dict[Role, Signal | None] | None = {} if self.signaling else None
            stored_actions: dict[Role, Any] = {}
            logprobs: dict[Role, float] = {}
            values: dict[Role, float] = {}
            obs_np: dict[Role, np.ndarray] = {}

            with torch.no_grad():
                for r in ROLES:
                    o = self._obs(states, r)
                    obs_np[r] = o
                    ot = torch.as_tensor(o, device=self.device).unsqueeze(0)
                    a, lp, val = self._policy_act(r, ot, greedy=False)
                    values[r] = float(val.item())
                    logprobs[r] = float(lp.item())
                    if self.signaling:
                        row = a.squeeze(0).cpu().numpy().astype(int)
                        stored_actions[r] = row.tolist()
                        orders[r] = self._decode_order(int(row[0]), states[r])
                        assert signals is not None
                        signals[r] = self._decode_signal(
                            states[r], int(row[1]), int(row[2]), int(row[3])
                        )
                    else:
                        idx = int(a.item())
                        stored_actions[r] = idx
                        orders[r] = self._decode_order(idx, states[r])

            states, rewards, done, info = self.core.step(orders, signals)
            ep_cost_acc += info.system_cost
            ep_len += 1
            self.global_step += 1
            steps += 1

            for r in ROLES:
                self.buffers[r].obs.append(obs_np[r])
                self.buffers[r].actions.append(stored_actions[r])
                self.buffers[r].logprobs.append(logprobs[r])
                self.buffers[r].rewards.append(float(rewards[r]) * cfg.reward_scale)
                self.buffers[r].dones.append(bool(done))
                self.buffers[r].values.append(values[r])

            if done:
                ep_system_costs.append(ep_cost_acc / max(ep_len, 1))
                ep_cost_acc = 0.0
                ep_len = 0
                states = self.core.reset(cfg.seed + self.global_step)

        self._bootstrap = {}
        with torch.no_grad():
            for r in ROLES:
                o = torch.as_tensor(self._obs(states, r), device=self.device).unsqueeze(0)
                if self.signaling:
                    self._bootstrap[r] = float(self.policies[r]._dists(o)[4].item())  # type: ignore
                else:
                    _, v = self.policies[r].forward(o)  # type: ignore
                    self._bootstrap[r] = float(v.item())

        mean_ep = float(np.mean(ep_system_costs)) if ep_system_costs else float("nan")
        return {"rollout_mean_system_cost": mean_ep, "episodes": float(len(ep_system_costs))}

    def update(self) -> dict[str, float]:
        cfg = self.cfg
        metrics: dict[str, float] = {}

        for r in ROLES:
            buf = self.buffers[r]
            last_v = 0.0 if (buf.dones and buf.dones[-1]) else self._bootstrap.get(r, 0.0)
            advantages, returns = buf.compute_gae(last_v, cfg.gamma, cfg.gae_lambda)
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

            prefix = r.name.lower()
            metrics[f"{prefix}/policy_loss"] = policy_loss_acc / max(n_updates, 1)
            metrics[f"{prefix}/value_loss"] = value_loss_acc / max(n_updates, 1)
            metrics[f"{prefix}/entropy"] = entropy_acc / max(n_updates, 1)

        self.update_idx += 1
        return metrics

    def evaluate(self, n_episodes: int | None = None, seed: int | None = None) -> dict[str, float]:
        n_episodes = n_episodes or self.cfg.eval_episodes
        seed = self.cfg.seed + 10_000 if seed is None else seed
        costs = []
        local = {r: [] for r in ROLES}
        share_rates = []
        honesty_scores = []  # higher = more honest = more negative MAE mean abs error flipped
        order_series = {r: [] for r in ROLES}
        demand_series = []
        inflation_flags = []

        for ep in range(n_episodes):
            states = self.core.reset(seed + ep)
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
                signals: dict[Role, Signal | None] | None = {} if self.signaling else None
                with torch.no_grad():
                    for r in ROLES:
                        o = torch.as_tensor(self._obs(states, r), device=self.device).unsqueeze(0)
                        # Stochastic eval for Regime B so sharing/honesty reflect the policy
                        # distribution; greedy for A/C cost evaluation.
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
                for r in ROLES:
                    loc_acc[r] += info.local_costs[r]
                    order_series[r].append(info.orders_placed[r])
                demand_series.append(info.incoming_orders[Role.RETAILER])
                if self.signaling:
                    for r in ROLES:
                        broadcast_opps += 1
                        if info.signals_sent.get(r) is not None:
                            broadcasts += 1
                        h = info.honesty.get(r, {})
                        mae = h.get("mean_abs_error", float("nan"))
                        if mae == mae:  # not NaN
                            mae_sum += float(mae)
                            mae_n += 1
                    # inflation detector under rationing weeks
                    if info.rationed:
                        need = info.incoming_orders[Role.FACTORY]
                        if need > 0 and info.orders_placed[Role.FACTORY] > 1.5 * need:
                            inflation_flags.append(1.0)
                        else:
                            inflation_flags.append(0.0)
                steps += 1
            costs.append(sys_acc / steps)
            for r in ROLES:
                local[r].append(loc_acc[r] / steps)
            if self.signaling and broadcast_opps:
                share_rates.append(broadcasts / broadcast_opps)
            if mae_n:
                # honesty score = -mean |claim-truth| (higher better), normalize by order_cap
                honesty_scores.append(- (mae_sum / mae_n) / max(self.cfg.order_cap, 1))

        out: dict[str, float] = {
            "eval/mean_system_cost": float(np.mean(costs)),
            "eval/std_system_cost": float(np.std(costs)),
        }
        for r in ROLES:
            out[f"eval/{r.name.lower()}_cost"] = float(np.mean(local[r]))
        # bullwhip
        dvar = float(np.var(demand_series)) if len(demand_series) > 1 else 0.0
        for r in ROLES:
            ovar = float(np.var(order_series[r])) if len(order_series[r]) > 1 else 0.0
            out[f"eval/bullwhip_{r.name.lower()}"] = ovar / dvar if dvar > 1e-12 else float("inf")
        if self.signaling:
            out["eval/sharing_rate"] = float(np.mean(share_rates)) if share_rates else 0.0
            out["eval/honesty_score"] = (
                float(np.mean(honesty_scores)) if honesty_scores else 0.0
            )
            out["eval/inflation_rate"] = (
                float(np.mean(inflation_flags)) if inflation_flags else 0.0
            )
        return out

    def train(self) -> Path:
        cfg = self.cfg
        name = cfg.run_name or default_run_name(cfg)
        out = Path(cfg.out_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "config": asdict(cfg),
            "git_sha": git_sha(),
            "obs_dim": self.obs_dim,
            "n_order": self.n_order,
            "signaling": self.signaling,
            "capacity": self.env_config.capacity,
            "independent_policies": True,
            "shared_critic": False,
        }
        (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

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
                    extra = f" honesty={row['eval/honesty_score']:.3f} share={row['eval/sharing_rate']:.2f}"
                print(
                    f"[IPPO R{cfg.regime} cap={capacity_tag(cfg)} {cfg.rationing}] "
                    f"u={self.update_idx} step={self.global_step} cost={cost:.3f}{extra} "
                    f"t={time.time()-t0:.0f}s"
                )

        self.save(out / "checkpoints")
        (out / "history.json").write_text(json.dumps(self.history, indent=2))
        final_eval = self.evaluate(n_episodes=max(20, cfg.eval_episodes))
        (out / "final_eval.json").write_text(json.dumps(final_eval, indent=2))
        print("Final eval:", final_eval)
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for r in ROLES:
            torch.save(self.policies[r].state_dict(), path / f"policy_{r.name.lower()}.pt")

    def load(self, path: Path) -> None:
        for r in ROLES:
            self.policies[r].load_state_dict(
                torch.load(path / f"policy_{r.name.lower()}.pt", map_location=self.device, weights_only=True)
            )
