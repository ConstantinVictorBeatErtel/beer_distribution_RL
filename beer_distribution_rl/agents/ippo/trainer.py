"""Independent PPO trainer: one ActorCritic per role, no shared parameters."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from beer_distribution_rl.agents.ippo.buffer import RoleBuffer
from beer_distribution_rl.agents.ippo.networks import ActorCritic
from beer_distribution_rl.agents.ippo.obs import obs_dim, state_to_obs
from beer_distribution_rl.env.core import (
    BeerGameCore,
    EnvConfig,
    Role,
    ROLES,
    classic_env_config,
)
from beer_distribution_rl.env.demand import ClassicStepDemand, UniformDemand


@dataclass
class IPPOConfig:
    regime: str = "A"  # A or C only for M2 (B is M3+)
    horizon: int = 52
    total_timesteps: int = 200_000
    n_envs: int = 1  # sequential episodes; kept for future vectorization
    rollout_steps: int = 2048  # steps across roles counted per env-step
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
    # Relative actions: order = clip(last_demand + delta, 0, order_cap),
    # delta ∈ [-action_delta_max, +action_delta_max]. Matches DQN-beer-game practice.
    action_mode: str = "relative"  # relative | absolute
    action_delta_max: int = 8
    reward_scale: float = 0.1
    demand: str = "classic_step"  # classic_step | uniform
    seed: int = 0
    device: str = "cpu"
    log_every: int = 5
    eval_every: int = 10
    eval_episodes: int = 10
    out_dir: str = "artifacts/runs/ippo"


def build_env_config(cfg: IPPOConfig) -> EnvConfig:
    if cfg.regime not in ("A", "C"):
        raise ValueError("M2 IPPO supports regimes A and C only (B waits for phase diagram)")
    demand: Any
    if cfg.demand == "uniform":
        demand = UniformDemand(0, 15)
    elif cfg.demand == "classic_step":
        demand = ClassicStepDemand()
    else:
        raise ValueError(f"unknown demand: {cfg.demand}")
    return classic_env_config(
        horizon=cfg.horizon,
        demand=demand,
        regime=cfg.regime,  # type: ignore[arg-type]
        signaling_enabled=False,
        order_cap=cfg.order_cap,
        seed=cfg.seed,
    )


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "UNKNOWN"


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
            self.n_actions = 2 * cfg.action_delta_max + 1
        elif cfg.action_mode == "absolute":
            self.n_actions = cfg.order_cap + 1
        else:
            raise ValueError(f"unknown action_mode: {cfg.action_mode}")
        self.obs_dim = obs_dim(self.env_config)

        # CRITICAL: four separate modules — never share parameters or critics.
        self.policies: dict[Role, ActorCritic] = {
            r: ActorCritic(self.obs_dim, self.n_actions, cfg.hidden).to(self.device)
            for r in ROLES
        }
        self.optimizers: dict[Role, torch.optim.Optimizer] = {
            r: torch.optim.Adam(self.policies[r].parameters(), lr=cfg.learning_rate, eps=1e-5)
            for r in ROLES
        }
        self.buffers: dict[Role, RoleBuffer] = {r: RoleBuffer() for r in ROLES}
        self._assert_independent_params()

        self.global_step = 0
        self.update_idx = 0
        self.history: list[dict[str, Any]] = []

    def _assert_independent_params(self) -> None:
        """Reject accidental parameter sharing across roles."""
        ids = []
        for r in ROLES:
            for p in self.policies[r].parameters():
                ids.append(id(p))
        if len(ids) != len(set(ids)):
            raise RuntimeError("Parameter sharing detected across roles — invalidates emergence claim")
        # Critics must also be distinct modules (already implied by separate ActorCritic).
        critic_ids = [id(self.policies[r].critic_head) for r in ROLES]
        if len(set(critic_ids)) != 4:
            raise RuntimeError("Shared critic detected across roles")

    def _obs(self, states, role: Role) -> np.ndarray:
        return state_to_obs(states[role], role, self.core)

    def _decode_action(self, role: Role, action_idx: int, state) -> int:
        if self.cfg.action_mode == "absolute":
            return int(action_idx)
        delta = int(action_idx) - self.cfg.action_delta_max
        raw = int(state.last_demand_or_order) + delta
        return max(0, min(self.cfg.order_cap, raw))

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
            actions: dict[Role, int] = {}
            action_idx: dict[Role, int] = {}
            logprobs: dict[Role, float] = {}
            values: dict[Role, float] = {}
            obs_np: dict[Role, np.ndarray] = {}

            with torch.no_grad():
                for r in ROLES:
                    o = self._obs(states, r)
                    obs_np[r] = o
                    ot = torch.as_tensor(o, device=self.device).unsqueeze(0)
                    a_idx, lp, val = self.policies[r].act(ot)
                    action_idx[r] = int(a_idx.item())
                    actions[r] = self._decode_action(r, action_idx[r], states[r])
                    logprobs[r] = float(lp.item())
                    values[r] = float(val.item())

            states, rewards, done, info = self.core.step(actions)
            ep_cost_acc += info.system_cost
            ep_len += 1
            self.global_step += 1
            steps += 1

            for r in ROLES:
                self.buffers[r].obs.append(obs_np[r])
                self.buffers[r].actions.append(action_idx[r])
                self.buffers[r].logprobs.append(logprobs[r])
                self.buffers[r].rewards.append(float(rewards[r]) * cfg.reward_scale)
                self.buffers[r].dones.append(bool(done))
                self.buffers[r].values.append(values[r])

            if done:
                ep_system_costs.append(ep_cost_acc / max(ep_len, 1))
                ep_cost_acc = 0.0
                ep_len = 0
                states = self.core.reset(cfg.seed + self.global_step)

        # bootstrap values for incomplete episodes at rollout end
        self._bootstrap: dict[Role, float] = {}
        with torch.no_grad():
            for r in ROLES:
                o = torch.as_tensor(self._obs(states, r), device=self.device).unsqueeze(0)
                _, v = self.policies[r].forward(o)
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
            policy_loss_acc = 0.0
            value_loss_acc = 0.0
            entropy_acc = 0.0
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
        for ep in range(n_episodes):
            states = self.core.reset(seed + ep)
            done = False
            sys_acc = 0.0
            loc_acc = {r: 0.0 for r in ROLES}
            steps = 0
            while not done:
                actions = {}
                with torch.no_grad():
                    for r in ROLES:
                        o = torch.as_tensor(self._obs(states, r), device=self.device).unsqueeze(0)
                        dist, _ = self.policies[r].forward(o)
                        a_idx = int(dist.probs.argmax(dim=-1).item())
                        actions[r] = self._decode_action(r, a_idx, states[r])
                states, rewards, done, info = self.core.step(actions)
                sys_acc += info.system_cost
                for r in ROLES:
                    loc_acc[r] += info.local_costs[r]
                steps += 1
            costs.append(sys_acc / steps)
            for r in ROLES:
                local[r].append(loc_acc[r] / steps)
        out = {
            "eval/mean_system_cost": float(np.mean(costs)),
            "eval/std_system_cost": float(np.std(costs)),
        }
        for r in ROLES:
            out[f"eval/{r.name.lower()}_cost"] = float(np.mean(local[r]))
        return out

    def train(self) -> Path:
        cfg = self.cfg
        out = Path(cfg.out_dir) / f"regime{cfg.regime}_seed{cfg.seed}"
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "config": asdict(cfg),
            "git_sha": git_sha(),
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
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
                print(
                    f"[IPPO R{cfg.regime}] update={self.update_idx} "
                    f"step={self.global_step} cost={cost:.3f} "
                    f"elapsed={time.time()-t0:.0f}s"
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
                torch.load(path / f"policy_{r.name.lower()}.pt", map_location=self.device)
            )
