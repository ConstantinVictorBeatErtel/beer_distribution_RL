"""Rollout buffers — one independent buffer per role."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RoleBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list = field(default_factory=list)  # int or list[int] for multi-head
    logprobs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_gae(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        n_envs: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE over time. With n_envs>1, buffer layout is step-major then env:

        ``[t0e0, t0e1, ..., t0eN-1, t1e0, ...]``.
        """
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        n = len(rewards)
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if n % n_envs != 0:
            raise ValueError(f"buffer length {n} not divisible by n_envs={n_envs}")
        t_steps = n // n_envs

        if n_envs == 1:
            values_ext = np.concatenate([values, np.asarray([last_value], dtype=np.float32)])
            advantages = np.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(t_steps)):
                nonterminal = 1.0 - dones[t]
                delta = rewards[t] + gamma * values_ext[t + 1] * nonterminal - values[t]
                gae = delta + gamma * gae_lambda * nonterminal * gae
                advantages[t] = gae
            returns = advantages + values
            return (
                torch.as_tensor(advantages, dtype=torch.float32),
                torch.as_tensor(returns, dtype=torch.float32),
            )

        # Vectorized: reshape to (T, N)
        rew = rewards.reshape(t_steps, n_envs)
        val = values.reshape(t_steps, n_envs)
        don = dones.reshape(t_steps, n_envs)
        # last_value may be scalar (shared bootstrap) or length-N
        if np.ndim(last_value) == 0:
            last = np.full(n_envs, float(last_value), dtype=np.float32)
        else:
            last = np.asarray(last_value, dtype=np.float32)
            if last.shape != (n_envs,):
                raise ValueError(f"last_value shape {last.shape} != ({n_envs},)")
        val_ext = np.concatenate([val, last[None, :]], axis=0)
        advantages = np.zeros_like(rew)
        gae = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(t_steps)):
            nonterminal = 1.0 - don[t]
            delta = rew[t] + gamma * val_ext[t + 1] * nonterminal - val[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns = advantages + val
        return (
            torch.as_tensor(advantages.reshape(-1), dtype=torch.float32),
            torch.as_tensor(returns.reshape(-1), dtype=torch.float32),
        )

    def as_tensors(self) -> dict[str, torch.Tensor]:
        acts = np.asarray(self.actions)
        return {
            "obs": torch.as_tensor(np.stack(self.obs), dtype=torch.float32),
            "actions": torch.as_tensor(acts, dtype=torch.int64),
            "logprobs": torch.as_tensor(np.asarray(self.logprobs), dtype=torch.float32),
            "values": torch.as_tensor(np.asarray(self.values), dtype=torch.float32),
        }
