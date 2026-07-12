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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values + [last_value], dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return (
            torch.as_tensor(advantages, dtype=torch.float32),
            torch.as_tensor(returns, dtype=torch.float32),
        )

    def as_tensors(self) -> dict[str, torch.Tensor]:
        acts = np.asarray(self.actions)
        return {
            "obs": torch.as_tensor(np.stack(self.obs), dtype=torch.float32),
            "actions": torch.as_tensor(acts, dtype=torch.int64),
            "logprobs": torch.as_tensor(np.asarray(self.logprobs), dtype=torch.float32),
            "values": torch.as_tensor(np.asarray(self.values), dtype=torch.float32),
        }
