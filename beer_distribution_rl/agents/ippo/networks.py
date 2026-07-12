"""Per-role actor-critic networks. NEVER share weights across roles."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(in_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
    )


class ActorCritic(nn.Module):
    """Independent actor + critic for a single role (no cross-role sharing)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.actor_body = _mlp(obs_dim, hidden)
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_body = _mlp(obs_dim, hidden)
        self.critic_head = nn.Linear(hidden, 1)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        # Mild bias toward center action (Δ=0 for relative policies).
        with torch.no_grad():
            mid = self.actor_head.out_features // 2
            self.actor_head.bias.zero_()
            self.actor_head.bias[mid] = 1.0

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits = self.actor_head(self.actor_body(obs))
        value = self.critic_head(self.critic_body(obs)).squeeze(-1)
        return Categorical(logits=logits), value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        return dist.log_prob(actions), value, dist.entropy()
