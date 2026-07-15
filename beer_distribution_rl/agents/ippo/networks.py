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
    """Independent actor + critic for a single role (order action only)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.actor_body = _mlp(obs_dim, hidden)
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_body = _mlp(obs_dim, hidden)
        self.critic_head = nn.Linear(hidden, 1)
        self.signaling = False
        self.recurrent = False
        self._init_order_bias()

    def _init_order_bias(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        with torch.no_grad():
            mid = self.actor_head.out_features // 2
            self.actor_head.bias.zero_()
            self.actor_head.bias[mid] = 1.0

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits = self.actor_head(self.actor_body(obs))
        value = self.critic_head(self.critic_body(obs)).squeeze(-1)
        return Categorical(logits=logits), value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action [B], logprob [B], value [B])."""
        dist, value = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        return dist.log_prob(actions), value, dist.entropy()


class RecurrentActorCritic(nn.Module):
    """Order-only actor-critic with a GRU over own local observations.

    Memory-matched baseline for LLM agents that retain structured own-history
    (Check 3 / Check 4 of llm_tier_readiness). Input each week is the same
    local obs vector as the Markovian MLP (own inv/backlog/orders/pipelines
    only — E1 no-leak). The GRU hidden state is the learned analogue of the
    LLM's retained multi-week trajectory.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 256,
        gru_hidden: int = 128,
    ):
        super().__init__()
        self.signaling = False
        self.recurrent = True
        self.obs_dim = obs_dim
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(obs_dim, gru_hidden, batch_first=True)
        self.actor_body = _mlp(gru_hidden, hidden)
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_body = _mlp(gru_hidden, hidden)
        self.critic_head = nn.Linear(hidden, 1)
        self._init_order_bias()

    def _init_order_bias(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)
        for name, p in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        with torch.no_grad():
            mid = self.actor_head.out_features // 2
            self.actor_head.bias.zero_()
            self.actor_head.bias[mid] = 1.0

    def initial_hidden(self, batch: int, device: torch.device | None = None) -> torch.Tensor:
        """Zero hidden state [1, B, H] (GRU num_layers=1)."""
        return torch.zeros(1, batch, self.gru_hidden, device=device)

    def forward(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        """obs [B, D], h [1, B, H] → dist, value [B], h_new [1, B, H]."""
        x = obs.unsqueeze(1)
        out, h_new = self.gru(x, h)
        feat = out.squeeze(1)
        logits = self.actor_head(self.actor_body(feat))
        value = self.critic_head(self.critic_body(feat)).squeeze(-1)
        return Categorical(logits=logits), value, h_new

    def act(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action [B], logprob [B], value [B], h_new)."""
        dist, value, h_new = self.forward(obs, h)
        action = dist.sample()
        return action, dist.log_prob(action), value, h_new

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step BPTT with stored (detached) input hiddens."""
        dist, value, _ = self.forward(obs, h)
        return dist.log_prob(actions), value, dist.entropy()


class SignalingActorCritic(nn.Module):
    """Order + optional cheap-talk heads (Regime B). Still one module per role."""

    def __init__(
        self,
        obs_dim: int,
        n_order: int,
        n_claim: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.signaling = True
        self.recurrent = False
        self.actor_body = _mlp(obs_dim, hidden)
        self.order_head = nn.Linear(hidden, n_order)
        self.broadcast_head = nn.Linear(hidden, 2)
        self.claim_demand_head = nn.Linear(hidden, n_claim)
        self.claim_inventory_head = nn.Linear(hidden, n_claim)
        self.critic_body = _mlp(obs_dim, hidden)
        self.critic_head = nn.Linear(hidden, 1)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)
        for head in (
            self.order_head,
            self.broadcast_head,
            self.claim_demand_head,
            self.claim_inventory_head,
        ):
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        with torch.no_grad():
            mid = self.order_head.out_features // 2
            self.order_head.bias.zero_()
            self.order_head.bias[mid] = 1.0
            # Neutral broadcast prior — sharing must be learned, not suppressed.
            self.broadcast_head.bias.zero_()
            claim_mid = self.claim_demand_head.out_features // 2
            self.claim_demand_head.bias.zero_()
            self.claim_inventory_head.bias.zero_()
            self.claim_demand_head.bias[claim_mid] = 0.5
            self.claim_inventory_head.bias[claim_mid] = 0.5

    def _dists(self, obs: torch.Tensor):
        h = self.actor_body(obs)
        return (
            Categorical(logits=self.order_head(h)),
            Categorical(logits=self.broadcast_head(h)),
            Categorical(logits=self.claim_demand_head(h)),
            Categorical(logits=self.claim_inventory_head(h)),
            self.critic_head(self.critic_body(obs)).squeeze(-1),
        )

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns actions [B,4], logprob sum [B], value [B]."""
        o, b, cd, ci, value = self._dists(obs)
        ao, ab, acd, aci = o.sample(), b.sample(), cd.sample(), ci.sample()
        logp = o.log_prob(ao) + b.log_prob(ab) + cd.log_prob(acd) + ci.log_prob(aci)
        actions = torch.stack([ao, ab, acd, aci], dim=-1)
        return actions, logp, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o, b, cd, ci, value = self._dists(obs)
        ao, ab, acd, aci = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3]
        logp = o.log_prob(ao) + b.log_prob(ab) + cd.log_prob(acd) + ci.log_prob(aci)
        ent = o.entropy() + b.entropy() + cd.entropy() + ci.entropy()
        return logp, value, ent

    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        o, b, cd, ci, _ = self._dists(obs)
        return torch.stack(
            [o.probs.argmax(-1), b.probs.argmax(-1), cd.probs.argmax(-1), ci.probs.argmax(-1)],
            dim=-1,
        )
