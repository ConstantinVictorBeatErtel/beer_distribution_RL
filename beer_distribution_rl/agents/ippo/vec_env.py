"""Synchronous multi-env wrapper over BeerGameCore (no JAX).

One process, N independent cores — batches transitions for PPO so 2×256 MLPs
actually see large batches. GPU helps only after this; cell-level parallelism
is handled by the matrix runner (separate processes).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from beer_distribution_rl.env.core import BeerGameCore, EnvConfig, Role, RoleState, StepInfo
from beer_distribution_rl.env.signals import Signal


class SyncBeerGameVecEnv:
    """N independent ``BeerGameCore`` instances stepped in lockstep."""

    def __init__(self, config: EnvConfig, n_envs: int):
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.n_envs = int(n_envs)
        self.config = config
        # Distinct seeds per sub-env so rollouts are not identical copies.
        self.cores = [
            BeerGameCore(replace(config, seed=(config.seed or 0) + i * 10_007))
            for i in range(self.n_envs)
        ]
        self.roles = self.cores[0].roles
        self._states: list[dict[Role, RoleState]] = [{} for _ in range(self.n_envs)]

    def reset(self, seed: int) -> list[dict[Role, RoleState]]:
        out: list[dict[Role, RoleState]] = []
        for i, core in enumerate(self.cores):
            self._states[i] = core.reset(seed + i)
            out.append(self._states[i])
        return out

    def reset_one(self, env_idx: int, seed: int) -> dict[Role, RoleState]:
        self._states[env_idx] = self.cores[env_idx].reset(seed)
        return self._states[env_idx]

    def step(
        self,
        orders: list[dict[Role, int]],
        signals: list[dict[Role, Signal | None] | None] | None = None,
    ) -> tuple[
        list[dict[Role, RoleState]],
        list[dict[Role, float]],
        list[bool],
        list[StepInfo],
    ]:
        states: list[dict[Role, RoleState]] = []
        rewards: list[dict[Role, float]] = []
        dones: list[bool] = []
        infos: list[StepInfo] = []
        for i, core in enumerate(self.cores):
            sig = None if signals is None else signals[i]
            st, rew, done, info = core.step(orders[i], sig)
            self._states[i] = st
            states.append(st)
            rewards.append(rew)
            dones.append(bool(done))
            infos.append(info)
        return states, rewards, dones, infos

    def observe_role(self, role: Role) -> np.ndarray:
        """Unused helper — trainer featurizes via state_to_obs."""
        raise NotImplementedError

    @property
    def topology_name(self) -> str:
        return self.cores[0].topology.name
