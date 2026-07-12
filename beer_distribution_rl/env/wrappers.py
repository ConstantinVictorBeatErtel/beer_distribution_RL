"""PettingZoo ParallelEnv and Gymnasium single-agent wrappers."""

from __future__ import annotations

from typing import Any

from beer_distribution_rl.agents.baselines import CLASSIC_BASE_STOCK_VECTOR, base_stock_order
from beer_distribution_rl.env.core import (
    BeerGameCore,
    EnvConfig,
    Role,
    ROLES,
    Signal,
    classic_env_config,
)
from beer_distribution_rl.env.core_types import ROLE_NAMES

try:
    import gymnasium as gym
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "wrappers require pettingzoo and gymnasium; install with "
        'pip install -e ".[wrappers]"'
    ) from exc


AGENT_NAMES = [ROLE_NAMES[r] for r in ROLES]
NAME_TO_ROLE = {ROLE_NAMES[r]: r for r in ROLES}


def _obs_from_core(env: BeerGameCore, role: Role) -> dict[str, Any]:
    return env.observe(role)


class BeerGameParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv over BeerGameCore.

    Rewards come strictly from the core (local for A/B, system for C).
    No reward reshaping in the wrapper.
    """

    metadata = {"name": "beer_game_v0", "render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self._config = config or classic_env_config()
        self.core = BeerGameCore(self._config)
        self.possible_agents = list(AGENT_NAMES)
        self.agents = list(self.possible_agents)
        self._action_spaces = {
            name: self._make_action_space() for name in self.possible_agents
        }
        self._observation_spaces = {
            name: self._make_obs_space() for name in self.possible_agents
        }

    def _make_action_space(self):
        if self._config.signaling_enabled:
            # order, claimed_demand (-1=null), claimed_inventory (-1=null)
            return spaces.Dict(
                {
                    "order": spaces.Discrete(self._config.order_cap + 1),
                    "claimed_demand": spaces.Discrete(self._config.order_cap + 2),
                    "claimed_inventory": spaces.Discrete(self._config.order_cap + 2),
                    "broadcast": spaces.Discrete(2),
                }
            )
        return spaces.Discrete(self._config.order_cap + 1)

    def _make_obs_space(self):
        cap = float(self._config.order_cap)
        return spaces.Dict(
            {
                "inventory": spaces.Box(0, 1e6, shape=(), dtype=float),
                "backlog": spaces.Box(0, 1e6, shape=(), dtype=float),
                "on_order": spaces.Box(0, 1e6, shape=(), dtype=float),
                "last_demand_or_order": spaces.Box(0, cap, shape=(), dtype=float),
                "t": spaces.Box(0, float(self._config.horizon), shape=(), dtype=float),
            }
        )

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = list(self.possible_agents)
        states = self.core.reset(seed)
        obs = {name: self._vector_obs(NAME_TO_ROLE[name]) for name in self.agents}
        infos = {name: {} for name in self.agents}
        return obs, infos

    def _vector_obs(self, role: Role) -> dict:
        raw = self.core.observe(role)
        return {
            "inventory": float(raw["inventory"]),
            "backlog": float(raw["backlog"]),
            "on_order": float(self.core._states[role].on_order),
            "last_demand_or_order": float(raw["last_demand_or_order"]),
            "t": float(raw["t"]),
        }

    def step(self, actions: dict):
        orders = {}
        signals = {} if self._config.signaling_enabled else None
        for name in self.agents:
            role = NAME_TO_ROLE[name]
            act = actions[name]
            if self._config.signaling_enabled and isinstance(act, dict):
                orders[role] = int(act["order"])
                if int(act.get("broadcast", 0)) == 1:
                    cd = int(act["claimed_demand"])
                    ci = int(act["claimed_inventory"])
                    signals[role] = Signal(
                        claimed_demand=None if cd == 0 else cd - 1,
                        claimed_inventory=None if ci == 0 else ci - 1,
                    )
                else:
                    signals[role] = None
            else:
                orders[role] = int(act)
                if signals is not None:
                    signals[role] = None

        states, rewards_core, terminated, info = self.core.step(orders, signals)
        obs = {name: self._vector_obs(NAME_TO_ROLE[name]) for name in self.agents}
        rewards = {name: float(rewards_core[NAME_TO_ROLE[name]]) for name in self.agents}
        terminations = {name: terminated for name in self.agents}
        truncations = {name: False for name in self.agents}
        infos = {name: {"system_cost": info.system_cost} for name in self.agents}
        if terminated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos


class BeerGameSingleAgentEnv(gym.Env):
    """Control one role; others follow base-stock (debug helper)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        controlled_role: Role = Role.RETAILER,
        config: EnvConfig | None = None,
        base_stock_levels: tuple[int, ...] = CLASSIC_BASE_STOCK_VECTOR,
    ):
        super().__init__()
        self.controlled_role = controlled_role
        self._config = config or classic_env_config()
        self.core = BeerGameCore(self._config)
        self.levels = base_stock_levels
        self.action_space = spaces.Discrete(self._config.order_cap + 1)
        self.observation_space = spaces.Box(0, 1e6, shape=(5,), dtype=float)

    def _obs(self):
        s = self.core._states[self.controlled_role]
        return [
            float(s.inventory),
            float(s.backlog),
            float(s.on_order),
            float(s.last_demand_or_order),
            float(self.core.t),
        ]

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.core.reset(seed)
        return self._obs(), {}

    def step(self, action):
        orders = {}
        for r in ROLES:
            if r == self.controlled_role:
                orders[r] = int(action)
            else:
                orders[r] = base_stock_order(
                    self.core._states[r], self.levels[int(r)], self._config.order_cap
                )
        _, rewards, terminated, info = self.core.step(orders)
        return (
            self._obs(),
            float(rewards[self.controlled_role]),
            terminated,
            False,
            {"system_cost": info.system_cost},
        )
