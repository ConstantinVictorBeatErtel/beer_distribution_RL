"""Wrapper smoke tests (skipped if pettingzoo/gymnasium not installed)."""

from __future__ import annotations

import pytest

from beer_distribution_rl.env.core import ROLES, classic_env_config

pz = pytest.importorskip("pettingzoo")
gym = pytest.importorskip("gymnasium")

from beer_distribution_rl.env.wrappers import (  # noqa: E402
    BeerGameParallelEnv,
    BeerGameSingleAgentEnv,
)


def test_parallel_env_smoke():
    env = BeerGameParallelEnv(classic_env_config(horizon=5, seed=0))
    obs, infos = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents)
    done = False
    steps = 0
    while env.agents:
        actions = {a: 4 for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        steps += 1
        assert all(isinstance(rewards[a], float) for a in rewards)
    assert steps == 5


def test_single_agent_smoke():
    env = BeerGameSingleAgentEnv(config=classic_env_config(horizon=5, seed=1))
    obs, _ = env.reset(seed=1)
    assert len(obs) == 5
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(4)
    assert term
