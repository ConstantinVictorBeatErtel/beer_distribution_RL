"""Reward regime and demand-process tests."""

from __future__ import annotations

import random

import pytest

from beer_distribution_rl.env.core import (
    BeerGameCore,
    EnvConfig,
    Role,
    RoleCosts,
    ROLES,
    classic_env_config,
)
from beer_distribution_rl.env.demand import AR1Demand, ClassicStepDemand, UniformDemand


def test_regime_a_local_rewards():
    env = BeerGameCore(classic_env_config(regime="A", seed=0))
    env.reset(0)
    # Asymmetric inventories via different orders over time — just check equality rule
    _, rewards, _, info = env.step({r: 4 for r in ROLES})
    for r in ROLES:
        assert rewards[r] == pytest.approx(-info.local_costs[r])
    # Local may differ across roles
    assert rewards[Role.RETAILER] == pytest.approx(-info.local_costs[Role.RETAILER])


def test_regime_c_system_reward():
    env = BeerGameCore(classic_env_config(regime="C", seed=0))
    env.reset(0)
    _, rewards, _, info = env.step({r: 4 for r in ROLES})
    for r in ROLES:
        assert rewards[r] == pytest.approx(-info.system_cost)
    assert len(set(rewards.values())) == 1


def test_regime_b_still_local():
    env = BeerGameCore(
        classic_env_config(regime="B", signaling_enabled=True, seed=0)
    )
    env.reset(0)
    _, rewards, _, info = env.step({r: 4 for r in ROLES}, signals={r: None for r in ROLES})
    for r in ROLES:
        assert rewards[r] == pytest.approx(-info.local_costs[r])


def test_asymmetric_costs():
    costs = (
        RoleCosts(holding=0.5, backlog=2.0),  # retailer
        RoleCosts(holding=0.5, backlog=1.0),
        RoleCosts(holding=0.5, backlog=1.0),
        RoleCosts(holding=1.0, backlog=0.5),  # factory
    )
    env = BeerGameCore(EnvConfig(horizon=5, costs=costs, seed=0, demand=ClassicStepDemand()))
    env.reset(0)
    # Create backlog at retailer by ordering 0 while demand is 4
    for _ in range(3):
        _, _, _, info = env.step({r: 0 for r in ROLES})
    # Retailer backlog cost should use b=2.0
    st = env.states[Role.RETAILER]
    expected = 0.5 * st.inventory + 2.0 * st.backlog
    assert info.local_costs[Role.RETAILER] == pytest.approx(expected)


def test_classic_step_demand_shape():
    d = ClassicStepDemand()
    rng = random.Random(0)
    d.reset(rng)
    assert [d(t, rng) for t in range(1, 9)] == [4, 4, 4, 4, 8, 8, 8, 8]


def test_uniform_demand_bounds():
    d = UniformDemand(0, 15)
    rng = random.Random(1)
    d.reset(rng)
    vals = [d(t, rng) for t in range(1, 500)]
    assert min(vals) >= 0 and max(vals) <= 15


def test_ar1_nonnegative():
    d = AR1Demand(mu=8, phi=0.5, sigma=3, regime_shift_week=10, mu_after=20)
    rng = random.Random(2)
    d.reset(rng)
    vals = [d(t, rng) for t in range(1, 40)]
    assert all(v >= 0 for v in vals)
    # After shift, mean should rise (noisy but directionally)
    assert sum(vals[15:]) / len(vals[15:]) > sum(vals[:5]) / 5
