"""Rationing mechanism tests."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from beer_distribution_rl.env.core_types import Role
from beer_distribution_rl.env.rationing import (
    HonestyWeightedRationing,
    ProportionalRationing,
    RationContext,
    UniformRationing,
)


ROLES3 = (Role.RETAILER, Role.WHOLESALER, Role.DISTRIBUTOR)


@given(
    reqs=st.lists(st.integers(0, 50), min_size=3, max_size=3),
    available=st.integers(0, 100),
)
@settings(max_examples=80, deadline=None)
def test_proportional_invariants(reqs, available):
    requested = {ROLES3[i]: reqs[i] for i in range(3)}
    out = ProportionalRationing().allocate(requested, available)
    assert sum(out.values()) <= available
    for r in requested:
        assert 0 <= out[r] <= requested[r]


@given(
    reqs=st.lists(st.integers(0, 50), min_size=3, max_size=3),
    available=st.integers(0, 100),
)
@settings(max_examples=80, deadline=None)
def test_uniform_invariants(reqs, available):
    requested = {ROLES3[i]: reqs[i] for i in range(3)}
    out = UniformRationing().allocate(requested, available)
    assert sum(out.values()) <= available
    for r in requested:
        assert 0 <= out[r] <= requested[r]


@given(
    reqs=st.lists(st.integers(0, 50), min_size=3, max_size=3),
    available=st.integers(0, 100),
)
@settings(max_examples=80, deadline=None)
def test_honesty_weighted_invariants(reqs, available):
    requested = {ROLES3[i]: reqs[i] for i in range(3)}
    ctx = RationContext(
        honesty_ema={
            Role.RETAILER: 0.0,
            Role.WHOLESALER: -5.0,
            Role.DISTRIBUTOR: -1.0,
        }
    )
    out = HonestyWeightedRationing().allocate(requested, available, ctx)
    assert sum(out.values()) <= available
    for r in requested:
        assert 0 <= out[r] <= requested[r]


def test_identity_when_enough_supply():
    requested = {Role.RETAILER: 3, Role.WHOLESALER: 5, Role.DISTRIBUTOR: 2}
    for policy in (ProportionalRationing(), UniformRationing(), HonestyWeightedRationing()):
        assert policy.allocate(requested, 100) == requested


def test_proportional_splits():
    requested = {Role.RETAILER: 10, Role.WHOLESALER: 10}
    out = ProportionalRationing().allocate(requested, 10)
    assert out[Role.RETAILER] == 5 and out[Role.WHOLESALER] == 5


def test_honesty_prefers_honest():
    requested = {Role.RETAILER: 10, Role.WHOLESALER: 10}
    ctx = RationContext(honesty_ema={Role.RETAILER: 0.0, Role.WHOLESALER: -10.0})
    out = HonestyWeightedRationing().allocate(requested, 10, ctx)
    assert out[Role.RETAILER] > out[Role.WHOLESALER]


def test_proportional_strictly_rewards_larger_order_under_shortage():
    """Shortage-gaming incentive: larger claim gets strictly more units."""
    requested = {Role.RETAILER: 30, Role.RETAILER_B: 10}
    out = ProportionalRationing().allocate(requested, 20)
    assert sum(out.values()) == 20
    assert out[Role.RETAILER] > out[Role.RETAILER_B]
    assert out[Role.RETAILER] == 15 and out[Role.RETAILER_B] == 5


def test_uniform_ignores_order_size_under_shortage():
    requested = {Role.RETAILER: 30, Role.RETAILER_B: 10}
    out = UniformRationing().allocate(requested, 20)
    assert sum(out.values()) == 20
    assert out[Role.RETAILER] == 10 and out[Role.RETAILER_B] == 10


def test_honesty_weighted_responds_to_accuracy_history():
    """Equal requests: more honest EMA ⇒ strictly larger allocation."""
    requested = {Role.RETAILER: 20, Role.RETAILER_B: 20}
    ctx = RationContext(
        honesty_ema={Role.RETAILER: 0.0, Role.RETAILER_B: -8.0}  # A more honest
    )
    out = HonestyWeightedRationing().allocate(requested, 20, ctx)
    assert sum(out.values()) == 20
    assert out[Role.RETAILER] > out[Role.RETAILER_B]


def test_serial_single_claimant_policies_identical():
    """D1/P3 untestable on serial: all three mechanisms = identity fill."""
    # Single claimant is what every serial-chain node looks like.
    requested = {Role.RETAILER: 17}
    available = 10
    props = [
        ProportionalRationing().allocate(requested, available),
        UniformRationing().allocate(requested, available),
        HonestyWeightedRationing().allocate(
            requested,
            available,
            RationContext(honesty_ema={Role.RETAILER: -99.0}),
        ),
    ]
    assert props[0] == props[1] == props[2] == {Role.RETAILER: 10}


def test_honesty_weighted_does_not_enter_reward():
    """Mechanism changes fill dynamics only — Regime A/B rewards stay local costs."""
    from beer_distribution_rl.env.core import y_topology_env_config, BeerGameCore
    from beer_distribution_rl.env.core_types import Y_ROLES
    from beer_distribution_rl.env.demand import CorrelatedYDemand

    cfg = y_topology_env_config(
        horizon=5,
        demand=CorrelatedYDemand(mu=8.0, phi=0.0, sigma_common=0.0, sigma_idio=0.0),
        rationing=HonestyWeightedRationing(),
        regime="B",
        signaling_enabled=True,
        seed=0,
        init_inventory=(5, 5, 5, 5, 5),
        init_pipeline_ship=0,
        init_pipeline_order=0,
        capacity=None,
    )
    env = BeerGameCore(cfg)
    env.reset(0)
    env._honesty_ema[Role.RETAILER] = 0.0
    env._honesty_ema[Role.RETAILER_B] = -10.0
    orders = {r: 10 for r in Y_ROLES}
    _, rewards, _, info = env.step(orders, signals={r: None for r in Y_ROLES})
    for r in Y_ROLES:
        assert rewards[r] == pytest.approx(-info.local_costs[r])
        assert rewards[r] == pytest.approx(
            -(
                env.config.costs[int(r)].holding * env.states[r].inventory
                + env.config.costs[int(r)].backlog * env.states[r].backlog
            )
        )


def test_capacity_causes_shortage_in_core():
    from beer_distribution_rl.env.core import BeerGameCore, EnvConfig, ROLES
    from beer_distribution_rl.env.demand import ClassicStepDemand

    env = BeerGameCore(
        EnvConfig(
            horizon=20,
            capacity=3,  # tight vs demand 4→8
            demand=ClassicStepDemand(),
            seed=0,
        )
    )
    env.reset(0)
    saw_ration = False
    done = False
    while not done:
        _, _, done, info = env.step({r: 20 for r in ROLES})
        if info.rationed or info.factory_production < 20:
            saw_ration = True
    assert saw_ration
    assert info.factory_production <= 3
