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
