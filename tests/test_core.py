"""Unit and property tests for BeerGameCore."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from beer_distribution_rl.env.core import (
    BeerGameCore,
    EnvConfig,
    Role,
    RoleCosts,
    ROLES,
    classic_env_config,
)
from beer_distribution_rl.env.demand import ClassicStepDemand, UniformDemand


def _pass_through_orders(states, cap=64):
    """Order exactly last incoming demand (pass-through)."""
    return {r: min(cap, max(0, states[r].last_demand_or_order)) for r in ROLES}


def test_reset_determinism():
    cfg = classic_env_config(seed=123)
    a = BeerGameCore(cfg)
    b = BeerGameCore(cfg)
    sa = a.reset(123)
    sb = b.reset(123)
    for r in ROLES:
        assert sa[r].inventory == sb[r].inventory
        assert sa[r].ship_pipeline == sb[r].ship_pipeline
        assert sa[r].order_pipeline == sb[r].order_pipeline

    orders = {r: 4 for r in ROLES}
    for _ in range(5):
        sa, ra, ta, ia = a.step(orders)
        sb, rb, tb, ib = b.step(orders)
        assert ia.system_cost == ib.system_cost
        assert ra == rb
        for r in ROLES:
            assert sa[r].inventory == sb[r].inventory
            assert sa[r].backlog == sb[r].backlog


def test_order_clamp():
    env = BeerGameCore(classic_env_config(seed=0))
    env.reset(0)
    states, rewards, done, info = env.step({r: 999 for r in ROLES})
    assert all(info.orders_placed[r] == 64 for r in ROLES)
    assert all(info.orders_clamped[r] for r in ROLES)
    states, rewards, done, info = env.step({r: -3 for r in ROLES})
    assert all(info.orders_placed[r] == 0 for r in ROLES)


def test_no_negative_stocks_simple():
    env = BeerGameCore(classic_env_config(seed=1, horizon=36))
    env.reset(1)
    done = False
    while not done:
        orders = {r: 8 for r in ROLES}
        states, _, done, _ = env.step(orders)
        for r in ROLES:
            assert states[r].inventory >= 0
            assert states[r].backlog >= 0


def test_pipeline_lengths_stable():
    cfg = classic_env_config(seed=2)
    env = BeerGameCore(cfg)
    env.reset(2)
    for _ in range(10):
        states, _, _, _ = env.step({r: 5 for r in ROLES})
        for r in ROLES:
            assert len(states[r].ship_pipeline) == cfg.ship_delay
            assert len(states[r].order_pipeline) == cfg.order_delay


def test_delay_order_propagation():
    """Order placed by retailer at week t arrives at wholesaler at t+L_o."""
    cfg = classic_env_config(seed=0, order_delay=1, ship_delay=2, horizon=10)
    env = BeerGameCore(cfg)
    env.reset(0)
    # Week 1: retailer orders 17; others order 4
    orders = {Role.RETAILER: 17, Role.WHOLESALER: 4, Role.DISTRIBUTOR: 4, Role.FACTORY: 4}
    _, _, _, info1 = env.step(orders)
    assert info1.incoming_orders[Role.WHOLESALER] == 4  # still init pipeline
    # Week 2: wholesaler should see 17
    _, _, _, info2 = env.step({r: 4 for r in ROLES})
    assert info2.incoming_orders[Role.WHOLESALER] == 17


def test_delay_shipment_propagation():
    """Shipment from wholesaler enters retailer inventory after ship_delay weeks."""
    cfg = EnvConfig(
        horizon=10,
        ship_delay=2,
        order_delay=1,
        demand=ClassicStepDemand(pre=0, post=0, switch_week=99),
        init_inventory=(0, 100, 0, 0),
        init_pipeline_ship=0,
        init_pipeline_order=0,
        seed=0,
    )
    env = BeerGameCore(cfg)
    env.reset(0)
    # Force wholesaler to receive a huge order and ship
    # Week 1: put order 10 into wholesaler via retailer order; wholesaler has 100 inv
    # Actually retailer orders 10 → arrives wholesaler week 2.
    # Simpler: directly check that shipment appends and arrives after delay.
    # Week 1: wholesaler ships min(100, 0+0)=0 because incoming=0 from empty pipeline.
    # Give wholesaler incoming by seeding — use order_delay and prior order.

    # Reset with order pipeline so wholesaler sees demand 10 on week 1
    env._states[Role.WHOLESALER].order_pipeline = [10]
    env._states[Role.WHOLESALER].inventory = 100
    env._states[Role.RETAILER].inventory = 0
    env._states[Role.RETAILER].ship_pipeline = [0, 0]

    states, _, _, info = env.step({r: 0 for r in ROLES})
    assert info.shipments[Role.WHOLESALER] == 10
    # Immediately after step, shipment is in retailer's pipeline (not yet inventory)
    assert states[Role.RETAILER].inventory == 0
    assert states[Role.RETAILER].ship_pipeline[-1] == 10 or 10 in states[Role.RETAILER].ship_pipeline

    # After ship_delay steps of receiving, inventory should increase by 10
    # First pop may be 0, second pop is 10 if pipeline was [0, 10]
    got = 0
    for _ in range(cfg.ship_delay):
        states, _, _, info = env.step({r: 0 for r in ROLES})
        got += info.shipments_received[Role.RETAILER]
    assert got == 10


@given(
    seed=st.integers(0, 10_000),
    order_seq=st.lists(st.integers(0, 64), min_size=4, max_size=4),
)
@settings(max_examples=50, deadline=None)
def test_property_nonneg_and_costs(seed, order_seq):
    env = BeerGameCore(
        EnvConfig(
            horizon=20,
            demand=UniformDemand(0, 15),
            seed=seed,
        )
    )
    env.reset(seed)
    done = False
    steps = 0
    while not done and steps < 20:
        orders = {r: order_seq[int(r)] for r in ROLES}
        states, rewards, done, info = env.step(orders)
        steps += 1
        for r in ROLES:
            assert states[r].inventory >= 0
            assert states[r].backlog >= 0
            assert rewards[r] == pytest.approx(-info.local_costs[r])
        assert info.system_cost == pytest.approx(sum(info.local_costs.values()))
        # Inventory identity: shipped = min(inv+recv, backlog+incoming) at decision time
        # Reconstruct from info
        for r in ROLES:
            # After step, inventory and backlog are updated consistently:
            # inventory - backlog + shipped_this_week related — check backlog/inv exclusive
            assert not (states[r].inventory > 0 and states[r].backlog > 0)


@given(seed=st.integers(0, 5000))
@settings(max_examples=40, deadline=None)
def test_property_shipment_conservation(seed):
    """Units shipped by upstream equal units entering downstream ship pipeline."""
    env = BeerGameCore(classic_env_config(seed=seed, horizon=15, demand=UniformDemand(0, 12)))
    prev = env.reset(seed)
    done = False
    while not done:
        orders = {r: int(prev[r].last_demand_or_order) for r in ROLES}
        # Capture pipeline tails before step — actually check via info + post state
        before_pipelines = {r: list(env._states[r].ship_pipeline) for r in ROLES}
        states, _, done, info = env.step(orders)
        for upstream in (Role.WHOLESALER, Role.DISTRIBUTOR, Role.FACTORY):
            downstream = Role(int(upstream) - 1)
            shipped = info.shipments[upstream]
            # After step, last slot of downstream pipeline should be this shipment
            assert states[downstream].ship_pipeline[-1] == shipped
        prev = states


def test_golden_trajectory(tmp_path=None):
    """Fixed-seed 10-week snapshot for regression."""
    env = BeerGameCore(classic_env_config(seed=42, horizon=10))
    env.reset(42)
    traj = []
    done = False
    while not done:
        # Pass-through-ish: order 4 until step demand jumps awareness — use fixed 4
        orders = {r: 4 for r in ROLES}
        states, rewards, done, info = env.step(orders)
        traj.append(
            {
                "t": env.t,
                "inventories": {str(int(r)): states[r].inventory for r in ROLES},
                "backlogs": {str(int(r)): states[r].backlog for r in ROLES},
                "orders": {str(int(r)): info.orders_placed[r] for r in ROLES},
                "shipments": {str(int(r)): info.shipments[r] for r in ROLES},
                "system_cost": info.system_cost,
                "incoming": {str(int(r)): info.incoming_orders[r] for r in ROLES},
            }
        )

    golden_path = Path(__file__).parent / "golden_trajectory.json"
    if not golden_path.exists():
        golden_path.write_text(json.dumps(traj, indent=2, sort_keys=True))
        pytest.skip("Wrote golden trajectory; re-run to verify")
    expected = json.loads(golden_path.read_text())
    assert traj == expected
