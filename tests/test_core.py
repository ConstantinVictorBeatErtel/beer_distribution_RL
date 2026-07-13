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
from beer_distribution_rl.env.demand import ClassicStepDemand, DEFAULT_ORDER_CAP, UniformDemand


def _pass_through_orders(states, cap=DEFAULT_ORDER_CAP):
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
    assert all(info.orders_placed[r] == DEFAULT_ORDER_CAP for r in ROLES)
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
    env._states[Role.WHOLESALER].order_pipelines = {Role.RETAILER: [10]}
    env._states[Role.WHOLESALER].order_pipeline = [10]
    env._states[Role.WHOLESALER].claimant_backlog = {Role.RETAILER: 0}
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


def _expected_on_order(env: BeerGameCore, role: Role) -> int:
    """Outstanding replenishment: ship pipeline + upstream unfilled (serial)."""
    st = env._states[role]
    ship = sum(st.ship_pipeline)
    if role == Role.FACTORY:
        return ship
    up_role = env.topology.upstream[role]
    assert up_role is not None
    up = env._states[up_role]
    if role in up.order_pipelines:
        in_transit = sum(up.order_pipelines[role])
    else:
        in_transit = sum(up.order_pipeline)
    owed = int(up.claimant_backlog.get(role, 0)) if up.claimant_backlog else up.backlog
    return ship + in_transit + owed


def test_on_order_init_includes_order_delay_pipeline():
    """B1 regression: on_order must count orders still in upstream order_pipeline."""
    cfg = classic_env_config(
        seed=0,
        init_inventory=(12, 12, 12, 12),
        init_pipeline_ship=4,
        init_pipeline_order=4,
        ship_delay=2,
        order_delay=1,
    )
    env = BeerGameCore(cfg)
    states = env.reset(0)
    # Ship pipe sum = 8; non-factory also has 4 in upstream order pipe.
    assert states[Role.FACTORY].on_order == 8
    for role in (Role.RETAILER, Role.WHOLESALER, Role.DISTRIBUTOR):
        assert states[role].on_order == 12, f"{role}: {states[role].on_order}"
    # Retailer does not consume order_pipeline for incoming demand.
    assert states[Role.RETAILER].order_pipeline == [0]


def test_delay_unit_trace_classic_beer_game():
    """B1: single-unit order at week 1 arrives with L_o=1, L_s=2 (receipt week 4)."""
    cfg = EnvConfig(
        horizon=8,
        ship_delay=2,
        order_delay=1,
        demand=ClassicStepDemand(pre=0, post=0, switch_week=99),
        init_inventory=(0, 100, 0, 0),
        init_pipeline_ship=0,
        init_pipeline_order=0,
        capacity=None,
        seed=0,
    )
    env = BeerGameCore(cfg)
    env.reset(0)
    for r in ROLES:
        env._states[r].ship_pipeline = [0] * cfg.ship_delay
        env._states[r].order_pipeline = [0] * cfg.order_delay
        env._states[r].on_order = 0
        env._states[r].inventory = 100 if r == Role.WHOLESALER else 0
        env._states[r].backlog = 0
        if env.topology.downstream[r]:
            env._states[r].order_pipelines = {
                c: [0] * cfg.order_delay for c in env.topology.downstream[r]
            }
            env._states[r].claimant_backlog = {c: 0 for c in env.topology.downstream[r]}
        else:
            env._states[r].order_pipelines = {}
            env._states[r].claimant_backlog = {}

    wholesaler_sees = wholesaler_ships = retailer_recv = None
    for week in range(1, 7):
        orders = {
            Role.RETAILER: 1 if week == 1 else 0,
            Role.WHOLESALER: 0,
            Role.DISTRIBUTOR: 0,
            Role.FACTORY: 0,
        }
        _, _, _, info = env.step(orders)
        if info.incoming_orders[Role.WHOLESALER] == 1 and wholesaler_sees is None:
            wholesaler_sees = week
        if info.shipments[Role.WHOLESALER] == 1 and wholesaler_ships is None:
            wholesaler_ships = week
        if info.shipments_received[Role.RETAILER] == 1 and retailer_recv is None:
            retailer_recv = week

    assert wholesaler_sees == 2  # L_o = 1
    assert wholesaler_ships == 2
    assert retailer_recv == 4  # ship week 2 + L_s = 2


def test_factory_production_delay():
    """Factory production enters own ship pipeline and arrives after L_s weeks."""
    cfg = EnvConfig(
        horizon=6,
        ship_delay=2,
        order_delay=1,
        demand=ClassicStepDemand(pre=0, post=0, switch_week=99),
        init_inventory=(0, 0, 0, 0),
        init_pipeline_ship=0,
        init_pipeline_order=0,
        seed=0,
    )
    env = BeerGameCore(cfg)
    env.reset(0)
    for r in ROLES:
        env._states[r].ship_pipeline = [0] * cfg.ship_delay
        env._states[r].order_pipeline = [0] * cfg.order_delay
        env._states[r].on_order = 0
        env._states[r].inventory = 0
        env._states[r].backlog = 0
        if env.topology.downstream[r]:
            env._states[r].order_pipelines = {
                c: [0] * cfg.order_delay for c in env.topology.downstream[r]
            }
            env._states[r].claimant_backlog = {c: 0 for c in env.topology.downstream[r]}
        else:
            env._states[r].order_pipelines = {}
            env._states[r].claimant_backlog = {}

    recv_week = None
    for week in range(1, 6):
        orders = {r: (1 if (r == Role.FACTORY and week == 1) else 0) for r in ROLES}
        _, _, _, info = env.step(orders)
        if info.shipments_received[Role.FACTORY] == 1:
            recv_week = week
            break
    assert recv_week == 3


@given(seed=st.integers(0, 8000), scale=st.integers(0, 40))
@settings(max_examples=60, deadline=None)
def test_property_on_order_invariant(seed, scale):
    """on_order equals ship_pipeline + upstream outstanding for every week."""
    rng = __import__("random").Random(seed)
    env = BeerGameCore(
        EnvConfig(horizon=20, demand=UniformDemand(0, 15), capacity=None, seed=seed)
    )
    env.reset(seed)
    for r in ROLES:
        assert env._states[r].on_order == _expected_on_order(env, r)
    done = False
    while not done:
        orders = {r: rng.randint(0, scale) for r in ROLES}
        _, _, done, _ = env.step(orders)
        for r in ROLES:
            assert env._states[r].on_order == _expected_on_order(env, r), (
                f"t={env.t} role={r} got={env._states[r].on_order} "
                f"exp={_expected_on_order(env, r)}"
            )


@given(seed=st.integers(0, 8000))
@settings(max_examples=40, deadline=None)
def test_property_goods_conservation(seed):
    """init + cumulative production = physical stock + delivered to customers."""
    rng = __import__("random").Random(seed)
    env = BeerGameCore(
        EnvConfig(horizon=25, demand=UniformDemand(0, 15), capacity=None, seed=seed)
    )
    env.reset(seed)
    init_goods = sum(
        env._states[r].inventory + sum(env._states[r].ship_pipeline) for r in ROLES
    )
    cum_prod = 0
    cum_delivered = 0
    done = False
    while not done:
        orders = {r: rng.randint(0, 64) for r in ROLES}
        states, _, done, info = env.step(orders)
        cum_prod += info.factory_production
        cum_delivered += info.shipments[Role.RETAILER]
        physical = sum(states[r].inventory + sum(states[r].ship_pipeline) for r in ROLES)
        assert init_goods + cum_prod == physical + cum_delivered
