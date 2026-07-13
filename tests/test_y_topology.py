"""Y-topology + DAG env tests (Agent E2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from beer_distribution_rl.env.core import (
    BeerGameCore,
    Role,
    ROLES,
    classic_env_config,
    y_topology_env_config,
)
from beer_distribution_rl.env.core_types import Y_ROLES
from beer_distribution_rl.env.demand import CorrelatedYDemand
from beer_distribution_rl.env.rationing import (
    HonestyWeightedRationing,
    ProportionalRationing,
    RationContext,
    UniformRationing,
)
from beer_distribution_rl.env.signals import Signal
from beer_distribution_rl.env.topology import get_topology, serial_topology, y_topology


class _FixedPairDemand:
    """Deterministic dual-retailer demand for allocation tests."""

    def __init__(self, a: int = 8, b: int = 8):
        self.a = a
        self.b = b

    def reset(self, rng) -> None:
        return None

    def demands(self, t: int, rng) -> dict[Role, int]:
        return {Role.RETAILER: self.a, Role.RETAILER_B: self.b}

    def __call__(self, t: int, rng) -> int:
        return self.a + self.b


def test_topology_factories():
    s = serial_topology()
    assert s.name == "serial"
    assert len(s.roles) == 4
    assert s.downstream[Role.WHOLESALER] == (Role.RETAILER,)
    y = y_topology()
    assert y.name == "y"
    assert len(y.roles) == 5
    assert set(y.downstream[Role.WHOLESALER]) == {Role.RETAILER, Role.RETAILER_B}
    assert get_topology("y_topology").name == "y"


def test_serial_golden_trajectory_unchanged():
    """DAG refactor must reduce to the old serial env on serial config."""
    env = BeerGameCore(classic_env_config(seed=42, horizon=10))
    env.reset(42)
    traj = []
    done = False
    while not done:
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
    expected = json.loads(golden_path.read_text())
    assert traj == expected


def test_y_wholesaler_allocations_conserve_supply():
    cfg = y_topology_env_config(
        horizon=6,
        demand=_FixedPairDemand(10, 10),
        rationing=ProportionalRationing(),
        init_inventory=(0, 0, 20, 0, 0),  # idx: R=0,W=1,D=2,F=3,Rb=4 — put stock at W
        init_pipeline_ship=0,
        init_pipeline_order=0,
        seed=0,
    )
    env = BeerGameCore(cfg)
    env.reset(0)
    # Seed wholesaler to see both retailers' orders this week.
    env._states[Role.WHOLESALER].inventory = 20
    env._states[Role.WHOLESALER].order_pipelines = {
        Role.RETAILER: [15],
        Role.RETAILER_B: [15],
    }
    env._states[Role.WHOLESALER].claimant_backlog = {
        Role.RETAILER: 0,
        Role.RETAILER_B: 0,
    }
    env._states[Role.WHOLESALER].order_pipeline = [30]
    orders = {r: 0 for r in Y_ROLES}
    _, _, _, info = env.step(orders)
    alloc = info.allocations[Role.WHOLESALER]
    assert sum(alloc.values()) == 20
    assert alloc[Role.RETAILER] == 10 and alloc[Role.RETAILER_B] == 10
    assert info.shipments[Role.WHOLESALER] == 20


def test_y_proportional_rewards_larger_order_in_env():
    cfg = y_topology_env_config(
        horizon=4,
        demand=_FixedPairDemand(0, 0),
        rationing=ProportionalRationing(),
        init_inventory=(0, 0, 0, 0, 0),
        init_pipeline_ship=0,
        init_pipeline_order=0,
        seed=0,
    )
    env = BeerGameCore(cfg)
    env.reset(0)
    env._states[Role.WHOLESALER].inventory = 20
    env._states[Role.WHOLESALER].order_pipelines = {
        Role.RETAILER: [30],
        Role.RETAILER_B: [10],
    }
    env._states[Role.WHOLESALER].claimant_backlog = {
        Role.RETAILER: 0,
        Role.RETAILER_B: 0,
    }
    _, _, _, info = env.step({r: 0 for r in Y_ROLES})
    alloc = info.allocations[Role.WHOLESALER]
    assert alloc[Role.RETAILER] > alloc[Role.RETAILER_B]
    assert sum(alloc.values()) == 20


def test_y_honesty_weighted_allocation_in_env():
    cfg = y_topology_env_config(
        horizon=4,
        demand=_FixedPairDemand(0, 0),
        rationing=HonestyWeightedRationing(),
        init_inventory=(0, 0, 0, 0, 0),
        init_pipeline_ship=0,
        init_pipeline_order=0,
        seed=0,
    )
    env = BeerGameCore(cfg)
    env.reset(0)
    env._honesty_ema[Role.RETAILER] = 0.0
    env._honesty_ema[Role.RETAILER_B] = -10.0
    env._states[Role.WHOLESALER].inventory = 20
    env._states[Role.WHOLESALER].order_pipelines = {
        Role.RETAILER: [20],
        Role.RETAILER_B: [20],
    }
    env._states[Role.WHOLESALER].claimant_backlog = {
        Role.RETAILER: 0,
        Role.RETAILER_B: 0,
    }
    _, rewards, _, info = env.step({r: 0 for r in Y_ROLES})
    alloc = info.allocations[Role.WHOLESALER]
    assert alloc[Role.RETAILER] > alloc[Role.RETAILER_B]
    # Reward still local — honesty did not enter the objective.
    for r in Y_ROLES:
        assert rewards[r] == pytest.approx(-info.local_costs[r])


def test_y_signals_broadcast_to_rival():
    cfg = y_topology_env_config(
        horizon=5,
        demand=CorrelatedYDemand(mu=5, phi=0.5, sigma_common=0.5, sigma_idio=0.5),
        signaling_enabled=True,
        regime="B",
        seed=1,
    )
    env = BeerGameCore(cfg)
    env.reset(1)
    lie = Signal(claimed_demand=99, claimed_inventory=1)
    signals = {r: None for r in Y_ROLES}
    signals[Role.RETAILER] = lie
    env.step({r: 4 for r in Y_ROLES}, signals=signals)
    # After 1-week delay, everyone — including rival retailer_b — hears the board.
    _, _, _, info = env.step({r: 4 for r in Y_ROLES}, signals={r: None for r in Y_ROLES})
    heard = info.signals_received[Role.RETAILER_B][Role.RETAILER]
    assert heard is not None
    assert heard.claimed_demand == 99
    # Log who hears what: every role is a listener of every sender.
    assert set(info.signal_listeners[Role.RETAILER]) == set(Y_ROLES)
    assert Role.RETAILER_B in info.signal_listeners[Role.RETAILER]


def test_correlated_y_demand_shared_factor():
    """Common factor induces positive cross-retailer correlation (rival signal informative)."""
    d = CorrelatedYDemand(mu=8.0, phi=0.9, sigma_common=3.0, sigma_idio=0.3, common0=0.0)
    rng = __import__("random").Random(0)
    d.reset(rng)
    xs, ys = [], []
    for t in range(1, 400):
        dem = d.demands(t, rng)
        xs.append(dem[Role.RETAILER])
        ys.append(dem[Role.RETAILER_B])
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / len(xs)
    vx = sum((a - mx) ** 2 for a in xs) / len(xs)
    vy = sum((b - my) ** 2 for b in ys) / len(ys)
    corr = cov / (vx * vy) ** 0.5
    assert corr > 0.5


def test_y_policies_diverge_under_shortage():
    """P3 precondition: on Y, proportional ≠ uniform ≠ honesty-weighted."""
    requested = {Role.RETAILER: 30, Role.RETAILER_B: 10}
    available = 20
    ctx = RationContext(honesty_ema={Role.RETAILER: -5.0, Role.RETAILER_B: 0.0})
    prop = ProportionalRationing().allocate(requested, available)
    uni = UniformRationing().allocate(requested, available)
    hon = HonestyWeightedRationing().allocate(requested, available, ctx)
    assert prop != uni
    assert hon != prop
    assert sum(prop.values()) == sum(uni.values()) == sum(hon.values()) == 20


@given(seed=st.integers(0, 3000))
@settings(max_examples=25, deadline=None)
def test_y_goods_conservation(seed):
    rng = __import__("random").Random(seed)
    env = BeerGameCore(
        y_topology_env_config(
            horizon=20,
            demand=CorrelatedYDemand(mu=6, phi=0.5, sigma_common=1.5, sigma_idio=1.0),
            seed=seed,
            init_pipeline_ship=2,
            init_pipeline_order=2,
        )
    )
    env.reset(seed)
    init_goods = sum(
        env._states[r].inventory + sum(env._states[r].ship_pipeline) for r in Y_ROLES
    )
    cum_prod = 0
    cum_delivered = 0
    done = False
    while not done:
        orders = {r: rng.randint(0, 20) for r in Y_ROLES}
        states, _, done, info = env.step(orders)
        cum_prod += info.factory_production
        cum_delivered += sum(info.shipments[c] for c in (Role.RETAILER, Role.RETAILER_B))
        physical = sum(states[r].inventory + sum(states[r].ship_pipeline) for r in Y_ROLES)
        assert init_goods + cum_prod == physical + cum_delivered


def test_y_local_rewards_regime_a():
    env = BeerGameCore(y_topology_env_config(regime="A", seed=0, horizon=3))
    env.reset(0)
    _, rewards, _, info = env.step({r: 5 for r in Y_ROLES})
    for r in Y_ROLES:
        assert rewards[r] == pytest.approx(-info.local_costs[r])
