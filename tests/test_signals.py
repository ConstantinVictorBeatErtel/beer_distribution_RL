"""Cheap-talk signaling tests — honesty measured, never rewarded."""

from __future__ import annotations

import pytest

from beer_distribution_rl.env.core import BeerGameCore, Role, ROLES, classic_env_config
from beer_distribution_rl.env.signals import Signal, SignalChannel


def test_signal_delay_one_week():
    ch = SignalChannel(delay=1)
    ch.reset()
    # Week 0 buffer empty → receive all None
    board0 = ch.receive()
    assert all(v is None for v in board0[Role.RETAILER].values())
    sig = {r: None for r in ROLES}
    sig[Role.RETAILER] = Signal(claimed_demand=99, claimed_inventory=1)
    ch.send(sig)
    board1 = ch.receive()
    assert board1[Role.WHOLESALER][Role.RETAILER].claimed_demand == 99


def test_lies_accepted_and_measured():
    env = BeerGameCore(classic_env_config(regime="B", signaling_enabled=True, seed=0))
    env.reset(0)
    truth_inv = env.states[Role.RETAILER].inventory
    lie = Signal(claimed_demand=100, claimed_inventory=truth_inv + 50)
    signals = {r: None for r in ROLES}
    signals[Role.RETAILER] = lie
    _, rewards, _, info = env.step({r: 4 for r in ROLES}, signals=signals)
    h = info.honesty[Role.RETAILER]
    assert h["abs_demand_error"] == pytest.approx(100 - info.incoming_orders[Role.RETAILER])
    assert h["abs_inventory_error"] == pytest.approx(50)


def test_honesty_does_not_affect_reward():
    cfg = classic_env_config(regime="B", signaling_enabled=True, seed=7)
    env_truth = BeerGameCore(cfg)
    env_lie = BeerGameCore(cfg)
    env_truth.reset(7)
    env_lie.reset(7)
    orders = {r: 4 for r in ROLES}
    honest = {
        r: Signal(
            claimed_demand=env_truth.states[r].last_demand_or_order,
            claimed_inventory=env_truth.states[r].inventory,
        )
        for r in ROLES
    }
    # First step: last_demand is 0 at reset; just compare lie vs none on same orders
    _, r1, _, i1 = env_truth.step(orders, signals={r: None for r in ROLES})
    _, r2, _, i2 = env_lie.step(
        orders,
        signals={r: Signal(claimed_demand=999, claimed_inventory=999) for r in ROLES},
    )
    assert r1 == r2
    assert i1.local_costs == i2.local_costs
    assert i1.system_cost == i2.system_cost
    # Honesty present only when claims made
    assert any(
        (not (x != x))  # not NaN
        for x in [i2.honesty[Role.RETAILER]["mean_abs_error"]]
    )


def test_optional_broadcast():
    env = BeerGameCore(classic_env_config(regime="B", signaling_enabled=True, seed=0))
    env.reset(0)
    _, _, _, info = env.step({r: 4 for r in ROLES}, signals={r: None for r in ROLES})
    assert all(info.signals_sent[r] is None for r in ROLES)
