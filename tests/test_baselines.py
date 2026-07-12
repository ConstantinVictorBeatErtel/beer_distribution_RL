"""Baseline policy tests and bullwhip helpers."""

from __future__ import annotations

import statistics

from beer_distribution_rl.agents.baselines import (
    CLASSIC_BASE_STOCK_VECTOR,
    StermanAgent,
    StermanParams,
    base_stock_order,
)
from beer_distribution_rl.env.core import (
    BeerGameCore,
    Role,
    ROLES,
    classic_env_config,
    dqn_paper_env_config,
)


def _run_episode(env: BeerGameCore, policy_fn) -> tuple[list[float], dict[Role, list[int]], list[int]]:
    states = env.reset()
    system_costs: list[float] = []
    orders_hist: dict[Role, list[int]] = {r: [] for r in ROLES}
    demand_hist: list[int] = []
    done = False
    agents = policy_fn()
    while not done:
        orders = {r: agents[r](states[r]) for r in ROLES}
        states, _, done, info = env.step(orders)
        system_costs.append(info.system_cost)
        for r in ROLES:
            orders_hist[r].append(info.orders_placed[r])
        demand_hist.append(info.incoming_orders[Role.RETAILER])
    return system_costs, orders_hist, demand_hist


def bullwhip_ratios(orders_hist, demand_hist) -> dict[Role, float]:
    dvar = statistics.pvariance(demand_hist) if len(demand_hist) > 1 else 0.0
    out = {}
    for r in ROLES:
        ovar = statistics.pvariance(orders_hist[r]) if len(orders_hist[r]) > 1 else 0.0
        out[r] = ovar / dvar if dvar > 1e-12 else float("inf")
    return out


def test_dqn_base_stock_ballpark():
    costs_all = []
    for seed in range(15):
        env = BeerGameCore(
            dqn_paper_env_config(
                seed=seed,
                horizon=120,
                init_inventory=(0, 0, 0, 0),
                init_pipeline_ship=0,
                init_pipeline_order=0,
            )
        )

        def bs_policies():
            levels = CLASSIC_BASE_STOCK_VECTOR
            return {
                r: (lambda st, S=levels[int(r)]: base_stock_order(st, S)) for r in ROLES
            }

        costs, _, _ = _run_episode(env, bs_policies)
        # burn-in
        costs_all.append(sum(costs[40:]) / len(costs[40:]))
    mean = sum(costs_all) / len(costs_all)
    assert 4.0 < mean < 8.0, f"unexpected DQN-config base-stock mean: {mean}"


def test_sterman_bullwhip_amplification():
    env = BeerGameCore(classic_env_config(seed=0, horizon=36))

    def sterman_policies():
        agents = {r: StermanAgent(StermanParams(), expected_demand=4.0) for r in ROLES}

        def make(role):
            return lambda st, a=agents[role]: a.order(st)

        return {r: make(r) for r in ROLES}

    _, orders_hist, demand_hist = _run_episode(env, sterman_policies)
    bw = bullwhip_ratios(orders_hist, demand_hist)
    chain = [bw[Role.RETAILER], bw[Role.WHOLESALER], bw[Role.DISTRIBUTOR], bw[Role.FACTORY]]
    assert chain[-1] > chain[0]
    assert chain[-1] > 1.0 or chain[-2] > 1.0
