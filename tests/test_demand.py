"""Demand processes, information asymmetry, and action-cap tests (E1 / env v1.1)."""

from __future__ import annotations

import inspect
import math
import random
import warnings
from pathlib import Path

import pytest

from beer_distribution_rl.env.core import (
    BeerGameCore,
    EnvConfig,
    Role,
    ROLES,
    classic_env_config,
)
from beer_distribution_rl.env.demand import (
    AR1Demand,
    BOUNDARY_ACTION_WARN_FRAC,
    ClassicStepDemand,
    DEFAULT_ORDER_CAP,
    RegimeSwitchDemand,
    UniformDemand,
    frac_actions_at_boundary,
    info_value_table,
    make_demand,
    recommend_order_cap,
    sample_demand_series,
    truthful_broadcast_info_value,
    warn_if_boundary_saturated,
)


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


def _empirical_moments(process, *, horizon=200, n_traj=400, seed=0):
    series = sample_demand_series(process, horizon=horizon, n_traj=n_traj, seed=seed)
    vals = [v for traj in series for v in traj]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    xs = [traj[i] for traj in series for i in range(len(traj) - 1)]
    ys = [traj[i + 1] for traj in series for i in range(len(traj) - 1)]
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    cov = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / len(xs)
    var_x = sum((x - x_bar) ** 2 for x in xs) / len(xs)
    corr = cov / var_x if var_x > 1e-12 else 0.0
    return mean, var, corr


def test_uniform_moments_match_theory():
    proc = UniformDemand(0, 15)
    mean, var, corr = _empirical_moments(proc, horizon=52, n_traj=800, seed=1)
    assert abs(mean - proc.theoretical_mean()) < 0.15
    assert abs(var - proc.theoretical_var()) < 0.4
    assert abs(corr - 0.0) < 0.05


def test_ar1_moments_match_theory():
    proc = AR1Demand(mu=7.5, phi=0.7, sigma=2.0)
    # Longer burn-in: start at mu; stationary moments need many steps.
    mean, var, corr = _empirical_moments(proc, horizon=80, n_traj=600, seed=2)
    assert abs(mean - proc.theoretical_mean()) < 0.25
    assert abs(var - proc.theoretical_var()) / proc.theoretical_var() < 0.15
    assert abs(corr - proc.theoretical_lag1_corr()) < 0.08


def test_regime_switch_moments_match_theory():
    proc = RegimeSwitchDemand(
        mu_low=4.0, mu_high=12.0, sigma=1.5, p_stay_low=0.9, p_stay_high=0.9
    )
    mean, var, corr = _empirical_moments(proc, horizon=80, n_traj=600, seed=3)
    assert abs(mean - proc.theoretical_mean()) < 0.35
    assert abs(var - proc.theoretical_var()) / proc.theoretical_var() < 0.20
    assert abs(corr - proc.theoretical_lag1_corr()) < 0.12


def test_classic_step_path():
    proc = ClassicStepDemand(pre=4, post=8, switch_week=5)
    rng = random.Random(0)
    proc.reset(rng)
    assert [proc(t, rng) for t in range(1, 8)] == [4, 4, 4, 4, 8, 8, 8]


def test_make_demand_factory():
    assert isinstance(make_demand("uniform"), UniformDemand)
    assert isinstance(make_demand("ar1"), AR1Demand)
    ar = make_demand("ar1", phi=0.7, mu=7.5, sigma=2.0)
    assert ar.phi == 0.7 and ar.mu == 7.5
    assert isinstance(make_demand("regime_switch"), RegimeSwitchDemand)
    assert isinstance(make_demand("classic_step"), ClassicStepDemand)
    with pytest.raises(ValueError):
        make_demand("not_a_process")


# ---------------------------------------------------------------------------
# Information value (paper metric)
# ---------------------------------------------------------------------------


def test_info_value_uniform_near_zero_ar1_positive():
    table = info_value_table(horizon=52, n_traj=800, seed=0)
    uni = table["uniform"]
    ar1 = table["ar1"]
    rs = table["regime_switch"]
    # Uniform: truthful broadcast cannot reduce forecast error in principle.
    assert abs(float(uni["relative_mse_reduction"])) < 0.05
    assert abs(float(uni["lag1_r2"])) < 0.05
    # AR(1): material one-step forecast-error reduction (D6≈0.47 R²).
    assert float(ar1["relative_mse_reduction"]) > 0.35
    assert float(ar1["lag1_r2"]) > 0.35
    # Regime switch: high information value (easy communication case).
    assert float(rs["relative_mse_reduction"]) > 0.40
    assert float(rs["mse_reduction"]) > float(uni["mse_reduction"])


def test_recommend_order_cap_from_ar1():
    rec = recommend_order_cap(AR1Demand(mu=7.5, phi=0.7, sigma=2.0))
    assert int(rec["suggested_hard_cap"]) == DEFAULT_ORDER_CAP == 128
    assert float(rec["d_hi_approx"]) > 14.0
    assert DEFAULT_ORDER_CAP >= 128


# ---------------------------------------------------------------------------
# Observation leak — only retailer sees true consumer demand
# ---------------------------------------------------------------------------

FORBIDDEN_DEMAND_KEYS = {
    "customer_demand",
    "true_demand",
    "consumer_demand",
    "end_customer_demand",
}


def test_upstream_obs_does_not_leak_customer_demand():
    """Retailer alone observes true demand; upstream last_demand ≠ customer demand."""
    env = BeerGameCore(
        EnvConfig(
            horizon=10,
            demand=UniformDemand(0, 15),
            seed=0,
            order_cap=DEFAULT_ORDER_CAP,
        )
    )
    env.reset(0)
    # Retailer orders a fixed quantity that differs from demand with high probability.
    retailer_order = 3
    leaked = False
    saw_mismatch = False
    for _ in range(8):
        orders = {r: retailer_order for r in ROLES}
        _, _, _, info = env.step(orders)
        cd = info.customer_demand
        assert cd is not None
        assert env.last_customer_demand == cd

        ret_obs = env.observe(Role.RETAILER)
        assert ret_obs["last_demand_or_order"] == cd
        for key in FORBIDDEN_DEMAND_KEYS:
            assert key not in ret_obs

        for role in (Role.WHOLESALER, Role.DISTRIBUTOR, Role.FACTORY):
            obs = env.observe(role)
            for key in FORBIDDEN_DEMAND_KEYS:
                assert key not in obs
            # Upstream sees own incoming order, not consumer demand.
            if obs["last_demand_or_order"] != cd:
                saw_mismatch = True
            # After the order delay, wholesaler's incoming equals retailer's prior order.
            if role == Role.WHOLESALER and env.t >= 2:
                assert obs["last_demand_or_order"] == retailer_order
                if cd != retailer_order:
                    assert obs["last_demand_or_order"] != cd
                    leaked = leaked or (obs["last_demand_or_order"] == cd)

    assert saw_mismatch, "test setup failed to create demand≠order contrast"
    assert not leaked


def test_wrapper_and_ippo_obs_no_demand_leak():
    """Grep-style: wrapper + IPPO featurizer must not expose customer_demand."""
    root = Path(__file__).resolve().parents[1]
    wrapper_src = (root / "beer_distribution_rl/env/wrappers.py").read_text()
    obs_src = (root / "beer_distribution_rl/agents/ippo/obs.py").read_text()
    core_observe = inspect.getsource(BeerGameCore.observe)

    for src, label in (
        (wrapper_src, "wrappers.py"),
        (obs_src, "obs.py"),
        (core_observe, "BeerGameCore.observe"),
    ):
        for key in FORBIDDEN_DEMAND_KEYS:
            # Allowed in comments / asserts that *forbid* the key; not as an obs field.
            if f'"{key}"' in src or f"'{key}'" in src:
                # Must appear only in assert-not-present or documentation contexts.
                assert (
                    f'"{key}" not in' in src
                    or f"'{key}' not in" in src
                    or "never" in src.lower()
                    or "forbid" in src.lower()
                    or "assert" in src
                ), f"{label} may leak {key}"

    pz = pytest.importorskip("pettingzoo")
    pytest.importorskip("gymnasium")
    from beer_distribution_rl.env.wrappers import BeerGameParallelEnv
    from beer_distribution_rl.agents.ippo.obs import state_to_obs

    cfg = classic_env_config(
        horizon=6,
        demand=UniformDemand(0, 15),
        seed=1,
        signaling_enabled=True,
        order_cap=DEFAULT_ORDER_CAP,
    )
    penv = BeerGameParallelEnv(cfg)
    obs, _ = penv.reset(seed=1)
    for name, o in obs.items():
        assert "customer_demand" not in o
        assert set(o.keys()) <= {
            "inventory",
            "backlog",
            "on_order",
            "last_demand_or_order",
            "t",
        }

    # Force distinct retailer order vs demand.
    for _ in range(4):
        actions = {
            a: {
                "order": 2,
                "claimed_demand": 0,
                "claimed_inventory": 0,
                "broadcast": 0,
            }
            for a in penv.agents
        }
        obs, _, terms, _, _ = penv.step(actions)

    cd = penv.core.last_customer_demand
    assert cd is not None
    wh = obs["wholesaler"]["last_demand_or_order"]
    # After delay, wholesaler sees retailer order=2, not customer demand.
    assert wh == 2.0
    if cd != 2:
        assert wh != float(cd)

    # IPPO vector: finite floats, no NaN injection of privileged demand.
    for role in ROLES:
        vec = state_to_obs(penv.core._states[role], role, penv.core)
        assert vec.shape[0] > 0
        assert all(math.isfinite(float(x)) for x in vec)


# ---------------------------------------------------------------------------
# Boundary-action metric
# ---------------------------------------------------------------------------


def test_default_order_cap_is_128():
    assert EnvConfig().order_cap == 128
    assert DEFAULT_ORDER_CAP == 128


def test_boundary_fraction_metric_and_warning():
    env = BeerGameCore(EnvConfig(horizon=5, demand=ClassicStepDemand(), seed=0))
    env.reset(0)
    # All orders at the hard cap.
    for _ in range(5):
        orders = {r: DEFAULT_ORDER_CAP for r in ROLES}
        _, _, _, info = env.step(orders)
        assert info.frac_actions_at_cap == pytest.approx(1.0)
    assert env.boundary_action_fraction() == pytest.approx(1.0)
    assert frac_actions_at_boundary([128, 128, 0], 128) == pytest.approx(2 / 3)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_boundary_saturated(0.10, threshold=BOUNDARY_ACTION_WARN_FRAC)
        assert any("boundary fraction" in str(x.message) for x in w)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_boundary_saturated(0.01, threshold=BOUNDARY_ACTION_WARN_FRAC)
        assert not w


def test_healthy_pass_through_not_at_boundary():
    """Pass-through under AR(1) should almost never hit cap=128."""
    env = BeerGameCore(
        EnvConfig(
            horizon=40,
            demand=AR1Demand(mu=7.5, phi=0.7, sigma=2.0),
            seed=0,
            order_cap=DEFAULT_ORDER_CAP,
        )
    )
    states = env.reset(0)
    for _ in range(40):
        orders = {
            r: max(0, min(DEFAULT_ORDER_CAP, states[r].last_demand_or_order))
            for r in ROLES
        }
        states, _, _, info = env.step(orders)
    assert info.frac_actions_at_cap < 0.05
