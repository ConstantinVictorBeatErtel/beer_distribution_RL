"""Performance microbench: core step throughput."""

from __future__ import annotations

import time

import pytest

from beer_distribution_rl.env.core import BeerGameCore, ROLES, classic_env_config


def test_core_steps_per_sec():
    env = BeerGameCore(classic_env_config(horizon=10_000, seed=0))
    env.reset(0)
    orders = {r: 4 for r in ROLES}
    n = 20_000
    # Warmup
    for _ in range(100):
        if env.t >= env.config.horizon:
            env.reset(0)
        env.step(orders)
    env.reset(0)
    t0 = time.perf_counter()
    for _ in range(n):
        if env.t >= env.config.horizon:
            env.reset(0)
        env.step(orders)
    elapsed = time.perf_counter() - t0
    rate = n / elapsed
    assert rate > 10_000, f"only {rate:.0f} steps/sec (need >10k)"
