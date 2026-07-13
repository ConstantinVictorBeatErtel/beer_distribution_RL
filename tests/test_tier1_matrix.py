"""Tier-1 matrix runner + vec-env scaffolding tests (Agent R1)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from beer_distribution_rl.agents.ippo import IPPOConfig, IPPOTrainer, enumerate_cells, prune_summary
from beer_distribution_rl.agents.ippo.matrix import should_keep_cell
from beer_distribution_rl.agents.ippo.vec_env import SyncBeerGameVecEnv
from beer_distribution_rl.env.core import BeerGameCore, classic_env_config


def test_prune_drops_rationing_at_infinity_and_serial_variants():
    assert should_keep_cell("B", "serial", None, "proportional", "ar1")[0]
    assert not should_keep_cell("B", "serial", None, "uniform", "ar1")[0]
    assert not should_keep_cell("B", "serial", 1.0, "honesty_weighted", "ar1")[0]
    assert should_keep_cell("B", "y", 1.0, "honesty_weighted", "ar1")[0]
    assert not should_keep_cell("A", "y", None, "uniform", "ar1")[0]


def test_prune_summary_counts():
    s = prune_summary(seeds=tuple(range(10)))
    assert s["full_cartesian"] == 1440
    assert s["kept"] == 840
    assert s["pruned"] == 600
    assert "rationing_at_infinite_capacity" in s["prune_reasons"]
    assert "serial_single_claimant_equivalence" in s["prune_reasons"]


def test_enumerate_includes_regime_a_capacity_sweep():
    cells = enumerate_cells(seeds=(0,))
    a_caps = {
        c.capacity_mult
        for c in cells
        if c.regime == "A" and c.topology == "serial" and c.demand == "ar1"
    }
    assert None in a_caps
    assert 1.2 in a_caps and 1.0 in a_caps and 0.8 in a_caps
    # D1 requires matched A and B
    assert any(c.regime == "B" and c.capacity_mult == 1.0 for c in cells)


def test_vec_env_steps_n():
    cfg = classic_env_config(horizon=5, seed=0)
    vec = SyncBeerGameVecEnv(cfg, n_envs=4)
    states = vec.reset(0)
    assert len(states) == 4
    orders = [{r: 4 for r in vec.roles} for _ in range(4)]
    states, rewards, dones, infos = vec.step(orders)
    assert len(infos) == 4
    assert all(hasattr(info, "capacity_binds") for info in infos)


def test_stepinfo_bind_fields():
    env = BeerGameCore(classic_env_config(horizon=3, seed=0, capacity=1.0))
    env.reset(0)
    _, _, _, info = env.step({r: 8 for r in env.roles})
    assert hasattr(info, "capacity_binds")
    assert hasattr(info, "allocation_triggers")
    assert isinstance(info.capacity_binds, bool)
    assert isinstance(info.allocation_triggers, bool)


def test_trainer_logs_bind_events_and_config_yaml(tmp_path):
    cfg = IPPOConfig(
        regime="A",
        topology="serial",
        demand="ar1",
        capacity_mult=1.0,
        total_timesteps=512,
        rollout_steps=64,
        n_envs=4,
        update_epochs=1,
        minibatch_size=64,
        eval_every=1,
        eval_episodes=2,
        log_every=1,
        seed=0,
        horizon=20,
        out_dir=str(tmp_path / "ippo"),
    )
    trainer = IPPOTrainer(cfg)
    out = trainer.train()
    assert (out / "config.yaml").exists()
    assert (out / "run_meta.json").exists()
    assert (out / "week_events.json").exists()
    fe = json.loads((out / "final_eval.json").read_text())
    assert "eval/frac_capacity_binds" in fe
    assert "eval/frac_allocation_triggers" in fe
    meta = json.loads((out / "run_meta.json").read_text())
    assert "git_sha" in meta
    assert meta["config"]["seed"] == 0


def test_y_topology_ippo_smoke(tmp_path):
    cfg = IPPOConfig(
        regime="B",
        topology="y",
        demand="ar1",
        capacity_mult=1.0,
        rationing="proportional",
        total_timesteps=512,
        rollout_steps=64,
        n_envs=2,
        update_epochs=1,
        minibatch_size=64,
        eval_every=1,
        eval_episodes=1,
        log_every=1,
        seed=1,
        horizon=16,
        out_dir=str(tmp_path / "ippo"),
    )
    trainer = IPPOTrainer(cfg)
    assert len(trainer.roles) == 5
    out = trainer.train()
    assert (out / "checkpoints" / "policy_retailer_a.pt").exists()
    assert (out / "checkpoints" / "policy_retailer_b.pt").exists()
