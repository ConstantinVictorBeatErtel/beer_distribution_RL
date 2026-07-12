"""IPPO independence and smoke tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from beer_distribution_rl.agents.ippo import IPPOConfig, IPPOTrainer
from beer_distribution_rl.env.core import ROLES


def test_no_parameter_sharing():
    trainer = IPPOTrainer(IPPOConfig(total_timesteps=100, rollout_steps=64, seed=0))
    # Distinct parameter object identities
    params = []
    for r in ROLES:
        params.extend([id(p) for p in trainer.policies[r].parameters()])
    assert len(params) == len(set(params))
    # Distinct critic modules
    assert len({id(trainer.policies[r].critic_head) for r in ROLES}) == 4
    assert len({id(trainer.policies[r].actor_head) for r in ROLES}) == 4


def test_regime_b_rejected():
    with pytest.raises(ValueError, match="A and C"):
        IPPOTrainer(IPPOConfig(regime="B", total_timesteps=100, rollout_steps=32))


def test_ippo_smoke_learns_finite_cost(tmp_path):
    cfg = IPPOConfig(
        regime="A",
        total_timesteps=2048,
        rollout_steps=512,
        update_epochs=2,
        minibatch_size=128,
        eval_every=1,
        eval_episodes=3,
        log_every=1,
        seed=1,
        out_dir=str(tmp_path / "ippo"),
        horizon=36,
    )
    trainer = IPPOTrainer(cfg)
    out = trainer.train()
    assert (out / "checkpoints" / "policy_retailer.pt").exists()
    assert (out / "run_meta.json").exists()
    final = trainer.evaluate(n_episodes=5)
    assert final["eval/mean_system_cost"] < 5000  # not exploded


def test_regime_c_uses_shared_reward_signal():
    """Regime C: all roles receive identical reward each step (system cost)."""
    from beer_distribution_rl.env.core import BeerGameCore, classic_env_config

    env = BeerGameCore(classic_env_config(regime="C", horizon=5, seed=0))
    env.reset(0)
    _, rewards, _, info = env.step({r: 4 for r in ROLES})
    assert len(set(rewards.values())) == 1
    assert list(rewards.values())[0] == pytest.approx(-info.system_cost)
