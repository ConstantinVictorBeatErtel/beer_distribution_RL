"""Independent PPO (IPPO) — one policy and critic per role."""

from beer_distribution_rl.agents.ippo.trainer import IPPOConfig, IPPOTrainer, build_env_config, git_sha

__all__ = ["IPPOConfig", "IPPOTrainer", "build_env_config", "git_sha"]
