"""Independent PPO (IPPO) — one policy and critic per role."""

from beer_distribution_rl.agents.ippo.trainer import (
    IPPOConfig,
    IPPOTrainer,
    build_env_config,
    capacity_tag,
    default_run_name,
    git_sha,
)
from beer_distribution_rl.agents.ippo.matrix import (
    MatrixCell,
    enumerate_cells,
    prune_summary,
)

__all__ = [
    "IPPOConfig",
    "IPPOTrainer",
    "MatrixCell",
    "build_env_config",
    "capacity_tag",
    "default_run_name",
    "enumerate_cells",
    "git_sha",
    "prune_summary",
]
