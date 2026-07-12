#!/usr/bin/env python3
"""Train independent PPO from a YAML experiment config.

Every run logs config + seed + git SHA under artifacts/runs/ippo/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beer_distribution_rl.agents.ippo import IPPOConfig, IPPOTrainer  # noqa: E402


def load_config(path: Path, seed: int | None, total_timesteps: int | None) -> IPPOConfig:
    raw = yaml.safe_load(path.read_text())
    if seed is not None:
        raw["seed"] = seed
    if total_timesteps is not None:
        raw["total_timesteps"] = total_timesteps
    known = {f.name for f in IPPOConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw.items() if k in known}
    return IPPOConfig(**filtered)


def main() -> int:
    p = argparse.ArgumentParser(description="Train IPPO (one policy per role)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config, args.seed, args.total_timesteps)
    if cfg.regime not in ("A", "B", "C"):
        print(f"ERROR: unknown regime {cfg.regime}")
        return 2
    print(f"Training IPPO regime={cfg.regime} seed={cfg.seed} steps={cfg.total_timesteps}")
    trainer = IPPOTrainer(cfg)
    out = trainer.train()
    print(f"Wrote run artifacts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
