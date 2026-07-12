#!/usr/bin/env python3
"""Evaluate trained IPPO checkpoints vs Sterman / base-stock baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beer_distribution_rl.agents.baselines import (  # noqa: E402
    CLASSIC_BASE_STOCK_VECTOR,
    StermanAgent,
    StermanParams,
    base_stock_order,
)
from beer_distribution_rl.agents.ippo import IPPOConfig, IPPOTrainer  # noqa: E402
from beer_distribution_rl.env.core import ROLES, Role, classic_env_config  # noqa: E402
from beer_distribution_rl.env.demand import ClassicStepDemand  # noqa: E402


def eval_baseline(name: str, n_episodes: int, seed: int, horizon: int = 52) -> dict:
    costs = []
    for ep in range(n_episodes):
        env = __import__("beer_distribution_rl.env.core", fromlist=["BeerGameCore"]).BeerGameCore(
            classic_env_config(horizon=horizon, demand=ClassicStepDemand(), seed=seed + ep)
        )
        states = env.reset(seed + ep)
        if name == "base_stock":
            agents = None
        elif name == "sterman":
            agents = {r: StermanAgent(StermanParams(), expected_demand=4.0) for r in ROLES}
        else:
            raise ValueError(name)
        done = False
        acc = 0.0
        steps = 0
        while not done:
            if name == "base_stock":
                # Grid-searched-ish levels for classic step (higher than DQN U{0,1,2} levels)
                levels = (20, 20, 20, 20)
                orders = {r: base_stock_order(states[r], levels[int(r)]) for r in ROLES}
            else:
                orders = {r: agents[r].order(states[r]) for r in ROLES}
            states, _, done, info = env.step(orders)
            acc += info.system_cost
            steps += 1
        costs.append(acc / steps)
    return {"mean": float(np.mean(costs)), "std": float(np.std(costs))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True, help="artifacts/runs/ippo/regimeA_seed0")
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()

    meta = json.loads((args.run_dir / "run_meta.json").read_text())
    cfg = IPPOConfig(**{k: v for k, v in meta["config"].items() if k in IPPOConfig.__dataclass_fields__})
    trainer = IPPOTrainer(cfg)
    trainer.load(args.run_dir / "checkpoints")
    ippo = trainer.evaluate(n_episodes=args.episodes)

    bs = eval_baseline("base_stock", args.episodes, cfg.seed + 100)
    st = eval_baseline("sterman", args.episodes, cfg.seed + 100)

    report = {
        "run_dir": str(args.run_dir),
        "git_sha": meta.get("git_sha"),
        "regime": cfg.regime,
        "ippo": ippo,
        "base_stock": bs,
        "sterman": st,
        "beats_sterman": ippo["eval/mean_system_cost"] < st["mean"],
    }
    out = args.run_dir / "comparison.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
