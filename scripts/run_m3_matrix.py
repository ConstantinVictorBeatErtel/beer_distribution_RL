#!/usr/bin/env python3
"""Run Tier-1 IPPO matrix cells for the M3 phase diagram.

Default headline matrix (laptop-scale):
  Regime B × capacity ∈ {inf, 1.5, 1.2, 1.0, 0.8}μ × {proportional, honesty_weighted}
  × seeds (default 10) with moderate timesteps.

Every cell logs YAML-equivalent config + seed + git SHA via IPPOTrainer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beer_distribution_rl.agents.ippo import IPPOConfig, IPPOTrainer  # noqa: E402
from beer_distribution_rl.agents.ippo.trainer import capacity_tag, default_run_name  # noqa: E402


def parse_caps(s: str) -> list[float | None]:
    out: list[float | None] = []
    for part in s.split(","):
        part = part.strip()
        if part in ("inf", "none", "None"):
            out.append(None)
        else:
            out.append(float(part))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--regimes", default="B", help="comma list A,B,C")
    p.add_argument("--caps", default="inf,1.5,1.2,1.0,0.8")
    p.add_argument("--rationing", default="proportional,honesty_weighted")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--total-timesteps", type=int, default=50_000)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=52)
    p.add_argument("--demand", default="ar1")
    p.add_argument("--out-dir", default="artifacts/runs/ippo/m3")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]
    caps = parse_caps(args.caps)
    rations = [x.strip() for x in args.rationing.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    cells = list(itertools.product(regimes, caps, rations, seeds))
    print(f"Sweep: {len(cells)} cells → {args.out_dir}")
    results_index = []

    for i, (regime, cap, rat, seed) in enumerate(cells):
        cfg = IPPOConfig(
            regime=regime,
            capacity_mult=cap,
            rationing=rat,
            seed=seed,
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            horizon=args.horizon,
            demand=args.demand,
            out_dir=args.out_dir,
            eval_every=10,
            eval_episodes=8,
            log_every=10,
        )
        name = default_run_name(cfg)
        out = Path(args.out_dir) / name
        print(f"\n=== [{i+1}/{len(cells)}] {name} ===")
        if args.skip_existing and (out / "final_eval.json").exists():
            print("skip existing")
            fe = json.loads((out / "final_eval.json").read_text())
            results_index.append({"run": name, **fe, "skipped": True})
            continue
        trainer = IPPOTrainer(cfg)
        trainer.train()
        fe = json.loads((out / "final_eval.json").read_text())
        results_index.append(
            {
                "run": name,
                "regime": regime,
                "capacity_mult": cap,
                "capacity_tag": capacity_tag(cfg),
                "rationing": rat,
                "seed": seed,
                **fe,
                "skipped": False,
            }
        )
        # checkpoint index after each cell
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / "index.json").write_text(json.dumps(results_index, indent=2))

    (Path(args.out_dir) / "index.json").write_text(json.dumps(results_index, indent=2))
    print(f"\nWrote {args.out_dir}/index.json ({len(results_index)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
