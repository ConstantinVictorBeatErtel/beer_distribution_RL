#!/usr/bin/env python3
"""Parallel Tier-1 matrix runner (Agent R1).

Dominant speedup: multiprocess across cells (8–16 workers). Each worker runs
one IPPO cell with vectorized envs (n_envs≥64). Resumable via --skip-existing
(checkpoint = final_eval.json present).

Example (laptop):
  python scripts/run_tier1_matrix.py --workers 8 --skip-existing

Example (Colab GPU, fewer workers, larger n_envs):
  python scripts/run_tier1_matrix.py --workers 2 --n-envs 128 --device cuda --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beer_distribution_rl.agents.ippo import git_sha  # noqa: E402
from beer_distribution_rl.agents.ippo.matrix import (  # noqa: E402
    enumerate_cells,
    prune_summary,
)
from beer_distribution_rl.agents.ippo.trainer import capacity_tag  # noqa: E402


def _cell_done(out_dir: Path, run_name: str) -> bool:
    return (out_dir / run_name / "final_eval.json").exists()


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry — must be top-level for pickling."""
    # Limit BLAS/OMP threads so 8–16 workers don't oversubscribe.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    from beer_distribution_rl.agents.ippo import IPPOConfig, IPPOTrainer
    from beer_distribution_rl.agents.ippo.trainer import capacity_tag, default_run_name, git_sha

    cfg = IPPOConfig(**payload["config"])
    name = cfg.run_name or default_run_name(cfg)
    out = Path(cfg.out_dir) / name
    t0 = time.time()
    try:
        if payload.get("skip_existing") and (out / "final_eval.json").exists():
            fe = json.loads((out / "final_eval.json").read_text())
            return {
                "run": name,
                "status": "skipped",
                "elapsed_s": 0.0,
                "regime": cfg.regime,
                "topology": cfg.topology,
                "capacity_mult": cfg.capacity_mult,
                "capacity_tag": capacity_tag(cfg),
                "rationing": cfg.rationing,
                "demand": cfg.demand,
                "seed": cfg.seed,
                "git_sha": git_sha(),
                **{k: v for k, v in fe.items() if isinstance(v, (int, float, str, bool))},
            }
        trainer = IPPOTrainer(cfg)
        trainer.train()
        fe = json.loads((out / "final_eval.json").read_text())
        return {
            "run": name,
            "status": "ok",
            "elapsed_s": time.time() - t0,
            "regime": cfg.regime,
            "topology": cfg.topology,
            "capacity_mult": cfg.capacity_mult,
            "capacity_tag": capacity_tag(cfg),
            "rationing": cfg.rationing,
            "demand": cfg.demand,
            "seed": cfg.seed,
            "git_sha": git_sha(),
            **{k: v for k, v in fe.items() if isinstance(v, (int, float, str, bool))},
        }
    except Exception as exc:  # noqa: BLE001 — surface per-cell failure without killing pool
        return {
            "run": name,
            "status": "error",
            "elapsed_s": time.time() - t0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "regime": cfg.regime,
            "topology": cfg.topology,
            "capacity_mult": cfg.capacity_mult,
            "capacity_tag": capacity_tag(cfg),
            "rationing": cfg.rationing,
            "demand": cfg.demand,
            "seed": cfg.seed,
            "git_sha": git_sha(),
        }


def _write_index(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stable sort for diffs
    rows_sorted = sorted(rows, key=lambda r: r.get("run", ""))
    (out_dir / "index.json").write_text(json.dumps(rows_sorted, indent=2))


def _load_index(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "index.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


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
    p = argparse.ArgumentParser(description="Parallel Tier-1 IPPO matrix (R1)")
    p.add_argument("--regimes", default="A,B,C")
    p.add_argument("--topologies", default="serial,y")
    p.add_argument("--caps", default="inf,1.2,1.0,0.8")
    p.add_argument("--rationing", default="proportional,uniform,honesty_weighted")
    p.add_argument("--demands", default="ar1,regime_switch")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=400_000,
        help="env steps/cell; default ≈ M3 update count at n_envs=64, rollout=128",
    )
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--minibatch-size", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=52)
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    p.add_argument("--workers", type=int, default=8, help="parallel cells (processes)")
    p.add_argument("--out-dir", default="artifacts/runs/ippo/tier1_v11")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print cell count and exit")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny timesteps / 1 seed for runner validation",
    )
    args = p.parse_args()

    regimes = tuple(x.strip() for x in args.regimes.split(",") if x.strip())
    topologies = tuple(x.strip() for x in args.topologies.split(",") if x.strip())
    caps = tuple(parse_caps(args.caps))
    rations = tuple(x.strip() for x in args.rationing.split(",") if x.strip())
    demands = tuple(x.strip() for x in args.demands.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    if args.smoke:
        seeds = (0,)
        args.total_timesteps = min(args.total_timesteps, 2048)
        args.n_envs = min(args.n_envs, 4)
        args.rollout_steps = min(args.rollout_steps, 64)
        args.workers = min(args.workers, 2)

    summary = prune_summary(
        regimes=regimes,
        topologies=topologies,
        caps=caps,
        rationing=rations,
        demands=demands,
        seeds=seeds,
    )
    cells = enumerate_cells(
        regimes=regimes,
        topologies=topologies,
        caps=caps,
        rationing=rations,
        demands=demands,
        seeds=seeds,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prune_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "matrix_meta.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "prune_summary": summary,
                "defaults": {
                    "total_timesteps": args.total_timesteps,
                    "n_envs": args.n_envs,
                    "rollout_steps": args.rollout_steps,
                    "device": args.device,
                    "workers": args.workers,
                },
            },
            indent=2,
        )
    )

    print(
        f"Tier-1 matrix: {summary['kept']} cells "
        f"(pruned {summary['pruned']} of {summary['full_cartesian']}) "
        f"→ {out_dir}  workers={args.workers} n_envs={args.n_envs} device={args.device}"
    )
    print(f"  prune reasons: {summary['prune_reasons']}")
    if args.dry_run:
        for c in cells[:5]:
            print(" ", c.run_name)
        if len(cells) > 5:
            print(f"  ... +{len(cells)-5} more")
        return 0

    payloads = []
    for cell in cells:
        cfg = cell.to_config(
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            n_envs=args.n_envs,
            minibatch_size=args.minibatch_size,
            horizon=args.horizon,
            device=args.device,
            out_dir=str(out_dir),
            run_name=cell.run_name,
        )
        if args.skip_existing and _cell_done(out_dir, cell.run_name):
            # Still submit so index is refreshed from disk, or skip submit?
            # Prefer skip submit and stitch from disk for speed.
            continue
        payloads.append(
            {"config": asdict(cfg), "skip_existing": bool(args.skip_existing)}
        )

    # Seed index with already-completed cells
    index_by_run: dict[str, dict[str, Any]] = {
        r["run"]: r for r in _load_index(out_dir) if "run" in r
    }
    if args.skip_existing:
        for cell in cells:
            fe_path = out_dir / cell.run_name / "final_eval.json"
            if fe_path.exists() and cell.run_name not in index_by_run:
                fe = json.loads(fe_path.read_text())
                index_by_run[cell.run_name] = {
                    "run": cell.run_name,
                    "status": "skipped",
                    "regime": cell.regime,
                    "topology": cell.topology,
                    "capacity_mult": cell.capacity_mult,
                    "capacity_tag": capacity_tag(cell.to_config()),
                    "rationing": cell.rationing,
                    "demand": cell.demand,
                    "seed": cell.seed,
                    **{k: v for k, v in fe.items() if isinstance(v, (int, float, str, bool))},
                }

    print(f"Queued {len(payloads)} cells ({len(index_by_run)} already indexed/complete)")
    t0 = time.time()
    workers = max(1, args.workers)

    if workers == 1 or len(payloads) <= 1:
        for i, payload in enumerate(payloads):
            print(f"\n=== [{i+1}/{len(payloads)}] sequential ===", flush=True)
            row = _run_one(payload)
            index_by_run[row["run"]] = row
            _write_index(out_dir, list(index_by_run.values()))
            print(f"  → {row['status']} {row['run']} ({row.get('elapsed_s', 0):.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, pl): pl for pl in payloads}
            done_n = 0
            for fut in as_completed(futures):
                done_n += 1
                row = fut.result()
                index_by_run[row["run"]] = row
                _write_index(out_dir, list(index_by_run.values()))
                status = row["status"]
                err = f" err={row.get('error')}" if status == "error" else ""
                print(
                    f"[{done_n}/{len(payloads)}] {status} {row['run']} "
                    f"({row.get('elapsed_s', 0):.0f}s){err}",
                    flush=True,
                )

    _write_index(out_dir, list(index_by_run.values()))
    n_ok = sum(1 for r in index_by_run.values() if r.get("status") in ("ok", "skipped"))
    n_err = sum(1 for r in index_by_run.values() if r.get("status") == "error")
    print(
        f"\nDone in {time.time()-t0:.0f}s — index {len(index_by_run)} rows "
        f"(ok/skip={n_ok}, err={n_err}) → {out_dir}/index.json"
    )
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
