#!/usr/bin/env python3
"""D5 — Does rationing ever bind? (re-eval frozen tight-capacity checkpoints)."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.common import (  # noqa: E402
    CAP_LABEL,
    CACHE_DIR,
    DIAG_EVAL_SEED_OFFSET,
    FIG_DIR,
    TIGHT_CAPS,
    ci95,
    ensure_dirs,
    list_m3_runs,
    load_trainer,
    read_json,
    write_json,
)
from analysis.diag.eval_ablation import evaluate_with_ablation  # noqa: E402


def run(
    *,
    n_episodes: int = 30,
    rationing: str = "proportional",
    max_seeds: int | None = None,
    force: bool = False,
) -> dict:
    ensure_dirs()
    cache_path = CACHE_DIR / f"d5_rationing_{rationing}_ep{n_episodes}.json"
    if cache_path.exists() and not force:
        print(f"D5: loading cache {cache_path}")
        return read_json(cache_path)

    # Also include looser caps for context, but focus interpretation on tight.
    runs = list_m3_runs(rationing=rationing)
    if max_seeds is not None:
        runs = [r for r in runs if int(r["seed"]) < max_seeds]

    by_cap = defaultdict(
        lambda: {
            "cap_bind_frac": [],
            "alloc_frac": [],
            "infl_bind": [],
            "infl_nonbind": [],
            "rationed_frac": [],
        }
    )
    per_run = []

    for i, row in enumerate(runs):
        run_dir = Path(row["run_dir"])
        tag = row["capacity_tag"]
        seed = int(row["seed"])
        print(f"D5 [{i+1}/{len(runs)}] {run_dir.name}", flush=True)
        trainer, meta = load_trainer(run_dir)
        m = evaluate_with_ablation(
            trainer,
            n_episodes=n_episodes,
            seed=DIAG_EVAL_SEED_OFFSET + 30_000 + seed,
            ablation_mode="intact",
            collect_steps=True,
        )
        steps = m["steps"]
        n = len(steps)
        cap_bind = sum(1 for s in steps if s["capacity_binds"])
        alloc = sum(1 for s in steps if s["allocation_triggers"])
        rationed = sum(1 for s in steps if s["rationed"])

        infl_bind = [
            s["factory_order_inflation"]
            for s in steps
            if s["capacity_binds"] and s["factory_order_inflation"] == s["factory_order_inflation"]
        ]
        infl_non = [
            s["factory_order_inflation"]
            for s in steps
            if (not s["capacity_binds"])
            and s["factory_order_inflation"] == s["factory_order_inflation"]
        ]

        rec = {
            "run": row["run"],
            "capacity_tag": tag,
            "seed": seed,
            "capacity_abs": meta.get("capacity"),
            "n_weeks": n,
            "frac_capacity_binds": cap_bind / max(n, 1),
            "frac_allocation_triggers": alloc / max(n, 1),
            "frac_rationed_flag": rationed / max(n, 1),
            "mean_inflation_when_cap_binds": float(np.mean(infl_bind)) if infl_bind else float("nan"),
            "mean_inflation_when_not_cap_binds": float(np.mean(infl_non)) if infl_non else float("nan"),
        }
        per_run.append(rec)
        by_cap[tag]["cap_bind_frac"].append(rec["frac_capacity_binds"])
        by_cap[tag]["alloc_frac"].append(rec["frac_allocation_triggers"])
        by_cap[tag]["rationed_frac"].append(rec["frac_rationed_flag"])
        if infl_bind:
            by_cap[tag]["infl_bind"].append(rec["mean_inflation_when_cap_binds"])
        if infl_non:
            by_cap[tag]["infl_nonbind"].append(rec["mean_inflation_when_not_cap_binds"])
        del trainer

    summary = []
    for tag in ["inf", "1p5mu", "1p2mu", "1p0mu", "0p8mu"]:
        if tag not in by_cap:
            continue
        d = by_cap[tag]
        cb_m, cb_ci, n = ci95(d["cap_bind_frac"])
        al_m, al_ci, _ = ci95(d["alloc_frac"])
        ib_m, ib_ci, _ = ci95(d["infl_bind"]) if d["infl_bind"] else (float("nan"), float("nan"), 0)
        in_m, in_ci, _ = ci95(d["infl_nonbind"]) if d["infl_nonbind"] else (float("nan"), float("nan"), 0)
        summary.append(
            {
                "capacity_tag": tag,
                "capacity_label": CAP_LABEL[tag],
                "n": n,
                "frac_capacity_binds_mean": cb_m,
                "frac_capacity_binds_ci95": cb_ci,
                "frac_allocation_triggers_mean": al_m,
                "frac_allocation_triggers_ci95": al_ci,
                "inflation_when_binding_mean": ib_m,
                "inflation_when_binding_ci95": ib_ci,
                "inflation_when_not_binding_mean": in_m,
                "inflation_when_not_binding_ci95": in_ci,
                "is_tight": tag in TIGHT_CAPS,
            }
        )

    # Log field note
    log_note = (
        "Training logs (final_eval/history) do NOT capture per-week capacity-bind or "
        "allocation-shortfall events — only aggregate eval/inflation_rate on info.rationed weeks. "
        "D5 recomputes these by re-running eval episodes on frozen checkpoints."
    )

    tight = [s for s in summary if s["is_tight"]]
    alloc_tight = float(np.mean([s["frac_allocation_triggers_mean"] for s in tight])) if tight else float("nan")
    cap_tight = float(np.mean([s["frac_capacity_binds_mean"] for s in tight])) if tight else float("nan")
    if alloc_tight < 0.20 and cap_tight < 0.20:
        interpretation = "gaming incentive absent / rare (STRUCTURAL — capacity too soft)"
    elif alloc_tight < 0.20:
        interpretation = "allocation rarely binds; capacity binds more often"
    else:
        interpretation = "gaming incentive present (allocation binds often)"

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(summary))
    w = 0.35
    axes[0].bar(
        x - w / 2,
        [s["frac_capacity_binds_mean"] for s in summary],
        w,
        yerr=[s["frac_capacity_binds_ci95"] for s in summary],
        capsize=2,
        label="factory capacity binds",
        color="#a33b2b",
    )
    axes[0].bar(
        x + w / 2,
        [s["frac_allocation_triggers_mean"] for s in summary],
        w,
        yerr=[s["frac_allocation_triggers_ci95"] for s in summary],
        capsize=2,
        label="any-node shortfall",
        color="#4a6fa5",
    )
    axes[0].axhline(0.20, color="gray", ls="--", lw=1, label="20% threshold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s["capacity_label"] for s in summary])
    axes[0].set_ylabel("Fraction of weeks")
    axes[0].set_title("D5 binding frequency")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(
        x - w / 2,
        [s["inflation_when_binding_mean"] for s in summary],
        w,
        yerr=[s["inflation_when_binding_ci95"] for s in summary],
        capsize=2,
        label="cap-binding weeks",
        color="#a33b2b",
    )
    axes[1].bar(
        x + w / 2,
        [s["inflation_when_not_binding_mean"] for s in summary],
        w,
        yerr=[s["inflation_when_not_binding_ci95"] for s in summary],
        capsize=2,
        label="non-binding weeks",
        color="#2f6f4e",
    )
    axes[1].axhline(1.0, color="gray", ls=":", lw=1)
    axes[1].axhline(1.5, color="gray", ls="--", lw=1, label="1.5× inflation detector")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s["capacity_label"] for s in summary])
    axes[1].set_ylabel("Factory order / incoming need")
    axes[1].set_title("D5 order-inflation ratio")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d5_rationing_binding.png", dpi=160)
    plt.close(fig)

    result = {
        "n_episodes": n_episodes,
        "log_gap": log_note,
        "summary": summary,
        "tight_mean_alloc_frac": alloc_tight,
        "tight_mean_cap_bind_frac": cap_tight,
        "interpretation_key": interpretation,
        "serial_topology_note": (
            "On serial chain, proportional allocation with a single claimant is identity "
            "fill-to-available; 'allocation triggers' = physical shortfall (backlog>0 after fill)."
        ),
        "per_run": per_run,
    }
    write_json(cache_path, result)
    write_json(CACHE_DIR / "d5_summary.json", {k: v for k, v in result.items() if k != "per_run"})
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-seeds", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(n_episodes=args.episodes, max_seeds=args.max_seeds, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
