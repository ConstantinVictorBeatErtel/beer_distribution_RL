#!/usr/bin/env python3
"""D2 — Signal ablation at eval (frozen Regime-B checkpoints)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.common import (  # noqa: E402
    ABLATIONS,
    CAP_LABEL,
    CAP_ORDER,
    CACHE_DIR,
    DIAG_EVAL_SEED_OFFSET,
    FIG_DIR,
    ci95,
    ensure_dirs,
    list_m3_runs,
    load_trainer,
    write_json,
    read_json,
)
from analysis.diag.eval_ablation import evaluate_with_ablation  # noqa: E402


def run(
    *,
    n_episodes: int = 100,
    rationing: str = "proportional",
    max_seeds: int | None = None,
    force: bool = False,
) -> dict:
    ensure_dirs()
    cache_path = CACHE_DIR / f"d2_ablation_{rationing}_ep{n_episodes}.json"
    partial_path = CACHE_DIR / f"d2_partial_{rationing}_ep{n_episodes}.jsonl"
    if cache_path.exists() and not force:
        print(f"D2: loading cache {cache_path}")
        return read_json(cache_path)

    runs = list_m3_runs(rationing=rationing)
    if max_seeds is not None:
        runs = [r for r in runs if int(r["seed"]) < max_seeds]

    done: dict[str, dict] = {}
    if partial_path.exists() and not force:
        for line in partial_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["run"]] = rec
        print(f"D2: resumed {len(done)} cached runs from {partial_path}", flush=True)
    elif force and partial_path.exists():
        partial_path.unlink()

    per_run = []
    for i, row in enumerate(runs):
        if row["run"] in done:
            per_run.append(done[row["run"]])
            continue
        run_dir = Path(row["run_dir"])
        tag = row["capacity_tag"]
        seed = int(row["seed"])
        print(f"D2 [{i+1}/{len(runs)}] {run_dir.name}", flush=True)
        trainer, _meta = load_trainer(run_dir)
        run_metrics = {"run": row["run"], "capacity_tag": tag, "seed": seed, "conditions": {}}
        for j, cond in enumerate(ABLATIONS):
            eval_seed = DIAG_EVAL_SEED_OFFSET + seed * 1000 + j * 100
            m = evaluate_with_ablation(
                trainer,
                n_episodes=n_episodes,
                seed=eval_seed,
                ablation_mode=cond.mode,
            )
            run_metrics["conditions"][cond.name] = {
                "system_cost": m["eval/mean_system_cost"],
                "retailer_cost": m["eval/retailer_cost"],
                "wholesaler_cost": m["eval/wholesaler_cost"],
                "distributor_cost": m["eval/distributor_cost"],
                "factory_cost": m["eval/factory_cost"],
                "bullwhip_ratio": m["eval/bullwhip_ratio"],
                "sharing_rate": m.get("eval/sharing_rate"),
            }
        per_run.append(run_metrics)
        with partial_path.open("a") as f:
            f.write(json.dumps(run_metrics) + "\n")
        del trainer

    # Aggregate
    agg: dict = defaultdict(lambda: defaultdict(list))
    for run_metrics in per_run:
        tag = run_metrics["capacity_tag"]
        for cond in ABLATIONS:
            if cond.name in run_metrics["conditions"]:
                agg[tag][cond.name].append(run_metrics["conditions"][cond.name])

    summary = []
    for tag in CAP_ORDER:
        if tag not in agg:
            continue
        entry = {"capacity_tag": tag, "capacity_label": CAP_LABEL[tag], "conditions": {}}
        for cond in ABLATIONS:
            rows = agg[tag][cond.name]
            if not rows:
                continue
            sys_m, sys_ci, n = ci95([r["system_cost"] for r in rows])
            bw_m, bw_ci, _ = ci95([r["bullwhip_ratio"] for r in rows])
            entry["conditions"][cond.name] = {
                "n": n,
                "system_cost_mean": sys_m,
                "system_cost_ci95": sys_ci,
                "bullwhip_mean": bw_m,
                "bullwhip_ci95": bw_ci,
                "per_agent": {
                    role: {
                        "mean": ci95([r[f"{role}_cost"] for r in rows])[0],
                        "ci95": ci95([r[f"{role}_cost"] for r in rows])[1],
                    }
                    for role in ("retailer", "wholesaler", "distributor", "factory")
                },
            }
        # Channel load-bearing: intact must be clearly *better* (lower cost) than ablations.
        if "intact" in entry["conditions"]:
            intact = entry["conditions"]["intact"]["system_cost_mean"]
            intact_ci = entry["conditions"]["intact"]["system_cost_ci95"]
            abl = {
                c.name: entry["conditions"][c.name]
                for c in ABLATIONS
                if c.name != "intact" and c.name in entry["conditions"]
            }
            abl_costs = [v["system_cost_mean"] for v in abl.values()]
            gaps = [c - intact for c in abl_costs]  # positive ⇒ intact better
            entry["max_abs_gap_vs_intact"] = float(max(abs(g) for g in gaps)) if gaps else float("nan")
            entry["mean_ablation_minus_intact"] = float(sum(gaps) / len(gaps)) if gaps else float("nan")
            entry["intact_ci95"] = intact_ci
            # Intact better than every ablation by more than CI, and >5% relative.
            clearly_better = all(
                (c - intact) > max(intact_ci, 0.05 * max(intact, 1e-9)) for c in abl_costs
            )
            clearly_similar = all(
                abs(c - intact) <= max(intact_ci, 0.05 * max(intact, 1e-9)) for c in abl_costs
            )
            entry["channel_load_bearing"] = bool(clearly_better)
            entry["channel_ignored"] = bool(clearly_similar)
            entry["channel_harmful"] = bool(
                gaps and float(sum(gaps) / len(gaps)) < -max(intact_ci, 0.05 * max(intact, 1e-9))
            )
        summary.append(entry)

    # Plot: side-by-side conditions per capacity
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    cond_names = [c.name for c in ABLATIONS]
    colors = {"intact": "#2f6f4e", "zeroed": "#a33b2b", "shuffled": "#c45c26", "random": "#6b7c8a"}
    x = np.arange(len(summary))
    width = 0.18
    for i, cname in enumerate(cond_names):
        means = [s["conditions"][cname]["system_cost_mean"] for s in summary]
        errs = [s["conditions"][cname]["system_cost_ci95"] for s in summary]
        axes[0].bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=errs,
            capsize=2,
            label=cname,
            color=colors[cname],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s["capacity_label"] for s in summary])
    axes[0].set_ylabel("System cost / period")
    axes[0].set_title(f"D2 signal ablation — system cost ({rationing}, {n_episodes} eps/seed)")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    for i, cname in enumerate(cond_names):
        means = [s["conditions"][cname]["bullwhip_mean"] for s in summary]
        errs = [s["conditions"][cname]["bullwhip_ci95"] for s in summary]
        axes[1].bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=errs,
            capsize=2,
            label=cname,
            color=colors[cname],
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s["capacity_label"] for s in summary])
    axes[1].set_ylabel("Bullwhip ratio (factory / demand var)")
    axes[1].set_title("D2 signal ablation — bullwhip")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d2_signal_ablation.png", dpi=160)
    plt.close(fig)

    # Per-agent cost panel at tight capacity
    tight = [s for s in summary if s["capacity_tag"] in ("1p0mu", "0p8mu")]
    if tight:
        fig, axes = plt.subplots(1, len(tight), figsize=(5 * len(tight), 4), squeeze=False)
        roles = ["retailer", "wholesaler", "distributor", "factory"]
        for ax, s in zip(axes[0], tight):
            xr = np.arange(len(roles))
            for i, cname in enumerate(cond_names):
                means = [s["conditions"][cname]["per_agent"][r]["mean"] for r in roles]
                errs = [s["conditions"][cname]["per_agent"][r]["ci95"] for r in roles]
                ax.bar(xr + (i - 1.5) * width, means, width, yerr=errs, capsize=2, label=cname, color=colors[cname])
            ax.set_xticks(xr)
            ax.set_xticklabels(roles, rotation=20)
            ax.set_title(f"Per-agent cost @ {s['capacity_label']}")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "d2_ablation_per_agent_tight.png", dpi=160)
        plt.close(fig)

    n_bearing = sum(1 for s in summary if s.get("channel_load_bearing"))
    n_ignored = sum(1 for s in summary if s.get("channel_ignored"))
    n_harmful = sum(1 for s in summary if s.get("channel_harmful"))
    if n_bearing == 0 and (n_ignored + n_harmful) >= len(summary) // 2:
        interpretation = "listeners ignore channel (babbling / STRUCTURAL)"
    elif n_bearing == len(summary):
        interpretation = "channel load-bearing"
    else:
        interpretation = (
            f"mixed (load-bearing={n_bearing}, ignored={n_ignored}, harmful={n_harmful})"
        )

    result = {
        "n_episodes": n_episodes,
        "rationing": rationing,
        "n_checkpoints": len(runs),
        "summary": summary,
        "per_run": per_run,
        "interpretation_key": interpretation,
        "note": (
            "Ablation corrupts listener observations only; policies still emit signals. "
            "Primary matrix uses proportional rationing (honesty_weighted ≡ serial fill)."
        ),
    }
    write_json(cache_path, result)
    # lighter summary without per_run for report
    write_json(CACHE_DIR / "d2_summary.json", {k: v for k, v in result.items() if k != "per_run"})
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--rationing", default="proportional")
    p.add_argument("--max-seeds", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(n_episodes=args.episodes, rationing=args.rationing, max_seeds=args.max_seeds, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
