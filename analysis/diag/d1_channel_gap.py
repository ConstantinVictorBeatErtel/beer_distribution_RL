#!/usr/bin/env python3
"""D1 — Regime A vs B cost gap from existing run logs only."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.common import (  # noqa: E402
    CAP_LABEL,
    CAP_ORDER,
    FIG_DIR,
    M2_A,
    M2_C,
    M3_DIR,
    ROLES_ORDER,
    CACHE_DIR,
    ci95,
    ensure_dirs,
    load_index,
    read_json,
    write_json,
)


def run() -> dict:
    ensure_dirs()
    rows = load_index()

    # Group Regime B final eval costs by capacity (proportional; honesty_weighted ≡ serial).
    by_cap: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "system": [],
            **{f"{r}": [] for r in ROLES_ORDER},
            "rationing": [],
        }
    )
    for r in rows:
        if r.get("regime") != "B":
            continue
        # Prefer proportional for primary A/B comparison table; also keep both.
        if r.get("rationing") != "proportional":
            continue
        tag = r.get("capacity_tag", "inf")
        by_cap[tag]["system"].append(float(r["eval/mean_system_cost"]))
        for role in ROLES_ORDER:
            by_cap[tag][role].append(float(r[f"eval/{role}_cost"]))

    # Regime A/C: only M2 classic_step seed0 exists — no capacity matrix.
    a_meta = read_json(M2_A / "run_meta.json") if (M2_A / "run_meta.json").exists() else {}
    a_eval = read_json(M2_A / "final_eval.json") if (M2_A / "final_eval.json").exists() else {}
    c_eval = read_json(M2_C / "final_eval.json") if (M2_C / "final_eval.json").exists() else {}

    a_runs = list((M3_DIR.parent).glob("**/regimeA_*"))
    a_matrix_cells = [
        p
        for p in (M3_DIR.glob("regimeA_*") if M3_DIR.exists() else [])
    ]

    capacity_table = []
    for tag in CAP_ORDER:
        d = by_cap.get(tag)
        if not d or not d["system"]:
            continue
        sm, sci, sn = ci95(d["system"])
        row = {
            "capacity_tag": tag,
            "capacity_label": CAP_LABEL[tag],
            "n_B": sn,
            "B_system_mean": sm,
            "B_system_ci95": sci,
            "A_system_mean": None,
            "A_system_ci95": None,
            "n_A": 0,
            "gap_abs": None,
            "gap_pct_of_A": None,
            "gap_vs_ci": "no_regime_A_matrix",
            "per_agent_B": {},
        }
        for role in ROLES_ORDER:
            rm, rci, _ = ci95(d[role])
            row["per_agent_B"][role] = {"mean": rm, "ci95": rci}
        capacity_table.append(row)

    result = {
        "data_gap": (
            "Completed M3 matrix is Regime B only (5 caps × 2 rationing × 10 seeds). "
            "Regime A exists only as M2 classic_step seed0 (no capacity sweep, demand≠uniform). "
            "Matched A vs B cost gaps by capacity therefore cannot be computed from existing logs."
        ),
        "regime_A_available": {
            "path": str(M2_A),
            "demand": (a_meta.get("config") or {}).get("demand"),
            "capacity_mult": (a_meta.get("config") or {}).get("capacity_mult"),
            "seed": (a_meta.get("config") or {}).get("seed"),
            "final_eval": a_eval,
            "n_matrix_cells": len(a_matrix_cells),
            "extra_a_runs": [str(p) for p in a_runs],
        },
        "regime_C_available": {"path": str(M2_C), "final_eval": c_eval},
        "regime_B_by_capacity": capacity_table,
        "interpretation": (
            "D1 inconclusive on channel economic value: no Regime A capacity matrix. "
            "Defer to D2 (signal ablation) for whether the channel is load-bearing."
        ),
    }

    # Figure: Regime B system + per-agent costs by capacity (what we do have).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    tags = [r["capacity_tag"] for r in capacity_table]
    xs = np.arange(len(tags))
    means = [r["B_system_mean"] for r in capacity_table]
    cis = [r["B_system_ci95"] for r in capacity_table]
    axes[0].bar(xs, means, yerr=cis, color="#4a6fa5", capsize=3, alpha=0.9)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([CAP_LABEL[t] for t in tags])
    axes[0].set_ylabel("Mean system cost / period")
    axes[0].set_title("Regime B only (no matched Regime A)")
    axes[0].grid(True, axis="y", alpha=0.3)

    width = 0.18
    for i, role in enumerate(ROLES_ORDER):
        m = [r["per_agent_B"][role]["mean"] for r in capacity_table]
        e = [r["per_agent_B"][role]["ci95"] for r in capacity_table]
        axes[1].bar(xs + (i - 1.5) * width, m, width, yerr=e, capsize=2, label=role)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([CAP_LABEL[t] for t in tags])
    axes[1].set_ylabel("Mean local cost / period")
    axes[1].set_title("Regime B per-agent costs")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("D1: A vs B cost gap — Regime A capacity matrix missing", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d1_regime_b_costs.png", dpi=160)
    plt.close(fig)

    # Secondary: M2 A vs C classic (not A vs B, but documents available anchors).
    if a_eval and c_eval:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        labels = ["Regime A\n(classic, seed0)", "Regime C\n(classic, seed0)"]
        vals = [a_eval["eval/mean_system_cost"], c_eval["eval/mean_system_cost"]]
        ax.bar(labels, vals, color=["#6b7c8a", "#2f6f4e"])
        ax.set_ylabel("Mean system cost / period")
        ax.set_title("M2 anchors only (not capacity-matched A vs B)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "d1_m2_ac_anchors.png", dpi=160)
        plt.close(fig)

    write_json(CACHE_DIR / "d1_result.json", result)
    return result


if __name__ == "__main__":
    out = run()
    print(out["data_gap"])
    print("Wrote", CACHE_DIR / "d1_result.json")
