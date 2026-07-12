#!/usr/bin/env python3
"""Build phase diagram from M3 IPPO index.json artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ci95(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = float(np.mean(xs))
    if len(xs) < 2:
        return m, 0.0
    se = float(np.std(xs, ddof=1)) / math.sqrt(len(xs))
    return m, 1.96 * se


def capacity_tightness(tag: str, mult) -> float:
    """Map capacity to tightness in [0,1]: inf→0, 0.8μ→1."""
    if mult is None or tag == "inf":
        return 0.0
    # 1.5 → ~0, 0.8 → 1.0
    return float(min(1.0, max(0.0, (1.5 - float(mult)) / 0.7)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=Path("artifacts/runs/ippo/m3/index.json"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/runs/ippo/m3"))
    args = p.parse_args()

    rows = json.loads(args.index.read_text())
    # group: (rationing, capacity_tag) -> list of honesty / cost / share
    by = defaultdict(lambda: {"honesty": [], "cost": [], "share": [], "inflation": [], "mult": None})
    for r in rows:
        if r.get("regime") != "B":
            continue
        key = (r["rationing"], r.get("capacity_tag", "inf"))
        by[key]["honesty"].append(float(r.get("eval/honesty_score", 0.0)))
        by[key]["cost"].append(float(r.get("eval/mean_system_cost", float("nan"))))
        by[key]["share"].append(float(r.get("eval/sharing_rate", 0.0)))
        by[key]["inflation"].append(float(r.get("eval/inflation_rate", 0.0)))
        by[key]["mult"] = r.get("capacity_mult")

    summary = []
    for (rat, tag), d in sorted(by.items()):
        hm, he = _ci95(d["honesty"])
        cm, ce = _ci95(d["cost"])
        sm, se = _ci95(d["share"])
        im, ie = _ci95(d["inflation"])
        summary.append(
            {
                "rationing": rat,
                "capacity_tag": tag,
                "capacity_mult": d["mult"],
                "tightness": capacity_tightness(tag, d["mult"]),
                "n": len(d["honesty"]),
                "honesty_mean": hm,
                "honesty_ci95": he,
                "cost_mean": cm,
                "cost_ci95": ce,
                "sharing_mean": sm,
                "sharing_ci95": se,
                "inflation_mean": im,
                "inflation_ci95": ie,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "phase_summary.json").write_text(json.dumps(summary, indent=2))

    # Phase diagram: x=tightness, y=honesty
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"proportional": "#1f4e79", "honesty_weighted": "#c45c26", "uniform": "#2f6f4e"}
    for rat in sorted({s["rationing"] for s in summary}):
        pts = sorted([s for s in summary if s["rationing"] == rat], key=lambda z: z["tightness"])
        if not pts:
            continue
        xs = [p["tightness"] for p in pts]
        ys = [p["honesty_mean"] for p in pts]
        yerr = [p["honesty_ci95"] for p in pts]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            label=rat,
            color=colors.get(rat, "black"),
            linewidth=2,
            capsize=3,
        )
    ax.set_xlabel("Capacity tightness (0 = ∞, 1 = 0.8μ)")
    ax.set_ylabel("Honesty score (−mean |claim−truth| / cap)")
    ax.set_title("Tier-1 phase diagram: capacity tightness vs signaling honesty")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = args.out_dir / "phase_diagram.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    # Cost panel
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for rat in sorted({s["rationing"] for s in summary}):
        pts = sorted([s for s in summary if s["rationing"] == rat], key=lambda z: z["tightness"])
        xs = [p["tightness"] for p in pts]
        ys = [p["cost_mean"] for p in pts]
        yerr = [p["cost_ci95"] for p in pts]
        ax.errorbar(xs, ys, yerr=yerr, marker="s", label=rat, color=colors.get(rat, "black"), capsize=3)
    ax.set_xlabel("Capacity tightness")
    ax.set_ylabel("Mean system cost / period")
    ax.set_title("Tier-1 system cost vs capacity tightness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "cost_vs_tightness.png", dpi=160)
    plt.close(fig)

    # Markdown report
    lines = [
        "# M3 Phase diagram report",
        "",
        f"Source: `{args.index}`",
        "",
        "## Predictions",
        "",
        "- **P1** (slack capacity): Regime B shares honestly; system cost approaches Regime C.",
        "- **P2** (tight + proportional): order inflation / honesty collapse.",
        "- **P3** (honesty-weighted): truthful signaling restored.",
        "",
        "## Summary table",
        "",
        "| Rationing | Cap | n | Honesty ±CI | Share ±CI | Inflation ±CI | Cost ±CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(summary, key=lambda z: (z["rationing"], z["tightness"])):
        lines.append(
            f"| {s['rationing']} | {s['capacity_tag']} | {s['n']} | "
            f"{s['honesty_mean']:.3f}±{s['honesty_ci95']:.3f} | "
            f"{s['sharing_mean']:.2f}±{s['sharing_ci95']:.2f} | "
            f"{s['inflation_mean']:.2f}±{s['inflation_ci95']:.2f} | "
            f"{s['cost_mean']:.1f}±{s['cost_ci95']:.1f} |"
        )
    lines += [
        "",
        f"![phase diagram]({fig_path.name})",
        "",
        f"![cost](cost_vs_tightness.png)",
        "",
    ]
    (args.out_dir / "M3_REPORT.md").write_text("\n".join(lines))
    print(f"Wrote {args.out_dir}/phase_diagram.png and M3_REPORT.md ({len(summary)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
