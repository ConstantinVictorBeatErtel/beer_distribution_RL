#!/usr/bin/env python3
"""D4 — Sharing action: indifference vs converged preference."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from beer_distribution_rl.env.core import ROLES, Role  # noqa: E402

from analysis.diag.common import (  # noqa: E402
    CAP_LABEL,
    CAP_ORDER,
    CACHE_DIR,
    DIAG_EVAL_SEED_OFFSET,
    FIG_DIR,
    M3_DIR,
    ci95,
    ensure_dirs,
    list_m3_runs,
    load_trainer,
    read_json,
    write_json,
)


def _broadcast_entropy_and_share(trainer, n_episodes: int, seed: int) -> dict[str, float]:
    """Per-role Bernoulli entropy of broadcast head + empirical share rate."""
    core = trainer.core
    ent_acc = {r: [] for r in ROLES}
    share_n = {r: 0 for r in ROLES}
    share_yes = {r: 0 for r in ROLES}
    for ep in range(n_episodes):
        states = core.reset(seed + ep)
        done = False
        while not done:
            orders = {}
            signals = {}
            with torch.no_grad():
                for r in ROLES:
                    o = torch.as_tensor(trainer._obs(states, r), device=trainer.device).unsqueeze(0)
                    bdist = trainer.policies[r]._dists(o)[1]  # type: ignore[attr-defined]
                    p = float(bdist.probs[0, 1].item())
                    p = min(max(p, 1e-8), 1 - 1e-8)
                    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
                    ent_acc[r].append(h)
                    a, _, _ = trainer._policy_act(r, o, greedy=False)
                    row = a.squeeze(0).cpu().numpy().astype(int)
                    orders[r] = trainer._decode_order(int(row[0]), states[r])
                    signals[r] = trainer._decode_signal(
                        states[r], int(row[1]), int(row[2]), int(row[3])
                    )
                    share_n[r] += 1
                    if signals[r] is not None:
                        share_yes[r] += 1
            states, _, done, _ = core.step(orders, signals)
    out = {}
    for r in ROLES:
        name = r.name.lower()
        out[f"{name}_broadcast_entropy"] = float(np.mean(ent_acc[r])) if ent_acc[r] else float("nan")
        out[f"{name}_share_rate"] = share_yes[r] / max(share_n[r], 1)
        out[f"{name}_p_share"] = out[f"{name}_share_rate"]
    out["mean_broadcast_entropy"] = float(
        np.mean([out[f"{r.name.lower()}_broadcast_entropy"] for r in ROLES])
    )
    out["mean_share_rate"] = float(np.mean([out[f"{r.name.lower()}_share_rate"] for r in ROLES]))
    return out


def _share_trajectory(history: list[dict]) -> dict:
    xs, ys = [], []
    for row in history:
        if "eval/sharing_rate" in row:
            xs.append(int(row.get("update", row.get("step", len(xs)))))
            ys.append(float(row["eval/sharing_rate"]))
    if len(ys) < 2:
        return {"updates": xs, "share_rates": ys, "end_slope": float("nan"), "still_moving": False}
    # Slope over last third of logged eval points
    k = max(2, len(ys) // 3)
    x = np.arange(k, dtype=float)
    y = np.asarray(ys[-k:], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    # "Still moving" if |Δ| over last third > 0.05 absolute share-rate
    still = abs(y[-1] - y[0]) > 0.05
    return {
        "updates": xs,
        "share_rates": ys,
        "end_slope": slope,
        "end_delta": float(y[-1] - y[0]),
        "still_moving": still,
        "final_share": float(ys[-1]),
    }


def run(
    *,
    n_episodes: int = 20,
    rationing: str = "proportional",
    max_seeds: int | None = None,
    force: bool = False,
) -> dict:
    ensure_dirs()
    cache_path = CACHE_DIR / f"d4_sharing_{rationing}.json"
    if cache_path.exists() and not force:
        print(f"D4: loading cache {cache_path}")
        return read_json(cache_path)

    runs = list_m3_runs(rationing=rationing)
    if max_seeds is not None:
        runs = [r for r in runs if int(r["seed"]) < max_seeds]

    H_MAX = float(np.log(2))  # Bernoulli max entropy ≈ 0.693
    by_cap_final_share: dict[str, list[float]] = defaultdict(list)
    by_cap_entropy: dict[str, list[float]] = defaultdict(list)
    still_moving_flags = []
    per_run = []
    traj_examples = []

    for i, row in enumerate(runs):
        run_dir = Path(row["run_dir"])
        tag = row["capacity_tag"]
        seed = int(row["seed"])
        print(f"D4 [{i+1}/{len(runs)}] {run_dir.name}", flush=True)

        hist_path = run_dir / "history.json"
        traj = {"updates": [], "share_rates": [], "still_moving": False, "end_slope": float("nan")}
        if hist_path.exists():
            history = read_json(hist_path)
            traj = _share_trajectory(history)
            still_moving_flags.append(bool(traj["still_moving"]))
            if seed == 0:
                traj_examples.append({"capacity_tag": tag, "seed": seed, **traj})

        trainer, _ = load_trainer(run_dir)
        stats = _broadcast_entropy_and_share(
            trainer,
            n_episodes=n_episodes,
            seed=DIAG_EVAL_SEED_OFFSET + 20_000 + seed,
        )
        del trainer

        # Prefer final_eval sharing if present
        final_share = float(row.get("eval/sharing_rate", stats["mean_share_rate"]))
        by_cap_final_share[tag].append(final_share)
        by_cap_entropy[tag].append(stats["mean_broadcast_entropy"])

        per_run.append(
            {
                "run": row["run"],
                "capacity_tag": tag,
                "seed": seed,
                "final_share_rate": final_share,
                "mean_broadcast_entropy": stats["mean_broadcast_entropy"],
                "entropy_frac_of_max": stats["mean_broadcast_entropy"] / H_MAX,
                "still_moving": traj.get("still_moving", False),
                "end_delta": traj.get("end_delta"),
                "per_role": {
                    r.name.lower(): {
                        "broadcast_entropy": stats[f"{r.name.lower()}_broadcast_entropy"],
                        "share_rate": stats[f"{r.name.lower()}_share_rate"],
                    }
                    for r in ROLES
                },
            }
        )

    summary = []
    all_shares = []
    for tag in CAP_ORDER:
        if tag not in by_cap_final_share:
            continue
        shares = by_cap_final_share[tag]
        ents = by_cap_entropy[tag]
        sm, sci, n = ci95(shares)
        em, eci, _ = ci95(ents)
        all_shares.extend(shares)
        # Unimodal around 0.5 vs bimodal near 0/1: use fraction in (0.35, 0.65)
        mid = sum(1 for s in shares if 0.35 <= s <= 0.65) / max(len(shares), 1)
        extremes = sum(1 for s in shares if s < 0.2 or s > 0.8) / max(len(shares), 1)
        summary.append(
            {
                "capacity_tag": tag,
                "capacity_label": CAP_LABEL[tag],
                "n": n,
                "share_mean": sm,
                "share_ci95": sci,
                "entropy_mean": em,
                "entropy_ci95": eci,
                "entropy_frac_of_max": em / H_MAX,
                "frac_near_half": mid,
                "frac_extreme": extremes,
            }
        )

    frac_moving = float(np.mean(still_moving_flags)) if still_moving_flags else float("nan")
    mean_ent_frac = float(np.mean([s["entropy_frac_of_max"] for s in summary])) if summary else float("nan")
    mean_near_half = float(np.mean([s["frac_near_half"] for s in summary])) if summary else float("nan")

    if mean_ent_frac > 0.85 and frac_moving < 0.3 and mean_near_half > 0.6:
        interpretation = "indifference (STRUCTURAL)"
    elif frac_moving >= 0.3:
        interpretation = "undertrained (share-rate still trending)"
    elif mean_near_half < 0.4:
        interpretation = "multi-equilibrium (bimodal across seeds)"
    else:
        interpretation = "mixed / weak preference"

    # Figures
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    # (a) entropy
    x = np.arange(len(summary))
    axes[0].bar(
        x,
        [s["entropy_mean"] for s in summary],
        yerr=[s["entropy_ci95"] for s in summary],
        color="#4a6fa5",
        capsize=3,
    )
    axes[0].axhline(H_MAX, color="gray", ls="--", lw=1, label="max H=ln2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([s["capacity_label"] for s in summary])
    axes[0].set_ylabel("Broadcast Bernoulli entropy")
    axes[0].set_title("D4a end-of-train share entropy")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, axis="y", alpha=0.3)

    # (b) trajectories (seed0 examples)
    for ex in traj_examples:
        if ex["share_rates"]:
            axes[1].plot(ex["updates"], ex["share_rates"], marker="o", ms=3, label=CAP_LABEL[ex["capacity_tag"]])
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("eval sharing rate")
    axes[1].set_title("D4b share-rate trajectory (seed 0)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # (c) histogram of final share rates
    axes[2].hist(all_shares, bins=np.linspace(0, 1, 21), color="#2f6f4e", edgecolor="white")
    axes[2].axvline(0.5, color="gray", ls="--")
    axes[2].set_xlabel("Final sharing rate")
    axes[2].set_ylabel("Count (seeds × caps)")
    axes[2].set_title("D4c cross-seed share histogram")
    axes[2].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d4_sharing_preference.png", dpi=160)
    plt.close(fig)

    result = {
        "bernoulli_h_max": H_MAX,
        "summary": summary,
        "frac_still_moving": frac_moving,
        "mean_entropy_frac_of_max": mean_ent_frac,
        "mean_frac_near_half": mean_near_half,
        "interpretation_key": interpretation,
        "trajectory_examples": traj_examples,
        "per_run": per_run,
        "note": (
            "Broadcast entropy recomputed from frozen checkpoints (history only logs "
            "joint multi-head entropy). Share trajectories from history.json eval/sharing_rate."
        ),
    }
    write_json(cache_path, result)
    write_json(CACHE_DIR / "d4_summary.json", {k: v for k, v in result.items() if k != "per_run"})
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-seeds", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(n_episodes=args.episodes, max_seeds=args.max_seeds, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
