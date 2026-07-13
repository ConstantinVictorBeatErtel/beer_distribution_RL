#!/usr/bin/env python3
"""D3 — Listener sensitivity probe (signal vs inventory feature)."""

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
    ci95,
    ensure_dirs,
    inventory_index,
    list_m3_runs,
    load_trainer,
    read_json,
    signal_feature_slice,
    write_json,
)


def _collect_obs_batch(trainer, n_episodes: int, seed: int, max_obs: int = 256) -> dict[Role, np.ndarray]:
    """Record a fixed batch of observations under intact rollout."""
    core = trainer.core
    bags: dict[Role, list] = {r: [] for r in ROLES}
    for ep in range(n_episodes):
        states = core.reset(seed + ep)
        done = False
        while not done and sum(len(bags[r]) for r in ROLES) < max_obs * 4:
            obs = {r: trainer._obs(states, r) for r in ROLES}
            for r in ROLES:
                if len(bags[r]) < max_obs:
                    bags[r].append(obs[r].copy())
            orders = {}
            signals = {} if trainer.signaling else None
            with torch.no_grad():
                for r in ROLES:
                    o = torch.as_tensor(obs[r], device=trainer.device).unsqueeze(0)
                    a, _, _ = trainer._policy_act(r, o, greedy=False)
                    row = a.squeeze(0).cpu().numpy().astype(int)
                    orders[r] = trainer._decode_order(int(row[0]), states[r])
                    if signals is not None:
                        signals[r] = trainer._decode_signal(
                            states[r], int(row[1]), int(row[2]), int(row[3])
                        )
            states, _, done, _ = core.step(orders, signals)
    return {r: np.stack(bags[r], axis=0) for r in ROLES if bags[r]}


def _mean_order_action(trainer, role: Role, obs: np.ndarray) -> np.ndarray:
    """Greedy expected order quantity (decoded) from order-head argmax."""
    pol = trainer.policies[role]
    with torch.no_grad():
        ot = torch.as_tensor(obs, device=trainer.device)
        dist = pol._dists(ot)[0]  # type: ignore[attr-defined]
        idx = dist.probs.argmax(dim=-1).cpu().numpy().astype(int)
    # Approximate decode using last_demand feature (obs index 3) * scale
    scale = 20.0
    last_dem = obs[:, 3] * scale
    delta = idx - trainer.cfg.action_delta_max
    raw = last_dem + delta
    return np.clip(raw, 0, trainer.cfg.order_cap)


def _sensitivity(
    trainer,
    role: Role,
    obs: np.ndarray,
    feature_indices: list[int],
    values: np.ndarray,
) -> float:
    """Mean |Δaction| per unit change in the swept feature(s), averaged over batch."""
    base = _mean_order_action(trainer, role, obs)
    # Sweep: set features to each value, measure action change vs midpoint
    mid = values[len(values) // 2]
    actions = []
    for v in values:
        o = obs.copy()
        for fi in feature_indices:
            o[:, fi] = v
        actions.append(_mean_order_action(trainer, role, o))
    actions = np.stack(actions, axis=0)  # [V, B]
    # Finite difference vs mid across value grid
    dv = float(values[-1] - values[0])
    if dv <= 1e-12:
        return 0.0
    dact = np.abs(actions[-1] - actions[0]).mean()
    return float(dact / dv)


def run(
    *,
    n_record_episodes: int = 4,
    max_obs: int = 128,
    rationing: str = "proportional",
    max_seeds: int | None = None,
    force: bool = False,
) -> dict:
    ensure_dirs()
    cache_path = CACHE_DIR / f"d3_sensitivity_{rationing}.json"
    if cache_path.exists() and not force:
        print(f"D3: loading cache {cache_path}")
        return read_json(cache_path)

    runs = list_m3_runs(rationing=rationing)
    if max_seeds is not None:
        runs = [r for r in runs if int(r["seed"]) < max_seeds]

    # Signal claimed_demand features: for each role block, index base+1, base+4, ...
    # Sweep claimed_demand channels (the informative ones) and present flags jointly.
    inv_idx = inventory_index()
    # Obs scaled: inventory in [0, ~2], claims in [0, ~0.75]
    inv_grid = np.linspace(0.0, 2.0, 9)  # 0..40 physical / 20
    sig_grid = np.linspace(0.0, 0.75, 9)  # 0..15 / 20

    by_cap: dict = defaultdict(lambda: {"signal": [], "inventory": [], "ratio": []})
    per_run = []

    for i, row in enumerate(runs):
        run_dir = Path(row["run_dir"])
        tag = row["capacity_tag"]
        seed = int(row["seed"])
        print(f"D3 [{i+1}/{len(runs)}] {run_dir.name}", flush=True)
        trainer, _ = load_trainer(run_dir)
        sig_sl = signal_feature_slice(trainer.obs_dim)
        # claimed_demand feature indices within signal board (present, dem, inv) * 4 roles
        dem_indices = list(range(sig_sl.start + 1, sig_sl.stop, 3))
        batch = _collect_obs_batch(
            trainer,
            n_episodes=n_record_episodes,
            seed=DIAG_EVAL_SEED_OFFSET + 9_000 + seed,
            max_obs=max_obs,
        )
        sig_sens = []
        inv_sens = []
        for r in ROLES:
            if r not in batch:
                continue
            obs = batch[r]
            s = _sensitivity(trainer, r, obs, dem_indices, sig_grid)
            iv = _sensitivity(trainer, r, obs, [inv_idx], inv_grid)
            sig_sens.append(s)
            inv_sens.append(iv)
        sig_m = float(np.mean(sig_sens)) if sig_sens else float("nan")
        inv_m = float(np.mean(inv_sens)) if inv_sens else float("nan")
        ratio = sig_m / inv_m if inv_m and inv_m > 1e-12 else float("nan")
        by_cap[tag]["signal"].append(sig_m)
        by_cap[tag]["inventory"].append(inv_m)
        by_cap[tag]["ratio"].append(ratio)
        per_run.append(
            {
                "run": row["run"],
                "capacity_tag": tag,
                "seed": seed,
                "signal_sensitivity": sig_m,
                "inventory_sensitivity": inv_m,
                "ratio_signal_over_inv": ratio,
                "per_role_signal": {r.name.lower(): sig_sens[j] for j, r in enumerate(ROLES) if j < len(sig_sens)},
                "per_role_inventory": {
                    r.name.lower(): inv_sens[j] for j, r in enumerate(ROLES) if j < len(inv_sens)
                },
            }
        )
        del trainer

    summary = []
    for tag in CAP_ORDER:
        if tag not in by_cap:
            continue
        sm, sci, n = ci95(by_cap[tag]["signal"])
        im, ici, _ = ci95(by_cap[tag]["inventory"])
        rm, rci, _ = ci95(by_cap[tag]["ratio"])
        summary.append(
            {
                "capacity_tag": tag,
                "capacity_label": CAP_LABEL[tag],
                "n": n,
                "signal_mean": sm,
                "signal_ci95": sci,
                "inventory_mean": im,
                "inventory_ci95": ici,
                "ratio_mean": rm,
                "ratio_ci95": rci,
                "signal_lt_inventory": bool(sm < 0.25 * im) if im == im else False,
            }
        )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w / 2, [s["signal_mean"] for s in summary], w, yerr=[s["signal_ci95"] for s in summary],
           capsize=3, label="signal (claimed demand)", color="#a33b2b")
    ax.bar(x + w / 2, [s["inventory_mean"] for s in summary], w, yerr=[s["inventory_ci95"] for s in summary],
           capsize=3, label="own inventory (yardstick)", color="#2f6f4e")
    ax.set_xticks(x)
    ax.set_xticklabels([s["capacity_label"] for s in summary])
    ax.set_ylabel("Mean |Δ order| per unit feature change")
    ax.set_title("D3 listener sensitivity: signal vs inventory")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d3_listener_sensitivity.png", dpi=160)
    plt.close(fig)

    # Ratio panel
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x, [s["ratio_mean"] for s in summary], yerr=[s["ratio_ci95"] for s in summary],
           color="#4a6fa5", capsize=3)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="parity")
    ax.axhline(0.25, color="#a33b2b", ls=":", lw=1, label="≪ threshold (0.25)")
    ax.set_xticks(x)
    ax.set_xticklabels([s["capacity_label"] for s in summary])
    ax.set_ylabel("signal / inventory sensitivity")
    ax.set_title("D3 sensitivity ratio")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d3_sensitivity_ratio.png", dpi=160)
    plt.close(fig)

    n_ignore = sum(1 for s in summary if s["signal_lt_inventory"])
    interpretation = (
        "signal ≪ inventory (corroborates D2 ignore)"
        if n_ignore >= max(1, len(summary) - 1)
        else "mixed / signals matter at policy level"
    )

    result = {
        "summary": summary,
        "per_run": per_run,
        "interpretation_key": interpretation,
        "definition": (
            "Sensitivity = mean |Δ decoded order| / Δfeature over a fixed obs batch, "
            "sweeping claimed-demand signal features vs own-inventory feature."
        ),
    }
    write_json(cache_path, result)
    write_json(CACHE_DIR / "d3_summary.json", {k: v for k, v in result.items() if k != "per_run"})
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-seeds", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(max_seeds=args.max_seeds, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
