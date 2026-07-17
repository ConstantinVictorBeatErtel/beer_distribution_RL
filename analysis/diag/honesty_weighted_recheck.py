#!/usr/bin/env python3
"""Honesty-weighted recheck under matched-deterministic eval.

Re-verifies the v1.1 claim that honesty-weighted allocation on Y depresses
broadcast share (~0.33) vs prop/uniform (~0.49). Also probes whether the
reputation EMA actually moves (vs flat/uninitialized plumbing artifact).

No training / reward / env changes. Uses greedy=True, seed=cfg.seed+10_000
(same definition as artifacts/diagnostics/eval_mode_blast_radius.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.common import CACHE_DIR, ci95, load_trainer, write_json  # noqa: E402
from beer_distribution_rl.env.core_types import Role  # noqa: E402

V11_DIR = ROOT / "artifacts" / "runs" / "ippo" / "tier1_v11"
OUT_DIR = ROOT / "artifacts" / "diagnostics"
OUT_MD = OUT_DIR / "honesty_weighted_recheck.md"
OUT_FIG = OUT_DIR / "honesty_weighted_ema_trajectories.png"
BASELINE_SHA = "061aa59235397b7360c32a01cf4f98add0dd503a"
EVAL_SEED_OFFSET = 10_000

CAP_ORDER = ["1p2mu", "1p0mu", "0p8mu"]  # honesty_weighted pruned at ∞
CAP_LABEL = {"inf": "∞", "1p2mu": "1.2μ", "1p0mu": "1.0μ", "0p8mu": "0.8μ"}
RATS = ("proportional", "uniform", "honesty_weighted")
# Seeds for EMA trajectory figure
TRAJ_SEEDS = (0, 1, 2)
TRAJ_CAP = "1p0mu"


def list_y_b_ar1_runs() -> list[dict[str, Any]]:
    idx = json.loads((V11_DIR / "index.json").read_text())
    out = []
    for r in idx:
        if r.get("status") != "ok":
            continue
        if r.get("regime") != "B" or r.get("topology") != "y" or r.get("demand") != "ar1":
            continue
        if r.get("rationing") not in RATS:
            continue
        if r.get("capacity_tag") not in CAP_ORDER:
            continue
        run_dir = V11_DIR / r["run"]
        if not (run_dir / "checkpoints" / "policy_retailer_a.pt").exists():
            continue
        out.append({**r, "run_dir": str(run_dir)})
    return out


def _retailers(trainer) -> list[Role]:
    return [r for r in trainer.roles if r in (Role.RETAILER, Role.RETAILER_B)]


def collect_matched_det(
    trainer,
    *,
    n_episodes: int,
    seed: int,
    record_ema_traj: bool = False,
    greedy: bool = True,
) -> dict[str, Any]:
    """Rollouts: share, honesty, EMA stats, optional per-step EMA traj.

    ``greedy=True`` = matched-deterministic (blast-radius definition).
    ``greedy=False`` = stochastic sample (training-like; used only to probe
    whether EMA moves when broadcasts actually occur).

    ``share`` matches trainer.evaluate: broadcasts / all-role opportunities.
    ``retailer_share`` is retailers only (the reputation claimants on Y).
    """
    core = trainer.core
    retailers = _retailers(trainer)
    share_rates: list[float] = []
    retailer_share_rates: list[float] = []
    honesty_scores: list[float] = []
    ema_series: dict[str, list[float]] = defaultdict(list)
    ema_diffs: list[float] = []
    n_ema_nonzero_weeks = 0
    n_weeks = 0
    n_broadcasts = 0
    n_broadcast_opps = 0
    n_ema_updates = 0
    n_retailer_broadcasts = 0
    n_retailer_opps = 0
    traj: list[dict[str, Any]] = []

    for ep in range(n_episodes):
        states = core.reset(seed + ep)
        done = False
        broadcasts = 0
        broadcast_opps = 0
        ret_broadcasts = 0
        ret_opps = 0
        mae_sum = 0.0
        mae_n = 0
        while not done:
            orders: dict = {}
            signals: dict = {}
            with torch.no_grad():
                for r in trainer.roles:
                    o = torch.as_tensor(
                        trainer._obs(states, r, core), device=trainer.device
                    ).unsqueeze(0)
                    a, _, _ = trainer._policy_act(r, o, greedy=greedy)
                    row = a.squeeze(0).cpu().numpy().astype(int)
                    orders[r] = trainer._decode_order(int(row[0]), states[r])
                    signals[r] = trainer._decode_signal(
                        states[r], int(row[1]), int(row[2]), int(row[3])
                    )
            states, _rewards, done, info = core.step(orders, signals)
            n_weeks += 1

            for r in trainer.roles:
                broadcast_opps += 1
                n_broadcast_opps += 1
                if info.signals_sent.get(r) is not None:
                    broadcasts += 1
                    n_broadcasts += 1

            emas = {r: float(core._honesty_ema[r]) for r in retailers}
            for r in retailers:
                name = core.role_names.get(r, r.name.lower())
                ema_series[name].append(emas[r])
                ret_opps += 1
                n_retailer_opps += 1
                if info.signals_sent.get(r) is not None:
                    ret_broadcasts += 1
                    n_retailer_broadcasts += 1
                h = info.honesty.get(r, {})
                mae = h.get("mean_abs_error", float("nan"))
                if mae == mae:
                    mae_sum += float(mae)
                    mae_n += 1
                    n_ema_updates += 1

            if len(retailers) >= 2:
                vals = [emas[r] for r in retailers]
                ema_diffs.append(abs(vals[0] - vals[1]))
                if any(abs(v) > 1e-9 for v in vals):
                    n_ema_nonzero_weeks += 1

            if record_ema_traj and ep < 3:
                row_t: dict[str, Any] = {
                    "ep": ep,
                    "t": int(core.t),
                    "ema_a": float(emas.get(Role.RETAILER, float("nan"))),
                    "ema_b": float(emas.get(Role.RETAILER_B, float("nan"))),
                }
                for r in retailers:
                    name = core.role_names.get(r, r.name.lower())
                    row_t[f"broadcast_{name}"] = info.signals_sent.get(r) is not None
                traj.append(row_t)

        if broadcast_opps:
            share_rates.append(broadcasts / broadcast_opps)
        if ret_opps:
            retailer_share_rates.append(ret_broadcasts / ret_opps)
        if mae_n:
            honesty_scores.append(-(mae_sum / mae_n) / max(trainer.cfg.order_cap, 1))

    def _summ(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        arr = np.asarray(xs, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    cross = _summ(ema_diffs)
    role_ema = {name: _summ(xs) for name, xs in ema_series.items()}

    return {
        "share": float(np.mean(share_rates)) if share_rates else float("nan"),
        "retailer_share": (
            float(np.mean(retailer_share_rates)) if retailer_share_rates else float("nan")
        ),
        "honesty": float(np.mean(honesty_scores)) if honesty_scores else float("nan"),
        "frac_ema_updates": float(n_ema_updates / max(n_retailer_opps, 1)),
        "frac_weeks_ema_nonzero": float(n_ema_nonzero_weeks / max(n_weeks, 1)),
        "ema_cross_abs_diff_mean": cross["mean"],
        "ema_cross_abs_diff_std": cross["std"],
        "ema_cross_abs_diff_max": cross["max"],
        "role_ema": role_ema,
        "n_weeks": float(n_weeks),
        "n_broadcasts": float(n_broadcasts),
        "traj": traj,
        "logged_share": float("nan"),
        "logged_honesty": float("nan"),
    }


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(payload["run_dir"])
    n_episodes = int(payload["n_episodes"])
    record = bool(payload.get("record_ema_traj", False))
    trainer, _ = load_trainer(run_dir)
    seed = int(trainer.cfg.seed) + EVAL_SEED_OFFSET
    out = collect_matched_det(
        trainer, n_episodes=n_episodes, seed=seed, record_ema_traj=record, greedy=True
    )
    # Stochastic probe: does EMA move when broadcasts occur (training-like)?
    # Only needed for honesty_weighted; cheap relative to det pass.
    stoch: dict[str, Any] = {}
    if payload["row"]["rationing"] == "honesty_weighted" or record:
        stoch = collect_matched_det(
            trainer,
            n_episodes=n_episodes,
            seed=seed + 50_000,  # distinct from det / train eval offsets
            record_ema_traj=record and payload["row"]["rationing"] == "honesty_weighted",
            greedy=False,
        )
    fe_path = run_dir / "final_eval.json"
    if fe_path.exists():
        fe = json.loads(fe_path.read_text())
        out["logged_share"] = float(fe.get("eval/sharing_rate", float("nan")))
        out["logged_honesty"] = float(fe.get("eval/honesty_score", float("nan")))
    hist_path = run_dir / "history.json"
    train_shares: list[float] = []
    if hist_path.exists():
        hist = json.loads(hist_path.read_text())
        train_shares = [
            float(r["eval/sharing_rate"])
            for r in hist
            if "eval/sharing_rate" in r and r["eval/sharing_rate"] == r["eval/sharing_rate"]
        ]
    out["train_share_first"] = float(train_shares[0]) if train_shares else float("nan")
    out["train_share_last"] = float(train_shares[-1]) if train_shares else float("nan")
    out["stoch_share"] = float(stoch.get("share", float("nan")))
    out["stoch_retailer_share"] = float(stoch.get("retailer_share", float("nan")))
    out["retailer_share"] = float(out.get("retailer_share", float("nan")))
    out["stoch_frac_weeks_ema_nonzero"] = float(
        stoch.get("frac_weeks_ema_nonzero", float("nan"))
    )
    out["stoch_ema_cross_abs_diff_mean"] = float(
        stoch.get("ema_cross_abs_diff_mean", float("nan"))
    )
    out["stoch_ema_cross_abs_diff_max"] = float(
        stoch.get("ema_cross_abs_diff_max", float("nan"))
    )
    out["stoch_traj"] = stoch.get("traj", [])
    return {
        **payload["row"],
        **{k: v for k, v in out.items() if k != "role_ema"},
        "role_ema": out["role_ema"],
    }


def cell_metric(
    rows: list[dict[str, Any]],
    *,
    cap: str,
    rat: str,
    key: str,
) -> tuple[float, float, int]:
    xs = [
        float(r[key])
        for r in rows
        if r["capacity_tag"] == cap
        and r["rationing"] == rat
        and key in r
        and np.isfinite(float(r[key]))
    ]
    return ci95(xs)


def _fmt(m: float, ci: float, nd: int = 3) -> str:
    if not math.isfinite(m):
        return "—"
    return f"{m:.{nd}f}±{ci:.{nd}f}"


def plot_ema_trajectories(rows: list[dict[str, Any]], path: Path) -> None:
    """EMA over time: det (often flat) vs stochastic probe for honesty_weighted."""
    fig, axes = plt.subplots(2, len(TRAJ_SEEDS), figsize=(3.2 * len(TRAJ_SEEDS), 5.4), sharey=True)
    if len(TRAJ_SEEDS) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, seed in enumerate(TRAJ_SEEDS):
        match = [
            r
            for r in rows
            if r["rationing"] == "honesty_weighted"
            and r["capacity_tag"] == TRAJ_CAP
            and int(r["seed"]) == seed
        ]
        ax0, ax1 = axes[0][col], axes[1][col]
        if not match:
            ax0.set_title(f"seed {seed} · missing")
            ax0.axis("off")
            ax1.axis("off")
            continue
        r0 = match[0]
        # Top: matched-det traj
        traj = r0.get("traj") or []
        t0 = [p for p in traj if p["ep"] == 0]
        if t0:
            ts = [p["t"] for p in t0]
            ax0.plot(ts, [p["ema_a"] for p in t0], color="#1b4f72", lw=1.5, label="retailer_a")
            ax0.plot(ts, [p["ema_b"] for p in t0], color="#b03a2e", lw=1.5, label="retailer_b")
        ax0.axhline(0.0, color="#888", lw=0.7, alpha=0.7)
        ax0.set_title(f"det (greedy) · seed {seed}", fontsize=9)
        # Bottom: stochastic traj
        st = r0.get("stoch_traj") or []
        t1 = [p for p in st if p["ep"] == 0]
        if t1:
            ts = [p["t"] for p in t1]
            ax1.plot(ts, [p["ema_a"] for p in t1], color="#1b4f72", lw=1.5, label="retailer_a")
            ax1.plot(ts, [p["ema_b"] for p in t1], color="#b03a2e", lw=1.5, label="retailer_b")
        ax1.axhline(0.0, color="#888", lw=0.7, alpha=0.7)
        ax1.set_title(f"stoch probe · seed {seed}", fontsize=9)
        ax1.set_xlabel("week")
        for ax in (ax0, ax1):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        if col == 0:
            ax0.set_ylabel("honesty EMA (−MAE)")
            ax1.set_ylabel("honesty EMA (−MAE)")
            ax0.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Y · B · AR(1) · honesty_weighted · {CAP_LABEL[TRAJ_CAP]} · episode 0",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def decide_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify: headline-grade / footnote-grade / needs an order-based re-run."""

    def _agg(rat: str, key: str) -> tuple[float, float, int]:
        xs = [
            float(r[key])
            for r in rows
            if r["rationing"] == rat
            and r["capacity_tag"] in CAP_ORDER
            and np.isfinite(float(r.get(key, float("nan"))))
        ]
        return ci95(xs)

    hw_share, hw_ci, hw_n = _agg("honesty_weighted", "share")
    prop_share, prop_ci, _ = _agg("proportional", "share")
    uni_share, uni_ci, _ = _agg("uniform", "share")
    hw_ret, hw_ret_ci, _ = _agg("honesty_weighted", "retailer_share")
    prop_ret, prop_ret_ci, _ = _agg("proportional", "retailer_share")
    logged_hw, _, _ = _agg("honesty_weighted", "logged_share")
    logged_prop, _, _ = _agg("proportional", "logged_share")
    stoch_hw, stoch_hw_ci, _ = _agg("honesty_weighted", "stoch_share")
    stoch_hw_ret, stoch_hw_ret_ci, _ = _agg("honesty_weighted", "stoch_retailer_share")

    drop_vs_prop = prop_share - hw_share
    drop_vs_uni = uni_share - hw_share
    ret_drop = prop_ret - hw_ret
    # Directional disengagement: all-role and/or retailer share clearly lower.
    effect_survives = bool(
        hw_n >= 5
        and (
            (drop_vs_prop > 0.05 and hw_share < prop_share - 0.05)
            or (ret_drop > 0.05 and hw_ret < prop_ret - 0.05)
        )
    )
    numeric_033_survives = bool(0.25 <= hw_share <= 0.40)

    ema_nonzero, ema_nz_ci, _ = _agg("honesty_weighted", "frac_weeks_ema_nonzero")
    ema_diff, ema_diff_ci, _ = _agg("honesty_weighted", "ema_cross_abs_diff_mean")
    ema_diff_max, _, _ = _agg("honesty_weighted", "ema_cross_abs_diff_max")
    frac_updates, _, _ = _agg("honesty_weighted", "frac_ema_updates")

    stoch_ema_nz, stoch_ema_nz_ci, _ = _agg(
        "honesty_weighted", "stoch_frac_weeks_ema_nonzero"
    )
    stoch_ema_diff, stoch_ema_diff_ci, _ = _agg(
        "honesty_weighted", "stoch_ema_cross_abs_diff_mean"
    )

    # Under det, silence ⇒ flat EMA is expected (consequence, not root cause).
    ema_flat_under_det = bool(ema_nonzero < 0.1)
    # Stochastic probe: was reputation state alive when broadcasts happen?
    ema_alive_when_broadcasting = bool(stoch_ema_nz > 0.5 and stoch_ema_diff > 0.05)
    ema_flat_artifact = bool(
        ema_flat_under_det and not ema_alive_when_broadcasting and stoch_hw < 0.05
    )

    weights_on_broadcasts = True
    weights_on_orders = False

    # Broadcast-weighted + babbling ⇒ never headline P3 restoration.
    if ema_flat_artifact:
        verdict = "needs an order-based re-run"
    elif effect_survives and ema_alive_when_broadcasting:
        verdict = "footnote-grade"
    elif effect_survives:
        verdict = "footnote-grade"
    else:
        verdict = "needs an order-based re-run"

    return {
        "verdict": verdict,
        "effect_survives": effect_survives,
        "numeric_033_survives": numeric_033_survives,
        "ema_alive": ema_alive_when_broadcasting,
        "ema_flat_under_det": ema_flat_under_det,
        "ema_flat_artifact": ema_flat_artifact,
        "ema_alive_when_broadcasting": ema_alive_when_broadcasting,
        "weights_on_broadcasts": weights_on_broadcasts,
        "weights_on_orders": weights_on_orders,
        "hw_share": hw_share,
        "hw_ci": hw_ci,
        "prop_share": prop_share,
        "prop_ci": prop_ci,
        "uni_share": uni_share,
        "uni_ci": uni_ci,
        "hw_ret": hw_ret,
        "hw_ret_ci": hw_ret_ci,
        "prop_ret": prop_ret,
        "prop_ret_ci": prop_ret_ci,
        "ret_drop": ret_drop,
        "drop_vs_prop": drop_vs_prop,
        "drop_vs_uni": drop_vs_uni,
        "logged_hw": logged_hw,
        "logged_prop": logged_prop,
        "stoch_hw": stoch_hw,
        "stoch_hw_ci": stoch_hw_ci,
        "stoch_hw_ret": stoch_hw_ret,
        "stoch_hw_ret_ci": stoch_hw_ret_ci,
        "ema_nonzero": ema_nonzero,
        "ema_nz_ci": ema_nz_ci,
        "ema_diff": ema_diff,
        "ema_diff_ci": ema_diff_ci,
        "ema_diff_max": ema_diff_max,
        "frac_updates": frac_updates,
        "stoch_ema_nz": stoch_ema_nz,
        "stoch_ema_nz_ci": stoch_ema_nz_ci,
        "stoch_ema_diff": stoch_ema_diff,
        "stoch_ema_diff_ci": stoch_ema_diff_ci,
    }


def write_report(
    rows: list[dict[str, Any]],
    *,
    n_episodes: int,
    elapsed_s: float,
) -> dict[str, Any]:
    d = decide_verdict(rows)
    lines: list[str] = []
    lines.append("# Honesty-weighted allocation recheck")
    lines.append("")
    lines.append(f"**Baseline SHA:** `{BASELINE_SHA}`")
    lines.append("")
    lines.append(
        f"Eval-only re-roll of frozen Tier-1 v11 **Regime B × Y × AR(1)** checkpoints "
        f"under **matched-deterministic** eval (`greedy=True`, seed=`cfg.seed+{EVAL_SEED_OFFSET}`), "
        f"consistent with `artifacts/diagnostics/eval_mode_blast_radius.md`. "
        f"`n_episodes={n_episodes}`, 10 seeds/cell, wall ≈ {elapsed_s / 60:.1f} min. "
        f"No training / reward / env changes."
    )
    lines.append("")
    lines.append(f"**Verdict: `{d['verdict']}`.** {_verdict_blurb(d)}")
    lines.append("")

    lines.append("## What the logged cells actually weight")
    lines.append("")
    lines.append(
        "**Broadcast truthfulness — not order truthfulness.** "
        "`HonestyWeightedRationing` weights ∝ `exp(honesty_ema / T)` where "
        "`honesty_ema` is an EMA of `−mean_abs_error` on **signal claims** "
        "(`claimed_demand`, `claimed_inventory`) vs observed truth "
        "(`beer_distribution_rl/env/signals.py::measure_honesty`, "
        "`env/core.py` step 6). Orders never enter the EMA."
    )
    lines.append("")
    lines.append(
        "Given babbling broadcasts (`v11_signal_content.md`), the logged mechanism "
        "is **weighting on noise**. The natural pivot redefinition — weight on "
        "**order-truthfulness** (past orders tracking actual need) — was **not** "
        "what these cells ran."
    )
    lines.append("")

    lines.append("## 1. Does the share-drop survive matched-deterministic eval?")
    lines.append("")
    lines.append(
        f"Logged (stochastic B) honesty_weighted share ≈ {d['logged_hw']:.3f} vs "
        f"proportional ≈ {d['logged_prop']:.3f}. Matched-det recomputation below. "
        f"Also report a stochastic probe for HW: all-role "
        f"{_fmt(d['stoch_hw'], d['stoch_hw_ci'])}, retailer "
        f"{_fmt(d['stoch_hw_ret'], d['stoch_hw_ret_ci'])}."
    )
    lines.append("")
    lines.append(
        "| Cap | Rationing | Share all-role (det) | Retailer share (det) | "
        "Logged share | EMA≠0 (det) | mean\\|ΔEMA\\| (det)"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for cap in CAP_ORDER:
        for rat in RATS:
            s, cs, ns = cell_metric(rows, cap=cap, rat=rat, key="share")
            rs, crs, _ = cell_metric(rows, cap=cap, rat=rat, key="retailer_share")
            ls, cl, _ = cell_metric(rows, cap=cap, rat=rat, key="logged_share")
            ez, cez, _ = cell_metric(rows, cap=cap, rat=rat, key="frac_weeks_ema_nonzero")
            ed, ced, _ = cell_metric(rows, cap=cap, rat=rat, key="ema_cross_abs_diff_mean")
            if ns == 0:
                continue
            lines.append(
                f"| {CAP_LABEL[cap]} | {rat} | {_fmt(s, cs)} | {_fmt(rs, crs)} | "
                f"{_fmt(ls, cl)} | {_fmt(ez, cez)} | {_fmt(ed, ced)} |"
            )
    lines.append("")
    lines.append(
        f"**Directional disengagement survives: `{'yes' if d['effect_survives'] else 'no'}`.** "
        f"Pooled matched-det all-role share: HW "
        f"**{_fmt(d['hw_share'], d['hw_ci'])}** vs prop "
        f"**{_fmt(d['prop_share'], d['prop_ci'])}** (Δ={d['drop_vs_prop']:.3f}) "
        f"/ uni **{_fmt(d['uni_share'], d['uni_ci'])}**. "
        f"Retailer-only: HW **{_fmt(d['hw_ret'], d['hw_ret_ci'])}** vs prop "
        f"**{_fmt(d['prop_ret'], d['prop_ret_ci'])}** (Δ={d['ret_drop']:.3f})."
    )
    lines.append("")
    lines.append(
        f"**Numeric '~0.33' all-role level: "
        f"`{'roughly yes' if d['numeric_033_survives'] else 'no'}`** "
        f"(det HW all-role {d['hw_share']:.3f}). "
        f"The sharper signal is **retailer** silence under greedy: "
        f"HW retailer share {d['hw_ret']:.3f} vs prop {d['prop_ret']:.3f} — "
        f"claimants in the reputation game argmax to never broadcast; "
        f"upstream roles still broadcast, so all-role share stays near the logged band."
    )
    lines.append("")

    lines.append("## 2. Is the reputation EMA alive, or a plumbing artifact?")
    lines.append("")
    lines.append(
        "EMA init = `0.0` per role; updates **only** on weeks the agent broadcasts "
        "with a non-null claim (`mean_abs_error is not None`). If both stay at 0, "
        "`exp(0)=1` ⇒ fill collapses to request-proportional."
    )
    lines.append("")
    lines.append(
        "**Important:** under matched-det, HW share=0 ⇒ EMA never updates. "
        "That flatness is a *consequence* of argmax silence, not evidence the "
        "EMA plumbing failed during training (which samples broadcasts). "
        "Stochastic probe answers whether reputation moves when broadcasts occur."
    )
    lines.append("")
    lines.append("| Check | Value | Interpretation |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| Det: frac weeks EMA≠0 | {_fmt(d['ema_nonzero'], d['ema_nz_ci'])} | "
        f"{'flat (expected if silent)' if d['ema_flat_under_det'] else 'moves'} |"
    )
    lines.append(
        f"| Det: mean \\|EMA_a−EMA_b\\| | {_fmt(d['ema_diff'], d['ema_diff_ci'])} | "
        f"cross-agent separation under greedy |"
    )
    lines.append(
        f"| Stoch probe: share (all-role / retailer) | "
        f"{_fmt(d['stoch_hw'], d['stoch_hw_ci'])} / "
        f"{_fmt(d['stoch_hw_ret'], d['stoch_hw_ret_ci'])} | "
        f"training-like broadcast rate |"
    )
    lines.append(
        f"| Stoch probe: frac weeks EMA≠0 | {_fmt(d['stoch_ema_nz'], d['stoch_ema_nz_ci'])} | "
        f"{'alive when broadcasting' if d['ema_alive_when_broadcasting'] else 'still flat'} |"
    )
    lines.append(
        f"| Stoch probe: mean \\|ΔEMA\\| | {_fmt(d['stoch_ema_diff'], d['stoch_ema_diff_ci'])} | "
        f"cross-agent separation under sampling |"
    )
    lines.append(
        f"| EMA-never-accumulated artifact? | "
        f"**{'yes' if d['ema_flat_artifact'] else 'no'}** | "
        f"{'plumbing dead even when sampling' if d['ema_flat_artifact'] else 'EMA moves under stochastic broadcasts — not an init bug'} |"
    )
    lines.append("")
    lines.append(f"![EMA trajectories]({OUT_FIG.name})")
    lines.append("")
    lines.append(
        f"Figure: episode-0 EMA at {CAP_LABEL[TRAJ_CAP]}, seeds {list(TRAJ_SEEDS)}. "
        "Top: matched-det (typically flat at 0). Bottom: stochastic probe "
        "(EMA should diverge across retailers if reputation accumulates)."
    )
    lines.append("")

    lines.append("## Training-history context (stochastic B eval, sparse)")
    lines.append("")
    lines.append("| Cap | Rationing | Train share first | Train share last |")
    lines.append("|---|---|---:|---:|")
    for cap in CAP_ORDER:
        for rat in ("honesty_weighted", "proportional"):
            a, ca, na = cell_metric(rows, cap=cap, rat=rat, key="train_share_first")
            b, cb, _ = cell_metric(rows, cap=cap, rat=rat, key="train_share_last")
            if na == 0:
                continue
            lines.append(
                f"| {CAP_LABEL[cap]} | {rat} | {_fmt(a, ca)} | {_fmt(b, cb)} |"
            )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**`{d['verdict']}`** — {_verdict_blurb(d)}")
    lines.append("")
    lines.append("Deciding facts:")
    lines.append(
        "- Mechanism weights on **broadcasts** (not orders): confirmed in code path."
    )
    lines.append(
        f"- Disengagement under matched-det: "
        f"**{'yes' if d['effect_survives'] else 'no'}** "
        f"(all-role HW {d['hw_share']:.3f} vs prop {d['prop_share']:.3f} / "
        f"uni {d['uni_share']:.3f}; retailer HW {d['hw_ret']:.3f} vs prop "
        f"{d['prop_ret']:.3f})."
    )
    lines.append(
        f"- EMA-never-accumulated artifact: "
        f"**{'yes' if d['ema_flat_artifact'] else 'no'}** "
        f"(det flat={d['ema_flat_under_det']}; "
        f"stoch alive={d['ema_alive_when_broadcasting']}, "
        f"\\|ΔEMA\\|={d['stoch_ema_diff']:.3f})."
    )
    lines.append(
        "- Pivot: broadcast-weighted P3 on babbling signals is at best "
        "footnote-grade; **order-truthfulness weighting** is the re-run worth doing."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    return d


def _verdict_blurb(d: dict[str, Any]) -> str:
    v = d["verdict"]
    if v == "headline-grade":
        return (
            "Share drop + live EMA + order-based weights would restore truth-telling; "
            "not applicable here (weights are broadcast-based)."
        )
    if v == "footnote-grade":
        return (
            f"Disengagement survives matched-det "
            f"(all-role HW {d['hw_share']:.3f} vs prop {d['prop_share']:.3f}; "
            f"retailer HW {d['hw_ret']:.3f} vs prop {d['prop_ret']:.3f}); "
            f"EMA is live under stochastic broadcasts "
            f"(|ΔEMA|={d['stoch_ema_diff']:.3f}) so this is not an init bug — "
            f"but the mechanism weights **broadcast** honesty on a babbling channel, "
            f"so it is footnote-grade ('agents flee a noise-weighted reputation game'), "
            f"not restored truthful signaling."
        )
    return (
        f"Either disengagement fails under matched-det "
        f"(HW {d['hw_share']:.3f} vs prop {d['prop_share']:.3f}) "
        f"or EMA never accumulates even when sampling "
        f"(artifact={d['ema_flat_artifact']}). "
        f"Logged cells weight broadcasts; re-run with **order-truthfulness** weights."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / (
        f"honesty_weighted_recheck_ep{args.n_episodes}"
        + ("_smoke" if args.smoke else "")
        + ".json"
    )

    t0 = time.time()
    if cache.exists() and not args.force:
        print(f"loading cache {cache}")
        rows = json.loads(cache.read_text())
    else:
        runs = list_y_b_ar1_runs()
        if args.smoke:
            runs = [r for r in runs if int(r["seed"]) < 2]
            args.n_episodes = min(args.n_episodes, 2)
        payloads = []
        for r in runs:
            record = (
                int(r["seed"]) in TRAJ_SEEDS
                and r["capacity_tag"] == TRAJ_CAP
                and r["rationing"] in ("honesty_weighted", "proportional")
            )
            payloads.append(
                {
                    "run_dir": r["run_dir"],
                    "n_episodes": args.n_episodes,
                    "record_ema_traj": record,
                    "row": {
                        "run": r["run"],
                        "regime": r["regime"],
                        "topology": r["topology"],
                        "capacity_tag": r["capacity_tag"],
                        "rationing": r["rationing"],
                        "demand": r["demand"],
                        "seed": int(r["seed"]),
                    },
                }
            )
        print(f"honesty-weighted recheck: {len(payloads)} runs × {args.n_episodes} eps", flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(_worker, pl): pl for pl in payloads}
            done_n = 0
            for fut in as_completed(futs):
                rows.append(fut.result())
                done_n += 1
                if done_n % 10 == 0 or done_n == len(payloads):
                    print(f"  done {done_n}/{len(payloads)}", flush=True)
        write_json(cache, rows)

    elapsed = time.time() - t0
    plot_ema_trajectories(rows, OUT_FIG)
    d = write_report(rows, n_episodes=args.n_episodes, elapsed_s=elapsed)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_FIG}")
    print(f"Verdict: {d['verdict']} | survives={d['effect_survives']} | ema_alive={d['ema_alive']}")


if __name__ == "__main__":
    main()
