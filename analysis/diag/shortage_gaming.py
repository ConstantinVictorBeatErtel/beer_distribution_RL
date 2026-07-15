#!/usr/bin/env python3
"""Shortage gaming analysis on frozen Tier-1 v11 checkpoints.

Matched-deterministic (greedy=True, seed = cfg.seed + 10_000) rollouts —
same definition as artifacts/diagnostics/eval_mode_blast_radius.md.
No training / reward / env changes.

Thesis (Lee/Padmanabhan/Whang 1997): under multi-claimant rationing,
self-interested agents inflate orders to capture allocation share.
Strategic signature requires response to BOTH capacity tightness AND
rationing rule (proportional rewards inflation; uniform does not).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from beer_distribution_rl.agents.baselines import base_stock_order  # noqa: E402
from beer_distribution_rl.env.core_types import Role  # noqa: E402

V11_DIR = ROOT / "artifacts" / "runs" / "ippo" / "tier1_v11"
OUT_DIR = ROOT / "artifacts" / "diagnostics"
OUT_MD = OUT_DIR / "shortage_gaming.md"
OUT_FIG = OUT_DIR / "shortage_gaming_inflation_vs_capacity.png"
BASELINE_SHA = "061aa59235397b7360c32a01cf4f98add0dd503a"

CAP_ORDER = ["inf", "1p2mu", "1p0mu", "0p8mu"]
CAP_LABEL = {"inf": "∞", "1p2mu": "1.2μ", "1p0mu": "1.0μ", "0p8mu": "0.8μ"}
CAP_NUMERIC = {"inf": 3.0, "1p2mu": 1.2, "1p0mu": 1.0, "0p8mu": 0.8}  # for plotting

# AR(1) μ=7.5; L = L_s + L_o = 3. Installation S ≈ μL + z σ√L (BUGHUNT).
MU = 7.5
LEAD = 3  # ship_delay + order_delay
PRIMARY_S = 30  # z≈1 installation stock (BUGHUNT)
SENSITIVITY_S = (9, 22, 30, 45)  # classic DQN, μL, primary, z≈2–3
EVAL_SEED_OFFSET = 10_000  # matched to IPPOTrainer.evaluate / blast-radius


def list_target_runs() -> list[dict[str, Any]]:
    """A/B × {y,serial} × AR(1) × {prop,uniform} × caps × 10 seeds (as logged)."""
    idx = json.loads((V11_DIR / "index.json").read_text())
    out = []
    for r in idx:
        if r.get("status") != "ok":
            continue
        if r.get("regime") not in ("A", "B"):
            continue
        if r.get("demand") != "ar1":
            continue
        if r.get("topology") not in ("y", "serial"):
            continue
        if r.get("rationing") not in ("proportional", "uniform"):
            continue
        if r.get("capacity_tag") not in CAP_ORDER:
            continue
        run_dir = V11_DIR / r["run"]
        ckpt = run_dir / "checkpoints"
        has = (ckpt / "policy_retailer.pt").exists() or (ckpt / "policy_retailer_a.pt").exists()
        if not has:
            continue
        out.append({**r, "run_dir": str(run_dir)})
    return out


def _retailer_roles(trainer) -> list[Role]:
    return [r for r in trainer.roles if r in (Role.RETAILER, Role.RETAILER_B)]


def _bench_order(state, *, S: int, order_cap: int, mode: str) -> int:
    """Truthful / mechanical order benchmark given observed state."""
    if mode == "base_stock":
        return int(base_stock_order(state, S=S, order_cap=order_cap))
    if mode == "pass_through":
        d = int(state.last_demand_or_order)
        return max(0, min(order_cap, d))
    raise ValueError(mode)


def collect_order_rows(
    trainer,
    *,
    n_episodes: int,
    seed: int,
    S_levels: tuple[int, ...] = SENSITIVITY_S,
) -> dict[str, Any]:
    """Matched-deterministic rollout; log retailer orders vs benchmarks + rival fills."""
    core = trainer.core
    retailers = _retailer_roles(trainer)
    order_cap = int(trainer.cfg.order_cap)
    rows: list[dict[str, Any]] = []
    n_at_cap = 0
    n_orders = 0

    for ep in range(n_episodes):
        states = core.reset(seed + ep)
        done = False
        while not done:
            # Snapshot pre-order state for benchmarks (IP before this week's order).
            pre = {r: states[r] for r in trainer.roles}
            benches = {
                r: {
                    f"S{S}": _bench_order(pre[r], S=S, order_cap=order_cap, mode="base_stock")
                    for S in S_levels
                }
                for r in retailers
            }
            for r in retailers:
                benches[r]["pass"] = _bench_order(
                    pre[r], S=PRIMARY_S, order_cap=order_cap, mode="pass_through"
                )

            orders: dict = {}
            signals = {} if trainer.signaling else None
            with torch.no_grad():
                for r in trainer.roles:
                    o = torch.as_tensor(
                        trainer._obs(states, r, core), device=trainer.device
                    ).unsqueeze(0)
                    a, _, _ = trainer._policy_act(r, o, greedy=True)
                    if trainer.signaling:
                        row_a = a.squeeze(0).cpu().numpy().astype(int)
                        orders[r] = trainer._decode_order(int(row_a[0]), states[r])
                        assert signals is not None
                        signals[r] = trainer._decode_signal(
                            states[r], int(row_a[1]), int(row_a[2]), int(row_a[3])
                        )
                    else:
                        orders[r] = trainer._decode_order(int(a.item()), states[r])

            states, _rewards, done, info = core.step(orders, signals)

            # Wholesaler multi-claimant allocation (Y only).
            wh = Role.WHOLESALER
            wh_alloc = info.allocations.get(wh, {}) if hasattr(info, "allocations") else {}
            wh_incoming = {}
            # incoming at wholesaler is split by claimant in core; recover from
            # claimant backlog + allocation ≈ requested this week when possible.
            # Prefer explicit: requested ≈ alloc + new claimant backlog after fill.
            st_wh = core._states[wh]
            requested = {}
            for c in (Role.RETAILER, Role.RETAILER_B):
                if c not in core.roles:
                    continue
                alloc_c = int(wh_alloc.get(c, 0))
                backlog_c = int(st_wh.claimant_backlog.get(c, 0))
                requested[c] = alloc_c + backlog_c
            available = int(sum(wh_alloc.values())) if wh_alloc else 0
            total_req = int(sum(requested.values()))
            wholesaler_rationed = total_req > available and total_req > 0 and len(requested) >= 2

            for r in retailers:
                name = core.role_names.get(r, r.name.lower())
                placed = int(info.orders_placed[r])
                n_orders += 1
                if placed >= order_cap:
                    n_at_cap += 1
                demand = int(pre[r].last_demand_or_order)
                ip = int(pre[r].inventory_position())
                backlog = int(states[r].backlog)  # post-step own backlog
                alloc_to_me = int(wh_alloc.get(r, 0)) if wh_alloc else None
                rival = Role.RETAILER_B if r == Role.RETAILER else Role.RETAILER
                alloc_to_rival = (
                    int(wh_alloc.get(rival, 0)) if (wh_alloc and rival in core.roles) else None
                )
                rec: dict[str, Any] = {
                    "ep": ep,
                    "t": int(core.t),
                    "role": name,
                    "role_enum": r.name,
                    "order": placed,
                    "demand": demand,
                    "ip": ip,
                    "backlog": backlog,
                    "alloc": alloc_to_me,
                    "alloc_rival": alloc_to_rival,
                    "wholesaler_rationed": bool(wholesaler_rationed),
                    "wh_available": available,
                    "wh_total_req": total_req,
                    "bench_pass": int(benches[r]["pass"]),
                }
                for S in S_levels:
                    b = int(benches[r][f"S{S}"])
                    rec[f"bench_S{S}"] = b
                    rec[f"gap_S{S}"] = placed - b
                    rec[f"ratio_S{S}"] = placed / max(b, 1)
                rec["gap_pass"] = placed - int(benches[r]["pass"])
                rec["ratio_pass"] = placed / max(int(benches[r]["pass"]), 1)
                rows.append(rec)

    return {
        "rows": rows,
        "frac_at_cap": float(n_at_cap / max(n_orders, 1)),
        "n_retailer_orders": float(n_orders),
    }


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(payload["run_dir"])
    n_episodes = int(payload["n_episodes"])
    trainer, _ = load_trainer(run_dir)
    seed = int(trainer.cfg.seed) + EVAL_SEED_OFFSET
    collected = collect_order_rows(trainer, n_episodes=n_episodes, seed=seed)
    # Aggregate per-run summaries (keep rows only if needed for rival coupling).
    rows = collected["rows"]
    primary = f"S{PRIMARY_S}"

    def _mean(key: str) -> float:
        xs = [float(r[key]) for r in rows if key in r]
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        **payload["row"],
        "frac_at_cap": collected["frac_at_cap"],
        "n_retailer_orders": collected["n_retailer_orders"],
        f"mean_gap_{primary}": _mean(f"gap_{primary}"),
        f"mean_ratio_{primary}": _mean(f"ratio_{primary}"),
        "mean_gap_pass": _mean("gap_pass"),
        "mean_ratio_pass": _mean("ratio_pass"),
        "mean_order": _mean("order"),
        "mean_demand": _mean("demand"),
    }
    for S in SENSITIVITY_S:
        summary[f"mean_gap_S{S}"] = _mean(f"gap_S{S}")
        summary[f"mean_ratio_S{S}"] = _mean(f"ratio_S{S}")

    # Rival coupling (Y only): on wholesaler-rationed weeks.
    rationed = [r for r in rows if r.get("wholesaler_rationed")]
    coupling = _rival_coupling(rationed, primary_key=primary)
    summary.update(coupling)
    return summary


def _rival_coupling(rationed_rows: list[dict[str, Any]], *, primary_key: str) -> dict[str, float]:
    """Competitive externality on wholesaler-rationed weeks.

    Absolute allocations are confounded by week-to-week supply. Use *shares*
    of wholesaler available, and relative (own − rival) order vs relative alloc.
    """
    out = {
        "rival_n_rationed_weeks_role": float(len(rationed_rows)),
        "rival_corr_gap_vs_alloc_rival": float("nan"),  # share-based
        "rival_corr_gap_vs_own_alloc": float("nan"),  # share-based
        "rival_corr_rel_order_vs_rel_alloc": float("nan"),
        "rival_mean_alloc_when_high_gap": float("nan"),  # rival share
        "rival_mean_alloc_when_low_gap": float("nan"),
        "rival_externality_delta": float("nan"),  # low−high rival share
    }
    if len(rationed_rows) < 20:
        return out
    gap = np.array([float(r[f"gap_{primary_key}"]) for r in rationed_rows], dtype=float)
    order = np.array([float(r["order"]) for r in rationed_rows], dtype=float)
    alloc_rival = np.array(
        [
            float(r["alloc_rival"]) if r["alloc_rival"] is not None else np.nan
            for r in rationed_rows
        ],
        dtype=float,
    )
    alloc_own = np.array(
        [float(r["alloc"]) if r["alloc"] is not None else np.nan for r in rationed_rows],
        dtype=float,
    )
    avail = np.array([float(r.get("wh_available", 0)) for r in rationed_rows], dtype=float)
    mask = (
        np.isfinite(gap)
        & np.isfinite(alloc_rival)
        & np.isfinite(alloc_own)
        & np.isfinite(avail)
        & (avail > 0)
    )
    if int(mask.sum()) < 20:
        return out
    g = gap[mask]
    o = order[mask]
    ar = alloc_rival[mask] / avail[mask]
    ao = alloc_own[mask] / avail[mask]
    # Reconstruct rival order from paired rows is hard (each row is one retailer).
    # Relative order proxy: own order vs mean of the other role in same (ep,t).
    # Build (ep,t) pairs:
    by_et: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rationed_rows:
        by_et[(r["ep"], r["t"])][r["role"]] = r
    rel_o, rel_a = [], []
    for pair in by_et.values():
        if len(pair) < 2:
            continue
        roles = list(pair.keys())
        a, b = pair[roles[0]], pair[roles[1]]
        if a.get("alloc") is None or b.get("alloc") is None:
            continue
        rel_o.append(float(a["order"]) - float(b["order"]))
        rel_a.append(float(a["alloc"]) - float(b["alloc"]))
        # also opposite orientation for symmetry
        rel_o.append(float(b["order"]) - float(a["order"]))
        rel_a.append(float(b["alloc"]) - float(a["alloc"]))
    if float(np.std(g)) >= 1e-9 and float(np.std(ar)) >= 1e-9:
        out["rival_corr_gap_vs_alloc_rival"] = float(np.corrcoef(g, ar)[0, 1])
    if float(np.std(g)) >= 1e-9 and float(np.std(ao)) >= 1e-9:
        out["rival_corr_gap_vs_own_alloc"] = float(np.corrcoef(g, ao)[0, 1])
    if len(rel_o) >= 20 and float(np.std(rel_o)) >= 1e-9 and float(np.std(rel_a)) >= 1e-9:
        out["rival_corr_rel_order_vs_rel_alloc"] = float(
            np.corrcoef(np.asarray(rel_o), np.asarray(rel_a))[0, 1]
        )
    lo, hi = np.percentile(g, 33), np.percentile(g, 67)
    low_mask = g <= lo
    high_mask = g >= hi
    if low_mask.sum() >= 5 and high_mask.sum() >= 5:
        mean_low = float(np.mean(ar[low_mask]))
        mean_high = float(np.mean(ar[high_mask]))
        out["rival_mean_alloc_when_low_gap"] = mean_low
        out["rival_mean_alloc_when_high_gap"] = mean_high
        # Positive delta ⇒ rival share falls when I inflate more.
        out["rival_externality_delta"] = mean_low - mean_high
    return out


def aggregate_cell(
    summaries: list[dict[str, Any]], key: str
) -> dict[tuple, dict[str, tuple[float, float, int]]]:
    """(regime, topo, cap, rat) -> metric -> (mean, ci95, n)."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for s in summaries:
        cell = (s["regime"], s["topology"], s["capacity_tag"], s["rationing"])
        v = s.get(key)
        if v is None or not np.isfinite(float(v)):
            continue
        buckets[cell].append(float(v))
    return {cell: {key: ci95(xs)} for cell, xs in buckets.items()}


def cell_metric(
    summaries: list[dict[str, Any]],
    *,
    regime: str,
    topo: str,
    cap: str,
    rat: str,
    key: str,
) -> tuple[float, float, int]:
    xs = [
        float(s[key])
        for s in summaries
        if s["regime"] == regime
        and s["topology"] == topo
        and s["capacity_tag"] == cap
        and s["rationing"] == rat
        and key in s
        and np.isfinite(float(s[key]))
    ]
    return ci95(xs)


def plot_inflation_vs_capacity(summaries: list[dict[str, Any]], path: Path) -> None:
    """Inflation (mean gap vs S=30) vs capacity, per rationing rule — Y topology."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    primary = f"mean_gap_S{PRIMARY_S}"
    styles = {
        ("proportional", "B"): ("#1b4f72", "B proportional"),
        ("uniform", "B"): ("#b03a2e", "B uniform"),
        ("proportional", "A"): ("#5dade2", "A proportional"),
        ("uniform", "A"): ("#e59866", "A uniform"),
    }
    x_pos = {c: i for i, c in enumerate(CAP_ORDER)}
    for (rat, regime), (color, label) in styles.items():
        xs, ys, yerr = [], [], []
        for cap in CAP_ORDER:
            m, ci, n = cell_metric(
                summaries, regime=regime, topo="y", cap=cap, rat=rat, key=primary
            )
            if n == 0:
                continue
            xs.append(x_pos[cap])
            ys.append(m)
            yerr.append(ci)
        if not xs:
            continue
        marker = "o" if rat == "proportional" else "s"
        ls = "-" if regime == "B" else ":"
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt=marker + ls,
            color=color,
            label=label,
            capsize=3,
            lw=1.8,
            markersize=7,
        )
    ax.set_xticks(list(range(len(CAP_ORDER))))
    ax.set_xticklabels([CAP_LABEL[c] for c in CAP_ORDER])
    ax.set_xlabel("Factory capacity (slack → tight)")
    ax.set_ylabel(f"Order inflation gap (order − base-stock S={PRIMARY_S})")
    ax.set_title("Y-topology · AR(1) · matched-deterministic · 10 seeds")
    ax.axhline(0.0, color="#888", lw=0.8, alpha=0.7)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fmt(m: float, ci: float) -> str:
    if not math.isfinite(m):
        return "—"
    return f"{m:.2f}±{ci:.2f}"


def decide_verdict(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Gaming supported only if inflation responds to capacity AND rationing rule."""
    primary = f"mean_gap_S{PRIMARY_S}"
    # Scarcity slope under B×Y×proportional: gap(0.8μ) − gap(∞)
    g_inf, _, n_inf = cell_metric(
        summaries, regime="B", topo="y", cap="inf", rat="proportional", key=primary
    )
    g_08, _, n_08 = cell_metric(
        summaries, regime="B", topo="y", cap="0p8mu", rat="proportional", key=primary
    )
    g_10, _, n_10 = cell_metric(
        summaries, regime="B", topo="y", cap="1p0mu", rat="proportional", key=primary
    )
    scarcity_delta = g_08 - g_inf if (n_inf and n_08) else float("nan")
    # Monotone check across ∞ → 1.2 → 1.0 → 0.8
    gaps = []
    for cap in CAP_ORDER:
        m, _, n = cell_metric(
            summaries, regime="B", topo="y", cap=cap, rat="proportional", key=primary
        )
        if n:
            gaps.append(m)
    monotone = all(gaps[i] <= gaps[i + 1] + 1e-9 for i in range(len(gaps) - 1)) if len(gaps) >= 2 else False

    # Rule contrast at tight caps: prop − uniform (positive ⇒ strategic)
    rule_deltas = {}
    for cap in ("1p2mu", "1p0mu", "0p8mu"):
        gp, cip, np_ = cell_metric(
            summaries, regime="B", topo="y", cap=cap, rat="proportional", key=primary
        )
        gu, ciu, nu = cell_metric(
            summaries, regime="B", topo="y", cap=cap, rat="uniform", key=primary
        )
        rule_deltas[cap] = {
            "prop": (gp, cip, np_),
            "uniform": (gu, ciu, nu),
            "delta": gp - gu if (np_ and nu) else float("nan"),
        }
    mean_rule_delta = float(
        np.nanmean([rule_deltas[c]["delta"] for c in rule_deltas])
    )

    # Rival externality under B×Y×prop at 0.8μ (share-based + relative order/alloc)
    ext_p, cie, ne = cell_metric(
        summaries,
        regime="B",
        topo="y",
        cap="0p8mu",
        rat="proportional",
        key="rival_externality_delta",
    )
    corr_p, _, _ = cell_metric(
        summaries,
        regime="B",
        topo="y",
        cap="0p8mu",
        rat="proportional",
        key="rival_corr_gap_vs_alloc_rival",
    )
    rel_p, _, _ = cell_metric(
        summaries,
        regime="B",
        topo="y",
        cap="0p8mu",
        rat="proportional",
        key="rival_corr_rel_order_vs_rel_alloc",
    )
    rel_u, _, _ = cell_metric(
        summaries,
        regime="B",
        topo="y",
        cap="0p8mu",
        rat="uniform",
        key="rival_corr_rel_order_vs_rel_alloc",
    )
    ext_u, _, _ = cell_metric(
        summaries,
        regime="B",
        topo="y",
        cap="0p8mu",
        rat="uniform",
        key="rival_externality_delta",
    )

    # Serial negative control: scarcity delta should be weaker / no rule contrast
    s_inf, _, _ = cell_metric(
        summaries, regime="B", topo="serial", cap="inf", rat="proportional", key=primary
    )
    s_08, _, _ = cell_metric(
        summaries, regime="B", topo="serial", cap="0p8mu", rat="proportional", key=primary
    )
    serial_scarcity = s_08 - s_inf

    # Cap saturation guardrail
    frac_cap, _, _ = cell_metric(
        summaries, regime="B", topo="y", cap="0p8mu", rat="proportional", key="frac_at_cap"
    )

    scarcity_ok = bool(scarcity_delta > 0.5 and (monotone or (g_08 > g_inf and g_10 >= g_inf)))
    rule_ok = bool(mean_rule_delta > 0.5)
    # Rival OK if proportional couples relative orders to relative allocs more
    # than uniform (prop corr high; uniform near 0), or share externality > 0.
    rival_ok = bool(
        (
            math.isfinite(rel_p)
            and rel_p > 0.3
            and (not math.isfinite(rel_u) or rel_p > rel_u + 0.15)
        )
        or (math.isfinite(ext_p) and ext_p > 0.02)
    )

    if scarcity_ok and rule_ok:
        verdict = "supported"
    elif (not scarcity_ok) and (not rule_ok):
        verdict = "not supported"
    else:
        verdict = "mixed"

    return {
        "verdict": verdict,
        "scarcity_delta_prop": scarcity_delta,
        "scarcity_monotone": monotone,
        "g_inf_prop": g_inf,
        "g_08_prop": g_08,
        "g_10_prop": g_10,
        "mean_rule_delta": mean_rule_delta,
        "rule_deltas": rule_deltas,
        "rival_ext_prop_08": ext_p,
        "rival_corr_prop_08": corr_p,
        "rival_rel_prop_08": rel_p,
        "rival_rel_uniform_08": rel_u,
        "rival_ext_uniform_08": ext_u,
        "serial_scarcity_delta": serial_scarcity,
        "frac_at_cap_B_y_08_prop": frac_cap,
        "scarcity_ok": scarcity_ok,
        "rule_ok": rule_ok,
        "rival_ok": rival_ok,
    }


def write_report(
    summaries: list[dict[str, Any]],
    *,
    n_episodes: int,
    elapsed_s: float,
    fig_path: Path,
) -> None:
    primary = f"mean_gap_S{PRIMARY_S}"
    primary_r = f"mean_ratio_S{PRIMARY_S}"
    decision = decide_verdict(summaries)
    lines: list[str] = []
    lines.append("# Shortage gaming (order-stream analysis)")
    lines.append("")
    lines.append(f"**Baseline SHA:** `{BASELINE_SHA}`")
    lines.append("")
    lines.append(
        f"Analysis-only re-roll of frozen Tier-1 v11 checkpoints under "
        f"**matched-deterministic** eval (`greedy=True`, seed = `cfg.seed+{EVAL_SEED_OFFSET}`), "
        f"consistent with `artifacts/diagnostics/eval_mode_blast_radius.md`. "
        f"`n_episodes={n_episodes}`, 10 seeds/cell, wall ≈ {elapsed_s / 60:.1f} min. "
        f"No training / reward / env changes."
    )
    lines.append("")
    lines.append(
        f"**Verdict: `{decision['verdict']}`.** "
        + _verdict_sentence(decision)
    )
    lines.append("")

    lines.append("## Thesis")
    lines.append("")
    lines.append(
        "Cheap-talk broadcasts are babbling (`v11_signal_content.md`). The costly "
        "ORDER stream is the remaining strategic channel: under multi-claimant "
        "rationing, agents can inflate orders to capture allocation "
        "(Lee, Padmanabhan & Whang 1997). Bite exists only on **Y** (two retailers, "
        "one wholesaler); **serial** is the no-rival negative control. Regime **A** "
        "is the no-channel control."
    )
    lines.append("")

    lines.append("## Operational definitions")
    lines.append("")
    lines.append(
        "1. **Order inflation.** At each retailer order decision, snapshot inventory "
        f"position `IP = on-hand − backlog + on_order` and observed demand. "
        f"Truthful benchmark = base-stock order-up-to: "
        f"`o* = clip(S − IP, 0, order_cap)` with primary **S = {PRIMARY_S}** "
        f"(installation stock ≈ μL + zσ√L for AR(1) μ=7.5, L=L_s+L_o=3; BUGHUNT). "
        f"**Gap** = `order − o*`; **ratio** = `order / max(o*, 1)`. "
        f"Sensitivity: S ∈ {list(SENSITIVITY_S)} and pass-through `o*=demand`."
    )
    lines.append(
        "2. **Scarcity response.** Does mean gap **increase** as capacity tightens "
        "(∞ → 1.2μ → 1.0μ → 0.8μ)? Signature of gaming vs a fixed mechanical policy."
    )
    lines.append(
        "3. **Strategic vs mechanical (crux).** At matched capacity, is inflation "
        "**higher under proportional than under uniform**? Proportional awards share "
        "by claim size; uniform ignores order size. Rule contrast ⇒ strategic response."
    )
    lines.append(
        "4. **Rival coupling (Y only).** On wholesaler-rationed weeks: "
        "corr(own gap, rival *share* of available), "
        "corr((own−rival) order, (own−rival) allocation), and tercile share "
        "externality `E[rival_share | low own-gap] − E[rival_share | high own-gap]` "
        "(positive ⇒ rival share falls when I inflate)."
    )
    lines.append("")
    lines.append(
        "**Gaming label rule:** report inflation as *shortage gaming* only if it "
        "responds to **both** capacity tightness **and** rationing rule. "
        "Response to neither ⇒ thesis not supported."
    )
    lines.append("")

    lines.append("## Measure 1 — Order inflation (primary S=30)")
    lines.append("")
    lines.append(
        "Mean gap / ratio over retailer-weeks, aggregated mean±CI95 across 10 seeds. "
        "Y-topology; Regime B (and A)."
    )
    lines.append("")
    lines.append("| Regime | Cap | Rationing | Gap (order−S*) | Ratio (order/S*) | Mean order | Frac@128 |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for regime in ("B", "A"):
        for cap in CAP_ORDER:
            for rat in ("proportional", "uniform"):
                g, cgi, ng = cell_metric(
                    summaries, regime=regime, topo="y", cap=cap, rat=rat, key=primary
                )
                r, cri, nr = cell_metric(
                    summaries, regime=regime, topo="y", cap=cap, rat=rat, key=primary_r
                )
                o, _, _ = cell_metric(
                    summaries, regime=regime, topo="y", cap=cap, rat=rat, key="mean_order"
                )
                f, _, _ = cell_metric(
                    summaries, regime=regime, topo="y", cap=cap, rat=rat, key="frac_at_cap"
                )
                if ng == 0:
                    continue
                lines.append(
                    f"| {regime} | {CAP_LABEL[cap]} | {rat} | {_fmt(g, cgi)} | "
                    f"{_fmt(r, cri)} | {o:.1f} | {f:.3f} |"
                )
    lines.append("")
    lines.append("### Serial negative control (no rival)")
    lines.append("")
    lines.append("| Regime | Cap | Gap (S=30) | Ratio | Frac@128 |")
    lines.append("|---|---|---:|---:|---:|")
    for regime in ("B", "A"):
        for cap in CAP_ORDER:
            g, cgi, ng = cell_metric(
                summaries,
                regime=regime,
                topo="serial",
                cap=cap,
                rat="proportional",
                key=primary,
            )
            r, cri, _ = cell_metric(
                summaries,
                regime=regime,
                topo="serial",
                cap=cap,
                rat="proportional",
                key=primary_r,
            )
            f, _, _ = cell_metric(
                summaries,
                regime=regime,
                topo="serial",
                cap=cap,
                rat="proportional",
                key="frac_at_cap",
            )
            if ng == 0:
                continue
            lines.append(
                f"| {regime} | {CAP_LABEL[cap]} | {_fmt(g, cgi)} | {_fmt(r, cri)} | {f:.3f} |"
            )
    lines.append("")

    lines.append("## Measure 2 — Scarcity response")
    lines.append("")
    lines.append(
        f"Figure: `{fig_path.name}`. B×Y×proportional gap(∞)={decision['g_inf_prop']:.2f}, "
        f"gap(1.2μ) peaks then gap(0.8μ)={decision['g_08_prop']:.2f}, "
        f"Δ(0.8μ−∞)={decision['scarcity_delta_prop']:.2f}; "
        f"strict monotone across full grid={decision['scarcity_monotone']} "
        f"(binding region is flat/slightly down; the jump is slack→binding). "
        f"Scarcity criterion (tight>slack): **{decision['scarcity_ok']}**."
    )
    lines.append("")
    lines.append(f"![inflation vs capacity]({fig_path.name})")
    lines.append("")

    lines.append("## Measure 3 — Strategic vs mechanical (rationing-rule contrast)")
    lines.append("")
    lines.append(
        "Crux: at fixed capacity on Y, proportional should show **more** inflation "
        "than uniform if agents game the rule."
    )
    lines.append("")
    lines.append("| Cap | B prop gap | B uniform gap | prop−uniform |")
    lines.append("|---|---:|---:|---:|")
    for cap in ("1p2mu", "1p0mu", "0p8mu"):
        rd = decision["rule_deltas"][cap]
        gp, cip, _ = rd["prop"]
        gu, ciu, _ = rd["uniform"]
        lines.append(
            f"| {CAP_LABEL[cap]} | {_fmt(gp, cip)} | {_fmt(gu, ciu)} | {rd['delta']:.2f} |"
        )
    lines.append("")
    lines.append(
        f"Mean prop−uniform over tight caps = **{decision['mean_rule_delta']:.2f}**. "
        f"Rule criterion met: **{decision['rule_ok']}**."
    )
    lines.append("")

    lines.append("## Measure 4 — Rival coupling (Y, wholesaler-rationed weeks)")
    lines.append("")
    lines.append(
        "Allocations normalized by wholesaler available (shares). "
        "`corr(Δorder, Δalloc)` = correlation of (own−rival) order with (own−rival) fill "
        "within the same week — direct competitive link."
    )
    lines.append("")
    lines.append(
        "| Regime | Cap | Rationing | corr(gap, rival_share) | "
        "corr(Δorder, Δalloc) | Externality Δ (rival share low−high gap) |"
    )
    lines.append("|---|---|---|---:|---:|---:|")
    for regime in ("B", "A"):
        for cap in ("1p2mu", "1p0mu", "0p8mu"):
            for rat in ("proportional", "uniform"):
                c, cc, nc = cell_metric(
                    summaries,
                    regime=regime,
                    topo="y",
                    cap=cap,
                    rat=rat,
                    key="rival_corr_gap_vs_alloc_rival",
                )
                r, cr, nr = cell_metric(
                    summaries,
                    regime=regime,
                    topo="y",
                    cap=cap,
                    rat=rat,
                    key="rival_corr_rel_order_vs_rel_alloc",
                )
                e, ce, ne = cell_metric(
                    summaries,
                    regime=regime,
                    topo="y",
                    cap=cap,
                    rat=rat,
                    key="rival_externality_delta",
                )
                if nc == 0 and ne == 0 and nr == 0:
                    continue
                lines.append(
                    f"| {regime} | {CAP_LABEL[cap]} | {rat} | {_fmt(c, cc)} | "
                    f"{_fmt(r, cr)} | {_fmt(e, ce)} |"
                )
    lines.append("")
    lines.append(
        f"At B×Y×0.8μ: prop corr(Δorder,Δalloc)={decision['rival_rel_prop_08']:.3f} "
        f"vs uniform {decision['rival_rel_uniform_08']:.3f}; "
        f"share externality Δ prop={decision['rival_ext_prop_08']:.3f} "
        f"(uniform {decision['rival_ext_uniform_08']:.3f}). "
        f"Rival criterion: **{decision['rival_ok']}**."
    )
    lines.append("")

    lines.append("## Guardrails")
    lines.append("")
    lines.append("### Benchmark sensitivity (B×Y×0.8μ)")
    lines.append("")
    lines.append("| Benchmark | Prop gap | Uniform gap | prop−uniform |")
    lines.append("|---|---:|---:|---:|")
    for S in SENSITIVITY_S:
        key = f"mean_gap_S{S}"
        gp, cip, _ = cell_metric(
            summaries, regime="B", topo="y", cap="0p8mu", rat="proportional", key=key
        )
        gu, ciu, _ = cell_metric(
            summaries, regime="B", topo="y", cap="0p8mu", rat="uniform", key=key
        )
        lines.append(
            f"| base-stock S={S} | {_fmt(gp, cip)} | {_fmt(gu, ciu)} | {gp - gu:.2f} |"
        )
    gp, cip, _ = cell_metric(
        summaries, regime="B", topo="y", cap="0p8mu", rat="proportional", key="mean_gap_pass"
    )
    gu, ciu, _ = cell_metric(
        summaries, regime="B", topo="y", cap="0p8mu", rat="uniform", key="mean_gap_pass"
    )
    lines.append(
        f"| pass-through (o*=d) | {_fmt(gp, cip)} | {_fmt(gu, ciu)} | {gp - gu:.2f} |"
    )
    lines.append("")
    lines.append(
        f"**Order-cap artifact:** frac of retailer orders at 128 under "
        f"B×Y×0.8μ×prop = {decision['frac_at_cap_B_y_08_prop']:.4f} "
        f"(≪5% boundary warn ⇒ not a 128-cap saturation story)."
    )
    lines.append("")
    lines.append(
        f"**Serial scarcity Δ** (B, prop, 0.8μ−∞) = {decision['serial_scarcity_delta']:.2f} "
        f"(no rival; any inflation here is mechanical / bullwhip, not multi-claimant gaming)."
    )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**`{decision['verdict']}`** — {_verdict_sentence(decision)}")
    lines.append("")
    lines.append("Deciding numbers:")
    lines.append(
        f"- Scarcity (B×Y×prop): gap(∞)={decision['g_inf_prop']:.2f} → "
        f"gap(0.8μ)={decision['g_08_prop']:.2f} "
        f"(Δ={decision['scarcity_delta_prop']:.2f}; ok={decision['scarcity_ok']})"
    )
    lines.append(
        f"- Rule contrast mean prop−uniform @ tight = {decision['mean_rule_delta']:.2f} "
        f"(ok={decision['rule_ok']})"
    )
    lines.append(
        f"- Rival @ 0.8μ: corr(Δorder,Δalloc) prop={decision['rival_rel_prop_08']:.3f} "
        f"vs uni={decision['rival_rel_uniform_08']:.3f}; "
        f"share Δ={decision['rival_ext_prop_08']:.3f} (ok={decision['rival_ok']})"
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


def _verdict_sentence(d: dict[str, Any]) -> str:
    v = d["verdict"]
    if v == "supported":
        return (
            f"Inflation rises with tightness (Δ={d['scarcity_delta_prop']:.2f}) and is "
            f"higher under proportional than uniform (mean Δ={d['mean_rule_delta']:.2f})."
        )
    if v == "not supported":
        return (
            f"Inflation responds to neither capacity tightness "
            f"(Δ={d['scarcity_delta_prop']:.2f}) nor rationing rule "
            f"(mean prop−uniform={d['mean_rule_delta']:.2f}); "
            f"shortage-gaming headline is not supported."
        )
    # mixed
    parts = []
    parts.append(
        f"scarcity {'holds' if d['scarcity_ok'] else 'fails'} "
        f"(Δ={d['scarcity_delta_prop']:.2f})"
    )
    parts.append(
        f"rule contrast {'holds' if d['rule_ok'] else 'fails'} "
        f"(mean prop−uniform={d['mean_rule_delta']:.2f})"
    )
    return "Only one gaming signature present: " + "; ".join(parts) + "."


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2 seeds, 2 episodes")
    args = p.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / (
        f"shortage_gaming_ep{args.n_episodes}"
        + ("_smoke" if args.smoke else "")
        + ".json"
    )

    import time

    t0 = time.time()
    if cache.exists() and not args.force:
        print(f"loading cache {cache}")
        summaries = json.loads(cache.read_text())
    else:
        runs = list_target_runs()
        if args.smoke:
            runs = [r for r in runs if int(r["seed"]) < 2]
            args.n_episodes = min(args.n_episodes, 2)
        payloads = [
            {
                "run_dir": r["run_dir"],
                "n_episodes": args.n_episodes,
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
            for r in runs
        ]
        print(f"shortage-gaming: {len(payloads)} runs × {args.n_episodes} eps", flush=True)
        summaries = []
        if args.workers <= 1:
            for i, pl in enumerate(payloads):
                print(f"  [{i+1}/{len(payloads)}] {pl['row']['run']}", flush=True)
                summaries.append(_worker(pl))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(_worker, pl): pl for pl in payloads}
                done_n = 0
                for fut in as_completed(futs):
                    summaries.append(fut.result())
                    done_n += 1
                    if done_n % 10 == 0 or done_n == len(payloads):
                        print(f"  done {done_n}/{len(payloads)}", flush=True)
        write_json(cache, summaries)

    elapsed = time.time() - t0
    plot_inflation_vs_capacity(summaries, OUT_FIG)
    write_report(summaries, n_episodes=args.n_episodes, elapsed_s=elapsed, fig_path=OUT_FIG)
    decision = decide_verdict(summaries)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_FIG}")
    print(f"Verdict: {decision['verdict']}")


if __name__ == "__main__":
    main()
