#!/usr/bin/env python3
"""Markovian-vs-recurrent cost table + shortage-gaming recheck.

Matched-deterministic eval (greedy=True, seed = cfg.seed + 10_000), consistent
with artifacts/diagnostics/eval_mode_blast_radius.md. Regime A only (order-only
LLM-comparable setting). No training here — reads frozen checkpoints.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.common import ci95, load_trainer  # noqa: E402
from beer_distribution_rl.agents.baselines import base_stock_order  # noqa: E402
from beer_distribution_rl.agents.ippo.networks import RecurrentActorCritic  # noqa: E402
from beer_distribution_rl.env.core_types import Role  # noqa: E402

MARKOV_DIR = ROOT / "artifacts" / "runs" / "ippo" / "tier1_v11"
REC_DIR = ROOT / "artifacts" / "runs" / "ippo" / "recurrent_baseline"
OUT_MD = ROOT / "artifacts" / "diagnostics" / "recurrent_baseline.md"
CACHE = ROOT / "analysis" / "diag" / "cache" / "recurrent_baseline.json"

PRIMARY_S = 30
EVAL_SEED_OFFSET = 10_000
CAP_ORDER = ["inf", "1p0mu", "0p8mu"]
CAP_LABEL = {"inf": "∞", "1p0mu": "1.0μ", "0p8mu": "0.8μ"}
HEADLINE_CAPS = set(CAP_ORDER)
N_EP = 20


def _load_index(path: Path) -> list[dict[str, Any]]:
    return json.loads((path / "index.json").read_text())


def _headline_rows(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in index:
        if r.get("status") not in ("ok", "skipped"):
            continue
        if r.get("regime") != "A" or r.get("demand") != "ar1":
            continue
        if r.get("rationing") != "proportional":
            continue
        if r.get("topology") not in ("serial", "y"):
            continue
        if r.get("capacity_tag") not in HEADLINE_CAPS:
            continue
        out.append(r)
    return out


def _fmt(m: float, ci: float) -> str:
    if not math.isfinite(m):
        return "—"
    return f"{m:.1f}±{ci:.1f}"


def cost_table(markov: list[dict], rec: list[dict]) -> list[dict[str, Any]]:
    """Aggregate mean±CI95 system cost by (topo, cap)."""
    rows = []
    for topo in ("serial", "y"):
        for cap in CAP_ORDER:
            def _xs(src: list[dict]) -> list[float]:
                return [
                    float(r["eval/mean_system_cost"])
                    for r in src
                    if r.get("topology") == topo
                    and r.get("capacity_tag") == cap
                    and "eval/mean_system_cost" in r
                    and np.isfinite(float(r["eval/mean_system_cost"]))
                ]

            xm, cm, nm = ci95(_xs(markov))
            xr, cr, nr = ci95(_xs(rec))
            delta = xr - xm if (nm and nr) else float("nan")
            rows.append(
                {
                    "topology": topo,
                    "capacity_tag": cap,
                    "markov_mean": xm,
                    "markov_ci": cm,
                    "markov_n": nm,
                    "rec_mean": xr,
                    "rec_ci": cr,
                    "rec_n": nr,
                    "delta": delta,
                }
            )
    return rows


def _retailer_roles(trainer) -> list[Role]:
    return [r for r in trainer.roles if r in (Role.RETAILER, Role.RETAILER_B)]


def collect_order_rows(trainer, *, n_episodes: int, seed: int) -> dict[str, Any]:
    """Matched-deterministic rollout with GRU hidden carry when recurrent."""
    core = trainer.core
    retailers = _retailer_roles(trainer)
    order_cap = int(trainer.cfg.order_cap)
    rows: list[dict[str, Any]] = []
    n_at_cap = 0
    n_orders = 0

    for ep in range(n_episodes):
        states = core.reset(seed + ep)
        done = False
        eval_h: dict[Role, torch.Tensor] = {}
        if trainer.recurrent:
            for r in trainer.roles:
                pol = trainer.policies[r]
                assert isinstance(pol, RecurrentActorCritic)
                eval_h[r] = pol.initial_hidden(1, trainer.device)
        while not done:
            pre = {r: states[r] for r in trainer.roles}
            orders: dict = {}
            with torch.no_grad():
                for r in trainer.roles:
                    o = torch.as_tensor(
                        trainer._obs(states, r, core), device=trainer.device
                    ).unsqueeze(0)
                    if trainer.recurrent:
                        a, _, _, h_new = trainer._policy_act(
                            r, o, greedy=True, h=eval_h[r]
                        )
                        eval_h[r] = h_new
                    else:
                        a, _, _ = trainer._policy_act(r, o, greedy=True)
                    orders[r] = trainer._decode_order(int(a.item()), states[r])

            states, _rewards, done, info = core.step(orders, None)

            wh = Role.WHOLESALER
            wh_alloc = info.allocations.get(wh, {}) if hasattr(info, "allocations") else {}
            requested = {}
            for c in (Role.RETAILER, Role.RETAILER_B):
                if c not in core.roles:
                    continue
                alloc_c = int(wh_alloc.get(c, 0))
                backlog_c = int(core._states[wh].claimant_backlog.get(c, 0))
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
                bench = int(base_stock_order(pre[r], S=PRIMARY_S, order_cap=order_cap))
                alloc_to_me = int(wh_alloc.get(r, 0)) if wh_alloc else None
                rival = Role.RETAILER_B if r == Role.RETAILER else Role.RETAILER
                alloc_to_rival = (
                    int(wh_alloc.get(rival, 0)) if (wh_alloc and rival in core.roles) else None
                )
                rows.append(
                    {
                        "ep": ep,
                        "t": int(core.t),
                        "role": name,
                        "order": placed,
                        "gap_S30": placed - bench,
                        "ratio_S30": placed / max(bench, 1),
                        "alloc": alloc_to_me,
                        "alloc_rival": alloc_to_rival,
                        "wholesaler_rationed": bool(wholesaler_rationed),
                        "wh_available": available,
                    }
                )

    return {
        "rows": rows,
        "frac_at_cap": float(n_at_cap / max(n_orders, 1)),
        "mean_gap_S30": float(np.mean([r["gap_S30"] for r in rows])) if rows else float("nan"),
        "mean_ratio_S30": float(np.mean([r["ratio_S30"] for r in rows])) if rows else float("nan"),
        "mean_order": float(np.mean([r["order"] for r in rows])) if rows else float("nan"),
    }


def _gaming_worker(payload: dict[str, Any]) -> dict[str, Any]:
    trainer, _ = load_trainer(payload["run_dir"])
    seed = int(trainer.cfg.seed) + EVAL_SEED_OFFSET
    collected = collect_order_rows(trainer, n_episodes=N_EP, seed=seed)
    return {
        **payload["row"],
        "frac_at_cap": collected["frac_at_cap"],
        "mean_gap_S30": collected["mean_gap_S30"],
        "mean_ratio_S30": collected["mean_ratio_S30"],
        "mean_order": collected["mean_order"],
        "arch": payload["arch"],
    }


def list_gaming_runs(run_dir: Path, arch: str) -> list[dict[str, Any]]:
    idx = _load_index(run_dir)
    out = []
    for r in idx:
        if r.get("status") not in ("ok", "skipped"):
            continue
        if r.get("regime") != "A" or r.get("demand") != "ar1":
            continue
        if r.get("topology") != "y":
            continue
        if r.get("rationing") not in ("proportional", "uniform"):
            continue
        if r.get("capacity_tag") not in HEADLINE_CAPS:
            continue
        ckpt = run_dir / r["run"] / "checkpoints"
        has = (ckpt / "policy_retailer.pt").exists() or (ckpt / "policy_retailer_a.pt").exists()
        if not has:
            continue
        out.append({**r, "run_dir": str(run_dir / r["run"]), "arch": arch})
    return out


def cell_metric(
    summaries: list[dict[str, Any]],
    *,
    arch: str,
    cap: str,
    rat: str,
    key: str,
) -> tuple[float, float, int]:
    xs = [
        float(s[key])
        for s in summaries
        if s.get("arch") == arch
        and s.get("capacity_tag") == cap
        and s.get("rationing") == rat
        and key in s
        and np.isfinite(float(s[key]))
    ]
    return ci95(xs)


def run_gaming(workers: int = 8) -> list[dict[str, Any]]:
    payloads = []
    for arch, d in (("markovian", MARKOV_DIR), ("recurrent", REC_DIR)):
        for r in list_gaming_runs(d, arch):
            payloads.append({"run_dir": r["run_dir"], "row": r, "arch": arch})
    results: list[dict[str, Any]] = []
    if not payloads:
        return results
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(_gaming_worker, p): p for p in payloads}
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def write_report(
    *,
    sha: str,
    cost_rows: list[dict[str, Any]],
    gaming: list[dict[str, Any]],
    markov_n: int,
    rec_n: int,
) -> None:
    # Memory-only finding
    deltas = [r["delta"] for r in cost_rows if np.isfinite(r["delta"])]
    mean_abs_delta = float(np.mean(np.abs(deltas))) if deltas else float("nan")
    any_large = any(abs(d) > 50 for d in deltas)  # coarse heuristic vs cost scale

    lines: list[str] = []
    lines.append("# Recurrent IPPO baseline (memory-matched MLP)")
    lines.append("")
    lines.append(f"**Branch tip SHA (at report write):** `{sha}`")
    lines.append(f"**Branched from main:** `061aa59235397b7360c32a01cf4f98add0dd503a`")
    lines.append("")
    lines.append("## Architecture choice")
    lines.append("")
    lines.append(
        "**Chose: GRU over per-week local observations** "
        "(not stacked-history concatenation)."
    )
    lines.append("")
    lines.append("Rationale:")
    lines.append(
        "- Matches the planned LLM's full T=52 own-history retention without "
        "blowing up the observation dimension (W×obs_dim)."
    )
    lines.append(
        "- Fits the existing R1 runner (YAML + seed + git SHA, vec envs, "
        "matched-deterministic greedy eval) with hidden-state carry + reset on done."
    )
    lines.append(
        "- Single-step BPTT with stored (detached) input hiddens keeps the "
        "CleanRL-style shuffled minibatch update intact."
    )
    lines.append("")
    lines.append("Policy: `RecurrentActorCritic` — GRU(obs_dim→128) → shared 2×256 "
                 "actor/critic MLPs. One module per role; no parameter sharing.")
    lines.append("")
    lines.append("## Shared information set (apples-to-apples vs LLM)")
    lines.append("")
    lines.append(
        "Both the recurrent MLP and the planned order-only LLM see **own history only** "
        "(E1 no-leak). Per-week content aligned with Check 3 structured history:"
    )
    lines.append("")
    lines.append("| Check 3 history field | IPPO local obs / GRU input |")
    lines.append("|---|---|")
    lines.append("| `demand_or_incoming` | `last_demand_or_order` |")
    lines.append("| `ship_in` / `alloc_recv` | `last_shipment_received` |")
    lines.append("| `ordered` | `last_order_placed` |")
    lines.append("| `inv`, `backlog` | `inventory`, `backlog` |")
    lines.append("| `cost` | recoverable via `h`,`b` coeffs in obs × inv/backlog |")
    lines.append("")
    lines.append(
        "Plus pipelines, `on_order`, `t/horizon`. **Never** rival private inventories "
        "or privileged `customer_demand`/`true_demand` for upstream agents. "
        "Cheap-talk board is off (Regime A). Rewards remain strictly local "
        "(no system term, no honesty reward)."
    )
    lines.append("")
    lines.append("## Markovian vs recurrent cost (Regime A × prop × AR(1), 10 seeds)")
    lines.append("")
    lines.append(
        f"Matched-deterministic `final_eval` (`n_episodes≥20` at train end). "
        f"Markovian n≈{markov_n} cells; recurrent n≈{rec_n} cells from "
        "`artifacts/runs/ippo/tier1_v11` vs `.../recurrent_baseline`."
    )
    lines.append("")
    lines.append("| Topo | Cap | Markovian (mean±CI95) | Recurrent (mean±CI95) | Δ (rec−mark) |")
    lines.append("|---|---|---:|---:|---:|")
    for r in cost_rows:
        d = f"{r['delta']:+.1f}" if np.isfinite(r["delta"]) else "—"
        lines.append(
            f"| {r['topology']} | {CAP_LABEL[r['capacity_tag']]} | "
            f"{_fmt(r['markov_mean'], r['markov_ci'])} | "
            f"{_fmt(r['rec_mean'], r['rec_ci'])} | {d} |"
        )
    lines.append("")
    if any_large:
        finding = (
            f"Memory alone **moves** cost materially (mean |Δ|≈{mean_abs_delta:.1f}). "
            "History helps / hurts an MLP on this task — report cell-level signs above."
        )
    else:
        finding = (
            f"Memory alone does **not** dramatically change cost "
            f"(mean |Δ|≈{mean_abs_delta:.1f} across headline cells). "
            "The LLM-vs-MLP comparison can treat this recurrent run as the "
            "memory-matched reference; residual gaps are then more attributable "
            "to language priors / GRPO than to history length."
        )
    lines.append(f"**Memory-only finding:** {finding}")
    lines.append("")
    lines.append("## Shortage-gaming recheck (recurrent vs Markovian, Regime A × Y × AR(1))")
    lines.append("")
    lines.append(
        f"Order inflation gap vs base-stock S={PRIMARY_S}; matched-deterministic "
        f"re-roll (`greedy=True`, seed+{EVAL_SEED_OFFSET}, {N_EP} eps)."
    )
    lines.append("")
    lines.append("| Arch | Cap | Rationing | Gap (order−S*) | Ratio | Mean order | Frac@128 |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for arch in ("markovian", "recurrent"):
        for cap in CAP_ORDER:
            for rat in ("proportional", "uniform"):
                if cap == "inf" and rat != "proportional":
                    continue
                g, gci, n = cell_metric(gaming, arch=arch, cap=cap, rat=rat, key="mean_gap_S30")
                rr, rci, _ = cell_metric(gaming, arch=arch, cap=cap, rat=rat, key="mean_ratio_S30")
                o, oci, _ = cell_metric(gaming, arch=arch, cap=cap, rat=rat, key="mean_order")
                f, fci, _ = cell_metric(gaming, arch=arch, cap=cap, rat=rat, key="frac_at_cap")
                if n == 0:
                    continue
                lines.append(
                    f"| {arch} | {CAP_LABEL[cap]} | {rat} | {_fmt(g, gci)} | "
                    f"{_fmt(rr, rci)} | {_fmt(o, oci)} | {_fmt(f, fci)} |"
                )
    lines.append("")
    # Deciding numbers under recurrent
    g_inf, _, n_inf = cell_metric(
        gaming, arch="recurrent", cap="inf", rat="proportional", key="mean_gap_S30"
    )
    g_08, _, n_08 = cell_metric(
        gaming, arch="recurrent", cap="0p8mu", rat="proportional", key="mean_gap_S30"
    )
    scarcity = g_08 - g_inf if (n_inf and n_08) else float("nan")
    rule_deltas = []
    for cap in ("1p0mu", "0p8mu"):
        gp, _, np_ = cell_metric(
            gaming, arch="recurrent", cap=cap, rat="proportional", key="mean_gap_S30"
        )
        gu, _, nu = cell_metric(
            gaming, arch="recurrent", cap=cap, rat="uniform", key="mean_gap_S30"
        )
        if np_ and nu:
            rule_deltas.append(gp - gu)
    mean_rule = float(np.mean(rule_deltas)) if rule_deltas else float("nan")
    scarcity_ok = bool(np.isfinite(scarcity) and scarcity > 0)
    rule_ok = bool(np.isfinite(mean_rule) and mean_rule > 0)
    if scarcity_ok and rule_ok:
        verdict = "supported"
    elif scarcity_ok or rule_ok:
        verdict = "partial"
    else:
        verdict = "not_supported"
    lines.append(
        f"**Recurrent gaming verdict:** `{verdict}` — "
        f"scarcity Δ(0.8μ−∞) under prop = {scarcity:+.2f} (ok={scarcity_ok}); "
        f"mean prop−uniform @ tight = {mean_rule:+.2f} (ok={rule_ok})."
    )
    # Compare to markovian on same cells
    g_inf_m, _, _ = cell_metric(
        gaming, arch="markovian", cap="inf", rat="proportional", key="mean_gap_S30"
    )
    g_08_m, _, _ = cell_metric(
        gaming, arch="markovian", cap="0p8mu", rat="proportional", key="mean_gap_S30"
    )
    scarcity_m = g_08_m - g_inf_m if np.isfinite(g_08_m) and np.isfinite(g_inf_m) else float("nan")
    lines.append(
        f"Markovian reference scarcity Δ = {scarcity_m:+.2f}. "
        "If history alone suppressed gaming, recurrent scarcity/rule contrasts would collapse; "
        "if amplified, gaps would widen — interpret relative to these numbers."
    )
    lines.append("")
    lines.append("## Non-negotiables (unchanged)")
    lines.append("")
    lines.append("- One policy per role (A); independence asserted at init.")
    lines.append("- Strictly local per-agent cost rewards; no system term; no honesty reward.")
    lines.append("- Recurrence changes **memory**, not reward or information-leak rules.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Runs: `artifacts/runs/ippo/recurrent_baseline/`")
    lines.append("- Cache: `analysis/diag/cache/recurrent_baseline.json`")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--skip-gaming", action="store_true")
    args = p.parse_args()

    sha = (
        __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip()
    )
    markov = _headline_rows(_load_index(MARKOV_DIR))
    rec = _headline_rows(_load_index(REC_DIR))
    costs = cost_table(markov, rec)
    gaming: list[dict[str, Any]] = []
    if not args.skip_gaming:
        print(f"Shortage-gaming re-roll on {len(list_gaming_runs(MARKOV_DIR, 'markovian'))} markov + "
              f"{len(list_gaming_runs(REC_DIR, 'recurrent'))} recurrent cells…")
        gaming = run_gaming(workers=args.workers)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps({"sha": sha, "cost_rows": costs, "gaming": gaming}, indent=2)
    )
    write_report(
        sha=sha,
        cost_rows=costs,
        gaming=gaming,
        markov_n=len(markov),
        rec_n=len(rec),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
