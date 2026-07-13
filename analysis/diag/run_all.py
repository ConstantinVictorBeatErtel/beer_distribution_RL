#!/usr/bin/env python3
"""Run all diagnostics and write analysis/DIAGNOSTICS.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag import d1_channel_gap  # noqa: E402
from analysis.diag import d2_signal_ablation  # noqa: E402
from analysis.diag import d3_listener_sensitivity  # noqa: E402
from analysis.diag import d4_sharing_preference  # noqa: E402
from analysis.diag import d5_rationing_bind  # noqa: E402
from analysis.diag import d6_demand_info  # noqa: E402
from analysis.diag.common import FIG_DIR, REPORT_PATH, ensure_dirs  # noqa: E402


def _fmt_pct(x: float | None) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{100 * x:.1f}%"


def _verdict_d1(d1: dict) -> tuple[str, str]:
    return (
        "INCONCLUSIVE — no Regime A capacity matrix in completed logs "
        "(M3 is B-only; A exists only as classic_step seed0)",
        "dead channel / live channel (cannot decide from D1)",
    )


def _verdict_d2(d2: dict) -> tuple[str, str]:
    summary = d2.get("summary") or []
    if not summary:
        return "no data", "structural / other"
    bearing = sum(1 for s in summary if s.get("channel_load_bearing"))
    ignored = sum(1 for s in summary if s.get("channel_ignored"))
    harmful = sum(1 for s in summary if s.get("channel_harmful"))
    gaps = [s.get("max_abs_gap_vs_intact", 0) for s in summary]
    mean_gap = float(sum(gaps) / len(gaps)) if gaps else 0.0
    mean_signed = float(
        sum(s.get("mean_ablation_minus_intact", 0.0) for s in summary) / len(summary)
    )
    if bearing == 0 and ignored + harmful >= max(1, len(summary) // 2):
        return (
            f"(a) not better than (b–d): mean |Δ|≈{mean_gap:.1f}, "
            f"mean (ablation−intact)≈{mean_signed:.1f}; "
            f"ignored={ignored}, harmful={harmful} → listeners do not benefit from channel",
            "structural (babbling equilibrium)",
        )
    if bearing == len(summary):
        return (
            f"intact clearly best on {bearing}/{len(summary)} capacities "
            f"(mean |Δ|={mean_gap:.1f})",
            "other (channel used)",
        )
    return (
        f"mixed: load-bearing={bearing}, ignored={ignored}, harmful={harmful} "
        f"(mean |Δ|={mean_gap:.1f})",
        "mixed / structural lean" if bearing == 0 else "mixed",
    )


def _verdict_d3(d3: dict) -> tuple[str, str]:
    summary = d3.get("summary") or []
    if not summary:
        return "no data", "corroborates D2 or not"
    ratios = [s["ratio_mean"] for s in summary if s["ratio_mean"] == s["ratio_mean"]]
    mean_r = float(sum(ratios) / len(ratios)) if ratios else float("nan")
    n_low = sum(1 for s in summary if s.get("signal_lt_inventory"))
    if n_low >= max(1, len(summary) - 1):
        return (
            f"signal sensitivity ≪ inventory (ratio≈{mean_r:.2f}); ignored at policy level",
            "corroborates D2",
        )
    return (
        f"signal/inventory sensitivity ratio≈{mean_r:.2f} ({n_low}/{len(summary)} caps ≪)",
        "does not fully corroborate ignore",
    )


def _verdict_d4(d4: dict) -> tuple[str, str]:
    key = d4.get("interpretation_key", "mixed")
    ent = d4.get("mean_entropy_frac_of_max", float("nan"))
    moving = d4.get("frac_still_moving", float("nan"))
    half = d4.get("mean_frac_near_half", float("nan"))
    result = (
        f"H_broadcast≈{ent:.2f}·ln2; still-moving seeds≈{_fmt_pct(moving)}; "
        f"frac near 0.5≈{_fmt_pct(half)} → {key}"
    )
    if "indifference" in key:
        points = "indifference"
    elif "undertrained" in key:
        points = "undertrained"
    elif "multi-equilibrium" in key:
        points = "multi-equilibrium"
    else:
        points = "mixed"
    return result, points


def _verdict_d5(d5: dict) -> tuple[str, str]:
    alloc = d5.get("tight_mean_alloc_frac", float("nan"))
    cap = d5.get("tight_mean_cap_bind_frac", float("nan"))
    key = d5.get("interpretation_key", "")
    result = (
        f"tight caps (1.0μ, 0.8μ): capacity-bind≈{_fmt_pct(cap)}, "
        f"allocation-shortfall≈{_fmt_pct(alloc)} → {key}"
    )
    if "absent" in key or "rare" in key:
        points = "gaming incentive absent"
    else:
        points = "gaming incentive present"
    return result, points


def _verdict_d6(d6: dict) -> tuple[str, str]:
    u = d6["uniform"]
    result = (
        f"U[0,15] lag-1 R²≈{u['lag1_r2']:.3f}, MI≈{u['mi_bits']:.3f} bits; "
        f"AR(1) φ=0.7 R²≈{d6['ar1']['lag1_r2']:.3f}"
    )
    points = "demand uninformative" if "uninformative" in d6.get("interpretation_key", "") else "demand informative"
    return result, points


def _recommendation(points: dict[str, str]) -> str:
    structural = 0
    undertrained = 0
    if "structural" in points.get("D2", "").lower() or "babbling" in points.get("D2", "").lower():
        structural += 1
    if points.get("D4") == "indifference":
        structural += 1
    if points.get("D4") == "undertrained":
        undertrained += 1
    if "absent" in points.get("D5", ""):
        structural += 1
    if "uninformative" in points.get("D6", ""):
        structural += 1

    if undertrained and structural >= 2:
        choice = "C"
        text = (
            "**Recommendation (C) — both:** Extend training only if D4 indicated undertraining; "
            "regardless, environment v1.1 is required because the diagnostics show structural "
            "failures (uninformative demand and/or non-binding rationing and/or babbling channel) "
            "that more PPO steps cannot invent. Rerun the Tier-1 matrix on informative demand "
            "(AR(1)/regime-switch), tighter binding capacity, and Y-topology before LLM spend."
        )
    elif undertrained and structural < 2:
        choice = "B"
        text = (
            "**Recommendation (B) — extend training:** Share-rate / policy preference still looks "
            "unconverged; push past 50k steps on the current env before changing the environment "
            "or spending LLM compute."
        )
    else:
        choice = "A"
        text = (
            "**Recommendation (A) — environment v1.1 + rerun:** The verdict table points to "
            "structural environment issues (not merely short training): the cheap-talk channel is "
            "economically unused or babbling, demand U[0,15] has near-zero predictive value, and/or "
            "rationing rarely creates a shortage-gaming incentive. Switch to informative demand "
            "(AR(1) φ≈0.7 / regime-switching), ensure capacity/allocation actually binds, and "
            "introduce Y-topology for multi-claimant rationing — then rerun Tier-1 before LLM cells."
        )
    return f"{text}\n\nSelected option: **({choice})**."


def write_report(d1, d2, d3, d4, d5, d6) -> None:
    v1 = _verdict_d1(d1)
    v2 = _verdict_d2(d2)
    v3 = _verdict_d3(d3)
    v4 = _verdict_d4(d4)
    v5 = _verdict_d5(d5)
    v6 = _verdict_d6(d6)
    points = {"D1": v1[1], "D2": v2[1], "D3": v3[1], "D4": v4[1], "D5": v5[1], "D6": v6[1]}

    lines: list[str] = []
    lines.append("# Tier-1 diagnostics (eval-only)")
    lines.append("")
    lines.append(
        "Root-cause analysis of mediocre Regime-B sharing (~50%), weak honesty, and mild "
        "order inflation — **no new training**. Frozen M3 Regime-B checkpoints + existing "
        "logs. Figures: `analysis/figs/diag/`. Regenerate: `make diagnostics`."
    )
    lines.append("")
    lines.append("## Scope & artifacts")
    lines.append("")
    lines.append(
        "- **Matrix available:** Regime B × {∞,1.5μ,1.2μ,1.0μ,0.8μ} × {proportional, honesty_weighted} "
        "× 10 seeds (checkpoints present locally)."
    )
    lines.append(
        "- **Regime A/C:** M2 classic_step seed0 only — **no capacity-swept A/C matrix**."
    )
    lines.append(
        "- **Primary eval slice:** proportional rationing (on serial topology honesty_weighted ≡ "
        "single-claimant fill; policies still differ by seed/run)."
    )
    lines.append("- **Seeds:** fixed via `DIAG_EVAL_SEED_OFFSET` in `analysis/diag/common.py`.")
    lines.append("")

    # D1
    lines.append("## D1 — Does the channel matter at all? (Regime A vs B cost gap)")
    lines.append("")
    lines.append(f"**Data gap:** {d1['data_gap']}")
    lines.append("")
    lines.append(
        "Matched A vs B gaps by capacity are **not computable** from existing logs. "
        "Figure below shows Regime B costs only (95% CI across 10 proportional seeds)."
    )
    lines.append("")
    lines.append("![D1 B costs](figs/diag/d1_regime_b_costs.png)")
    lines.append("")
    lines.append("![D1 M2 anchors](figs/diag/d1_m2_ac_anchors.png)")
    lines.append("")
    lines.append("| Cap | B system ±CI95 | A system ±CI95 | Gap % of A |")
    lines.append("|---|---:|---:|---:|")
    for row in d1.get("regime_B_by_capacity") or []:
        lines.append(
            f"| {row['capacity_label']} | {row['B_system_mean']:.1f}±{row['B_system_ci95']:.1f} | "
            f"n/a | n/a |"
        )
    lines.append("")
    lines.append(
        "**Interpretation key:** cannot apply |gap|<CI test — no Regime A cells. "
        f"{d1.get('interpretation','')}"
    )
    lines.append("")

    # D2
    lines.append("## D2 — Signal ablation at eval (decisive)")
    lines.append("")
    lines.append(
        f"For each frozen Regime-B checkpoint (proportional, n={d2.get('n_checkpoints')}), "
        f"{d2.get('n_episodes')} eval episodes × conditions "
        "(a) intact (b) signals zeroed (c) shuffled across agents (d) random valid values. "
        "Ablation corrupts **listener observations only**; policies still emit signals."
    )
    lines.append("")
    lines.append("![D2 ablation](figs/diag/d2_signal_ablation.png)")
    lines.append("")
    lines.append("![D2 per-agent tight](figs/diag/d2_ablation_per_agent_tight.png)")
    lines.append("")
    lines.append("| Cap | intact | zeroed | shuffled | random | max\\|Δ\\| vs intact | load-bearing? |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for s in d2.get("summary") or []:
        def cell(name: str) -> str:
            c = s["conditions"][name]
            return f"{c['system_cost_mean']:.1f}±{c['system_cost_ci95']:.1f}"

        lines.append(
            f"| {s['capacity_label']} | {cell('intact')} | {cell('zeroed')} | "
            f"{cell('shuffled')} | {cell('random')} | "
            f"{s.get('max_abs_gap_vs_intact', float('nan')):.1f} | "
            f"{'yes' if s.get('channel_load_bearing') else 'no'} |"
        )
    lines.append("")
    lines.append(f"**Result:** {v2[0]}")
    lines.append("")
    lines.append(
        "**Interpretation key:** (a)≈(b)≈(c)≈(d) ⇒ listeners ignore channel ⇒ babbling ⇒ "
        "STRUCTURAL. (a) clearly better ⇒ channel load-bearing."
    )
    lines.append("")

    # D3
    lines.append("## D3 — Listener sensitivity probe")
    lines.append("")
    lines.append(
        "Fixed obs batches from intact rollouts; sweep incoming claimed-demand signal features "
        "vs own-inventory feature; measure mean |Δ order| per unit feature change."
    )
    lines.append("")
    lines.append("![D3 sensitivity](figs/diag/d3_listener_sensitivity.png)")
    lines.append("")
    lines.append("![D3 ratio](figs/diag/d3_sensitivity_ratio.png)")
    lines.append("")
    lines.append("| Cap | signal sens. | inventory sens. | ratio |")
    lines.append("|---|---:|---:|---:|")
    for s in d3.get("summary") or []:
        lines.append(
            f"| {s['capacity_label']} | {s['signal_mean']:.3f}±{s['signal_ci95']:.3f} | "
            f"{s['inventory_mean']:.3f}±{s['inventory_ci95']:.3f} | "
            f"{s['ratio_mean']:.3f}±{s['ratio_ci95']:.3f} |"
        )
    lines.append("")
    lines.append(f"**Result:** {v3[0]}")
    lines.append("")

    # D4
    lines.append("## D4 — Sharing action: indifference vs converged preference")
    lines.append("")
    lines.append(
        "Per seed/agent: (a) Bernoulli entropy of broadcast head at frozen checkpoint; "
        "(b) share-rate trajectory from `history.json`; (c) cross-seed final share histogram. "
        "Note: training logs only store **joint** multi-head entropy — broadcast entropy was "
        "recomputed from checkpoints."
    )
    lines.append("")
    lines.append("![D4 sharing](figs/diag/d4_sharing_preference.png)")
    lines.append("")
    lines.append("| Cap | share ±CI | H_broadcast ±CI | H/ln2 | frac near 0.5 |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in d4.get("summary") or []:
        lines.append(
            f"| {s['capacity_label']} | {s['share_mean']:.2f}±{s['share_ci95']:.2f} | "
            f"{s['entropy_mean']:.3f}±{s['entropy_ci95']:.3f} | "
            f"{s['entropy_frac_of_max']:.2f} | {s['frac_near_half']:.2f} |"
        )
    lines.append("")
    lines.append(
        f"Fraction of runs with share-rate still moving at end of training: "
        f"**{_fmt_pct(d4.get('frac_still_moving'))}**."
    )
    lines.append("")
    lines.append(f"**Result:** {v4[0]}")
    lines.append("")

    # D5
    lines.append("## D5 — Does rationing ever bind?")
    lines.append("")
    lines.append(f"**Log gap:** {d5.get('log_gap')}")
    lines.append("")
    lines.append(d5.get("serial_topology_note", ""))
    lines.append("")
    lines.append("![D5 binding](figs/diag/d5_rationing_binding.png)")
    lines.append("")
    lines.append(
        "| Cap | frac capacity binds | frac allocation shortfall | infl\\|binding | infl\\|non-binding |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for s in d5.get("summary") or []:
        lines.append(
            f"| {s['capacity_label']} | "
            f"{s['frac_capacity_binds_mean']:.2f}±{s['frac_capacity_binds_ci95']:.2f} | "
            f"{s['frac_allocation_triggers_mean']:.2f}±{s['frac_allocation_triggers_ci95']:.2f} | "
            f"{s['inflation_when_binding_mean']:.2f}±{s.get('inflation_when_binding_ci95', float('nan')):.2f} | "
            f"{s['inflation_when_not_binding_mean']:.2f}±{s.get('inflation_when_not_binding_ci95', float('nan')):.2f} |"
        )
    lines.append("")
    lines.append(f"**Result:** {v5[0]}")
    lines.append("")
    lines.append(
        "**Interpretation key:** allocation triggers in <20% of weeks ⇒ shortage-gaming "
        "incentive rarely exists ⇒ STRUCTURAL (capacity/backlog too soft)."
    )
    lines.append("")

    # D6
    lines.append("## D6 — Information value of demand signal")
    lines.append("")
    lines.append(d6.get("paragraph", ""))
    lines.append("")
    lines.append("![D6 info](figs/diag/d6_demand_info.png)")
    lines.append("")
    lines.append("![D6 lag scatter](figs/diag/d6_lag_scatter.png)")
    lines.append("")
    lines.append(f"**Result:** {v6[0]} → {v6[1]}")
    lines.append("")

    # Verdict table
    lines.append("## Verdict table")
    lines.append("")
    lines.append("| Test | Result | Points to |")
    lines.append("|---|---|---|")
    lines.append(f"| D1 | {v1[0]} | {v1[1]} |")
    lines.append(f"| D2 | {v2[0]} | {v2[1]} |")
    lines.append(f"| D3 | {v3[0]} | {v3[1]} |")
    lines.append(f"| D4 | {v4[0]} | {v4[1]} |")
    lines.append(f"| D5 | {v5[0]} | {v5[1]} |")
    lines.append(f"| D6 | {v6[0]} | {v6[1]} |")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(_recommendation(points))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Generated by `python -m analysis.diag.run_all`. Figures under `{FIG_DIR.relative_to(ROOT)}/`.*"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


def main() -> int:
    p = argparse.ArgumentParser(description="Run Tier-1 eval-only diagnostics")
    p.add_argument("--episodes-d2", type=int, default=100)
    p.add_argument("--episodes-d4", type=int, default=20)
    p.add_argument("--episodes-d5", type=int, default=30)
    p.add_argument("--max-seeds", type=int, default=None, help="Limit seeds for smoke tests")
    p.add_argument("--force", action="store_true", help="Ignore caches")
    p.add_argument("--skip-d2", action="store_true")
    p.add_argument("--skip-heavy", action="store_true", help="Skip D2/D3/D4/D5 checkpoint evals")
    args = p.parse_args()

    ensure_dirs()
    print("=== D1 ===", flush=True)
    d1 = d1_channel_gap.run()
    print("=== D6 ===", flush=True)
    d6 = d6_demand_info.run()

    if args.skip_heavy:
        # Minimal stubs if only regenerating light parts
        from analysis.diag.common import CACHE_DIR, read_json

        d2 = read_json(CACHE_DIR / "d2_summary.json") if (CACHE_DIR / "d2_summary.json").exists() else {"summary": []}
        d3 = read_json(CACHE_DIR / "d3_summary.json") if (CACHE_DIR / "d3_summary.json").exists() else {"summary": []}
        d4 = read_json(CACHE_DIR / "d4_summary.json") if (CACHE_DIR / "d4_summary.json").exists() else {"summary": []}
        d5 = read_json(CACHE_DIR / "d5_summary.json") if (CACHE_DIR / "d5_summary.json").exists() else {"summary": []}
    else:
        if not args.skip_d2:
            print("=== D2 ===", flush=True)
            d2 = d2_signal_ablation.run(
                n_episodes=args.episodes_d2,
                max_seeds=args.max_seeds,
                force=args.force,
            )
        else:
            from analysis.diag.common import CACHE_DIR, read_json

            d2 = read_json(CACHE_DIR / "d2_summary.json")
        print("=== D3 ===", flush=True)
        d3 = d3_listener_sensitivity.run(max_seeds=args.max_seeds, force=args.force)
        print("=== D4 ===", flush=True)
        d4 = d4_sharing_preference.run(
            n_episodes=args.episodes_d4, max_seeds=args.max_seeds, force=args.force
        )
        print("=== D5 ===", flush=True)
        d5 = d5_rationing_bind.run(
            n_episodes=args.episodes_d5, max_seeds=args.max_seeds, force=args.force
        )

    write_report(d1, d2, d3, d4, d5, d6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
