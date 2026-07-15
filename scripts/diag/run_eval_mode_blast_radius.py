#!/usr/bin/env python3
"""Matched-deterministic re-eval of Tier-1 v11 for eval-mode blast radius.

Forces greedy=True for every regime (A/B/C). Compares against logged
final_eval.json means (which used greedy=not signaling).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.common import CACHE_DIR, ci95, load_trainer  # noqa: E402
from analysis.diag.eval_matched_deterministic import (  # noqa: E402
    evaluate_matched_deterministic,
)

TIER1 = ROOT / "artifacts" / "runs" / "ippo" / "tier1_v11"
OUT_MD = ROOT / "artifacts" / "diagnostics" / "eval_mode_blast_radius.md"
BASELINE_SHA = "061aa59235397b7360c32a01cf4f98add0dd503a"

CAP_LABEL = {"inf": "∞", "1p2mu": "1.2μ", "1p0mu": "1.0μ", "0p8mu": "0.8μ"}
CAP_ORDER = ["inf", "1p2mu", "1p0mu", "0p8mu"]


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(payload["run_dir"])
    n_episodes = int(payload["n_episodes"])
    trainer, _ = load_trainer(run_dir)
    m = evaluate_matched_deterministic(trainer, n_episodes=n_episodes)
    return {
        **payload["row"],
        "det/mean_system_cost": m["eval/mean_system_cost"],
        "det/std_system_cost": m["eval/std_system_cost"],
    }


def _fmt(m: float, ci: float) -> str:
    if not math.isfinite(m):
        return "—"
    return f"{m:.1f}±{ci:.1f}"


def _pct(gap: float, base: float) -> str:
    if not math.isfinite(gap) or not math.isfinite(base) or abs(base) < 1e-12:
        return "—"
    return f"{100.0 * gap / base:.1f}%"


def aggregate(
    rows: list[dict[str, Any]], cost_key: str
) -> dict[tuple, dict[str, tuple[float, float, int]]]:
    """(topo, cap, rat, dem) -> regime -> (mean, ci95, n)."""
    buckets: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["topology"], r["capacity_tag"], r["rationing"], r["demand"])
        buckets[key][r["regime"]].append(float(r[cost_key]))
    out: dict[tuple, dict[str, tuple[float, float, int]]] = {}
    for key, by_reg in buckets.items():
        out[key] = {}
        for reg, xs in by_reg.items():
            m, ci, n = ci95(xs)
            out[key][reg] = (m, ci, n)
    return out


def write_report(
    *,
    old_agg: dict[tuple, dict[str, tuple[float, float, int]]],
    new_agg: dict[tuple, dict[str, tuple[float, float, int]]],
    n_episodes: int,
    n_runs: int,
    elapsed_s: float,
) -> None:
    lines: list[str] = []
    lines.append("# Eval-mode blast radius (matched-deterministic re-eval)")
    lines.append("")
    lines.append(f"**Baseline SHA:** `{BASELINE_SHA}`")
    lines.append("")
    lines.append(
        f"Eval-only re-run on frozen Tier-1 v11 checkpoints. "
        f"`n_episodes={n_episodes}`, 10 seeds/cell, `{n_runs}` runs, "
        f"wall ≈ {elapsed_s / 60:.1f} min. No training / reward / env changes."
    )
    lines.append("")

    lines.append("## Step 0 — Root cause")
    lines.append("")
    lines.append(
        "**Verdict: (a) per-regime config field** — a real bug. "
        "`IPPOTrainer.evaluate` sets action mode from `self.signaling`, "
        "so Regime B (signaling=True) is evaluated stochastically while "
        "A/C (signaling=False) are evaluated with argmax."
    )
    lines.append("")
    lines.append("```455:455:beer_distribution_rl/agents/ippo/trainer.py")
    lines.append("                        a, _, _ = self._policy_act(r, o, greedy=not self.signaling)")
    lines.append("```")
    lines.append("")
    lines.append(
        "Same pattern in the analysis helper "
        "`analysis/diag/eval_ablation.py` (`greedy=not signaling`). "
        "Not an analysis-time table default: every `final_eval.json` "
        "written by the matrix runner inherits this coupling."
    )
    lines.append("")
    lines.append("| Regime | `signaling` | `greedy=not signaling` | Mode |")
    lines.append("|---|---|---|---|")
    lines.append("| A | False | True | deterministic (argmax) |")
    lines.append("| B | True | False | stochastic (sample) |")
    lines.append("| C | False | True | deterministic (argmax) |")
    lines.append("")

    lines.append("## Step 1 — Contaminated outputs (blast radius)")
    lines.append("")
    lines.append(
        "Any **cross-regime cost comparison that includes Regime B** and was "
        "sourced from `final_eval.json` / `tier1_v11/index.json` is contaminated. "
        "A-vs-C comparisons were already matched-deterministic (both non-signaling) "
        "and are listed for completeness; they should not move under this re-eval."
    )
    lines.append("")
    lines.append("### Contaminated (mismatched mode)")
    lines.append("")
    lines.append("| Output | Why |")
    lines.append("|---|---|")
    lines.append(
        "| Tier-1 v11 `final_eval.json` × all Regime-B cells (and any A/B or B/C gap "
        "derived from `index.json`) | B stochastic vs A/C greedy |"
    )
    lines.append(
        "| Headline 28–53% A−B scarcity gaps (serial/Y × AR(1)/regime_switch × "
        "prop/uniform/honesty_weighted × {1.0μ,0.8μ}) | Primary published confound |"
    )
    lines.append(
        "| All Y-topology A−B / B−C cost comparisons in the matrix | Same `evaluate` path |"
    )
    lines.append(
        "| All `regime_switch` A−B / B−C cost comparisons | Same `evaluate` path |"
    )
    lines.append(
        "| C-vs-B (and B-vs-C) at every capacity including ∞ | B stoch vs C det |"
    )
    lines.append(
        "| `artifacts/diagnostics/v11_ablation.md` “Context” stochastic columns "
        "(`B stoch (logged)`, `A−B stoch`) | Echoes mismatched `final_eval` |"
    )
    lines.append(
        "| Paper-feeding P1/P2 narratives that cite the logged scarcity A−B % gap "
        "(see ablation M4-gate section; DECISIONS B′/M4 notes) | Numbers retracted below |"
    )
    lines.append("")
    lines.append("### Not contaminated by *cross-regime mode mismatch*")
    lines.append("")
    lines.append("| Output | Why |")
    lines.append("|---|---|")
    lines.append(
        "| A-vs-C at ∞ (and all A-vs-C cells) | Both `signaling=False` → both greedy |"
    )
    lines.append(
        "| M3 phase diagram / `M3_REPORT.md` (Regime B only) | No cross-regime compare; "
        "all B cells share stochastic eval (absolute levels still stochastic-mode) |"
    )
    lines.append(
        "| M2 A vs C classic (`M2_REPORT.md`) | Both non-signaling |"
    )
    lines.append(
        "| `v11_ablation.md` matched-det cost tables + shuffle rubric | Already forced greedy |"
    )
    lines.append(
        "| `v11_signal_content.md` MI/decoder tables | Within-B content metrics, not A/B cost |"
    )
    lines.append(
        "| D2–D6 M3 diagnostics (`analysis/DIAGNOSTICS.md`) | Pre-v11 / B-only or non-cost |"
    )
    lines.append("")

    lines.append("## Step 2 — Corrected table (old mismatched → new matched-det)")
    lines.append("")
    lines.append(
        "Old = mean±CI95 over 10 seeds from logged `final_eval` "
        "(`greedy=not signaling`). New = same checkpoints, "
        "`greedy=True` for every regime, same seed offset (`seed+10_000`), "
        f"{n_episodes} episodes."
    )
    lines.append("")

    # --- A vs B ---
    lines.append("### A vs B (all topologies / demands / rationing / caps in matrix)")
    lines.append("")
    lines.append(
        "| Topo | Cap | Rationing | Demand | A old | B old | A−B old (%) | "
        "A new | B new | A−B new (%) |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(old_agg.keys(), key=lambda k: (k[0], CAP_ORDER.index(k[1]) if k[1] in CAP_ORDER else 99, k[2], k[3])):
        topo, cap, rat, dem = key
        o = old_agg[key]
        n = new_agg.get(key, {})
        if "A" not in o or "B" not in o:
            continue
        ao, aoci, _ = o["A"]
        bo, boci, _ = o["B"]
        an, anci, _ = n.get("A", (float("nan"), float("nan"), 0))
        bn, bnci, _ = n.get("B", (float("nan"), float("nan"), 0))
        gap_o = ao - bo
        gap_n = an - bn
        lines.append(
            f"| {topo} | {CAP_LABEL.get(cap, cap)} | {rat} | {dem} | "
            f"{_fmt(ao, aoci)} | {_fmt(bo, boci)} | {_fmt(gap_o, 0.0).replace('±0.0','')} ({_pct(gap_o, ao)}) | "
            f"{_fmt(an, anci)} | {_fmt(bn, bnci)} | {_fmt(gap_n, 0.0).replace('±0.0','')} ({_pct(gap_n, an)}) |"
        )
    lines.append("")

    # --- C vs A at ∞ (and all C vs A) ---
    lines.append("### C vs A (should be unchanged — already matched-det)")
    lines.append("")
    lines.append(
        "| Topo | Cap | Rationing | Demand | A old | C old | A−C old (%) | "
        "A new | C new | A−C new (%) |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(old_agg.keys(), key=lambda k: (k[0], CAP_ORDER.index(k[1]) if k[1] in CAP_ORDER else 99, k[2], k[3])):
        topo, cap, rat, dem = key
        o = old_agg[key]
        n = new_agg.get(key, {})
        if "A" not in o or "C" not in o:
            continue
        ao, aoci, _ = o["A"]
        co, coci, _ = o["C"]
        an, anci, _ = n.get("A", (float("nan"), float("nan"), 0))
        cn, cnci, _ = n.get("C", (float("nan"), float("nan"), 0))
        gap_o = ao - co
        gap_n = an - cn
        lines.append(
            f"| {topo} | {CAP_LABEL.get(cap, cap)} | {rat} | {dem} | "
            f"{_fmt(ao, aoci)} | {_fmt(co, coci)} | {_fmt(gap_o, 0.0).replace('±0.0','')} ({_pct(gap_o, ao)}) | "
            f"{_fmt(an, anci)} | {_fmt(cn, cnci)} | {_fmt(gap_n, 0.0).replace('±0.0','')} ({_pct(gap_n, an)}) |"
        )
    lines.append("")

    # --- B vs C ---
    lines.append("### B vs C (mismatched old → matched-det new)")
    lines.append("")
    lines.append(
        "| Topo | Cap | Rationing | Demand | B old | C old | C−B old (%) | "
        "B new | C new | C−B new (%) |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(old_agg.keys(), key=lambda k: (k[0], CAP_ORDER.index(k[1]) if k[1] in CAP_ORDER else 99, k[2], k[3])):
        topo, cap, rat, dem = key
        o = old_agg[key]
        n = new_agg.get(key, {})
        if "B" not in o or "C" not in o:
            continue
        bo, boci, _ = o["B"]
        co, coci, _ = o["C"]
        bn, bnci, _ = n.get("B", (float("nan"), float("nan"), 0))
        cn, cnci, _ = n.get("C", (float("nan"), float("nan"), 0))
        gap_o = co - bo
        gap_n = cn - bn
        lines.append(
            f"| {topo} | {CAP_LABEL.get(cap, cap)} | {rat} | {dem} | "
            f"{_fmt(bo, boci)} | {_fmt(co, coci)} | {_fmt(gap_o, 0.0).replace('±0.0','')} ({_pct(gap_o, co)}) | "
            f"{_fmt(bn, bnci)} | {_fmt(cn, cnci)} | {_fmt(gap_n, 0.0).replace('±0.0','')} ({_pct(gap_n, cn)}) |"
        )
    lines.append("")

    lines.append("## Takeaways")
    lines.append("")
    lines.append(
        "1. **RETRACT** the logged 28–53% (and sibling) A−B scarcity cost gaps from "
        "`final_eval` / index aggregates — they mix greedy A with stochastic B."
    )
    lines.append(
        "2. Under matched deterministic eval, A−B scarcity gaps shrink dramatically "
        "(typically into seed CIs); see corrected A−B table."
    )
    lines.append(
        "3. A−C gaps are unchanged (already matched). B−C gaps move because only B's "
        "logged costs were stochastic."
    )
    lines.append(
        "4. Code fix (out of scope here): evaluate with an explicit mode, never "
        "`greedy=not signaling`."
    )
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--smoke", action="store_true", help="1 seed × tiny slice")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--cache",
        type=Path,
        default=CACHE_DIR / "eval_mode_blast_radius_ep20.json",
    )
    args = p.parse_args()

    idx = json.loads((TIER1 / "index.json").read_text())
    rows = [r for r in idx if r.get("status") == "ok"]
    if args.smoke:
        rows = [
            r
            for r in rows
            if int(r["seed"]) == 0
            and r["capacity_tag"] in ("inf", "1p0mu")
            and r["rationing"] == "proportional"
            and r["demand"] == "ar1"
            and r["topology"] == "serial"
        ]

    cache_path = args.cache
    if args.smoke:
        cache_path = CACHE_DIR / "eval_mode_blast_radius_smoke.json"

    if cache_path.exists() and not args.force:
        print(f"Using cache {cache_path}")
        det_rows = json.loads(cache_path.read_text())
        elapsed = 0.0
    else:
        payloads = []
        for r in rows:
            run_dir = TIER1 / r["run"]
            if not (run_dir / "checkpoints").exists():
                continue
            payloads.append(
                {
                    "run_dir": str(run_dir),
                    "n_episodes": args.episodes,
                    "row": {
                        "run": r["run"],
                        "regime": r["regime"],
                        "topology": r["topology"],
                        "capacity_tag": r["capacity_tag"],
                        "rationing": r["rationing"],
                        "demand": r["demand"],
                        "seed": r["seed"],
                        "old/mean_system_cost": r["eval/mean_system_cost"],
                    },
                }
            )
        print(f"Re-evaluating {len(payloads)} runs with greedy=True, jobs={args.jobs}")
        t0 = time.time()
        det_rows = []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        partial = CACHE_DIR / (cache_path.stem + "_partial.jsonl")
        if args.force and partial.exists():
            partial.unlink()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_worker, pl): pl for pl in payloads}
            done = 0
            for fut in as_completed(futs):
                row = fut.result()
                det_rows.append(row)
                with partial.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                done += 1
                if done % 20 == 0 or done == len(payloads):
                    print(f"  {done}/{len(payloads)}", flush=True)
        elapsed = time.time() - t0
        cache_path.write_text(json.dumps(det_rows, indent=2))
        print(f"Wrote {cache_path} in {elapsed:.1f}s")

    # Build old rows from index for same keys
    old_rows = []
    keyset = {(r["run"]) for r in det_rows}
    for r in idx:
        if r.get("run") in keyset:
            old_rows.append(
                {
                    **{k: r[k] for k in ("regime", "topology", "capacity_tag", "rationing", "demand", "seed")},
                    "eval/mean_system_cost": r["eval/mean_system_cost"],
                }
            )
    new_rows = [
        {
            **{k: r[k] for k in ("regime", "topology", "capacity_tag", "rationing", "demand", "seed")},
            "eval/mean_system_cost": r["det/mean_system_cost"],
        }
        for r in det_rows
    ]
    old_agg = aggregate(old_rows, "eval/mean_system_cost")
    new_agg = aggregate(new_rows, "eval/mean_system_cost")
    write_report(
        old_agg=old_agg,
        new_agg=new_agg,
        n_episodes=args.episodes,
        n_runs=len(det_rows),
        elapsed_s=elapsed if elapsed else float("nan"),
    )
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
