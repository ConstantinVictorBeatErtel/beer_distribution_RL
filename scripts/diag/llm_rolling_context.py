#!/usr/bin/env python3
"""Rolling-context diagnostic: tokens @ W=8, T=52 persistence, corrected budget.

Inference-only arithmetic + local env rollouts. No GRPO. No paid spend ($0).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from beer_distribution_rl.agents.baselines import base_stock_order
from beer_distribution_rl.agents.ippo.trainer import IPPOConfig, build_env_config
from beer_distribution_rl.agents.llm import (
    DEFAULT_ROLLING_WINDOW,
    AgentMemory,
    WeekRecord,
    estimate_prompt_tokens,
    observe_local,
    prompt_leak_report,
    serialize_prompt,
)
from beer_distribution_rl.env.core import BeerGameCore


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "artifacts" / "diagnostics" / "llm_rolling_context.json"

# Final parse-fail from artifacts/diagnostics/llm_text_io.md (schema decode).
LLM_TEXT_IO_PARSE_FAIL_RATE = 0.0  # 0/780 pooled
RESAMPLING_FACTOR = 1.0 / (1.0 - LLM_TEXT_IO_PARSE_FAIL_RATE)

# Check-7 GRPO plan assumptions (order-only after channel drop).
CELLS = 9  # Y × {∞, 1.0μ, 0.8μ} × prop × 3 seeds
UPDATES = 50  # lean schedule that made naive W8 fit ~$100
GROUP_SIZE = 4
N_ROLES = 5  # Y topology
T = 52
COMPLETION_TOKENS = 8  # constrained {"delta": k} ~ few tokens; keep audit parity
THROUGHPUT_TOK_S = 400  # blended 4090
RATE_PER_HR = 0.50  # mid of spec $0.30–0.70
BUDGET_CAP = 250.0
QWEN_CTX = 32_768


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        )
    except Exception:
        return "unknown"


def run_persistence_episode(
    *,
    seed: int = 0,
    window: int = DEFAULT_ROLLING_WINDOW,
    horizon: int = T,
) -> dict[str, Any]:
    cfg = IPPOConfig(
        regime="A",
        topology="y",
        capacity_mult=None,
        rationing="proportional",
        demand="ar1",
        seed=seed,
        horizon=horizon,
    )
    env_cfg = replace(build_env_config(cfg), signaling_enabled=False, regime="A")
    core = BeerGameCore(env_cfg)
    core.reset(seed)
    role = core.roles[0]
    mem = AgentMemory(
        role=role, role_name=core.role_names[role], window=window
    )
    levels = {r: 30 for r in core.roles}
    tokens_by_week: list[int] = []
    hist_lens: list[int] = []
    persistence_misses: list[int] = []
    leak_hits = 0
    prompts_at_w8: list[str] = []

    for week in range(horizon):
        obs = observe_local(core, role)
        prompt = serialize_prompt(
            mem,
            obs,
            order_cap=env_cfg.order_cap,
            holding=0.5,
            backlog_cost=1.0,
        )
        ntok = estimate_prompt_tokens(prompt)
        tokens_by_week.append(ntok)
        hist_lens.append(len(mem.windowed_history()))
        if week >= 8:
            prompts_at_w8.append(prompt)
        leaks = prompt_leak_report(prompt, role, core)
        leak_hits += len(leaks)

        if week >= 1:
            prev = mem.history[week - 1]
            ok = (
                f"week={prev.week}:" in prompt
                and f"ordered={prev.ordered}" in prompt
                and f"demand_or_incoming={prev.demand_or_incoming}" in prompt
                and f"alloc_recv={prev.alloc_recv}" in prompt
                and f"backlog={prev.backlog}" in prompt
            )
            if not ok:
                persistence_misses.append(week)

        ship_before = int(core._states[role].last_shipment_received)
        orders = {
            r: base_stock_order(
                core._states[r], levels[r], order_cap=env_cfg.order_cap
            )
            for r in core.roles
        }
        _, _, done, info = core.step(orders)
        st = core._states[role]
        mem.append(
            WeekRecord(
                week=core.t - 1,
                demand_or_incoming=int(obs["last_demand_or_order"]),
                ship_in=ship_before,
                ordered=int(orders[role]),
                alloc_recv=int(st.last_shipment_received),
                inventory=int(st.inventory),
                backlog=int(st.backlog),
                on_order=int(st.on_order),
                ship_pipeline=list(obs["ship_pipeline"]),
                order_pipeline=list(obs["order_pipeline"]),
                local_cost=float(info.local_costs[role]),
            )
        )
        if done:
            break

    # Steady-state = weeks with a full W-week history in the prompt
    steady = [t for t, h in zip(tokens_by_week, hist_lens) if h == window]
    return {
        "seed": seed,
        "role": core.role_names[role],
        "window": window,
        "horizon": horizon,
        "n_weeks_run": len(tokens_by_week),
        "persistence_ok": len(persistence_misses) == 0,
        "persistence_misses": persistence_misses,
        "leak_hits": leak_hits,
        "tokens_by_week": tokens_by_week,
        "windowed_lens": hist_lens,
        "tokens_week0": tokens_by_week[0] if tokens_by_week else None,
        "tokens_per_prompt_w8_mean": (
            sum(steady) / len(steady) if steady else None
        ),
        "tokens_per_prompt_w8_max": max(steady) if steady else None,
        "tokens_per_prompt_mean_all_weeks": sum(tokens_by_week) / len(tokens_by_week),
        "tokens_per_prompt_max_all_weeks": max(tokens_by_week),
        "fits_qwen_32k_with_margin": max(tokens_by_week) < QWEN_CTX * 0.25,
        "example_prompt_tail_w8": (
            "\n".join(prompts_at_w8[0].splitlines()[-14:]) if prompts_at_w8 else None
        ),
    }


def project_grpo_budget(tokens_per_prompt_mean: float) -> dict[str, Any]:
    """tokens/week × T × rollouts × updates × cells, × 1/(1−p), @ 4090 rates."""
    tok_per_episode = N_ROLES * T * (tokens_per_prompt_mean + COMPLETION_TOKENS)
    tok_naive = CELLS * UPDATES * GROUP_SIZE * tok_per_episode
    tok_corrected = tok_naive * RESAMPLING_FACTOR
    gpu_h_naive = tok_naive / THROUGHPUT_TOK_S / 3600.0
    gpu_h_corrected = tok_corrected / THROUGHPUT_TOK_S / 3600.0
    cost_naive = gpu_h_naive * RATE_PER_HR
    cost_corrected = gpu_h_corrected * RATE_PER_HR
    return {
        "cells": CELLS,
        "updates": UPDATES,
        "group_size_G": GROUP_SIZE,
        "n_roles": N_ROLES,
        "T": T,
        "tokens_per_prompt_mean": tokens_per_prompt_mean,
        "completion_tokens": COMPLETION_TOKENS,
        "tok_per_episode_all_roles": tok_per_episode,
        "parse_fail_rate_p": LLM_TEXT_IO_PARSE_FAIL_RATE,
        "resampling_factor_1_over_1_minus_p": RESAMPLING_FACTOR,
        "tokens_naive": tok_naive,
        "tokens_corrected": tok_corrected,
        "throughput_tok_s": THROUGHPUT_TOK_S,
        "rate_per_hr": RATE_PER_HR,
        "gpu_h_naive": gpu_h_naive,
        "gpu_h_corrected": gpu_h_corrected,
        "cost_usd_naive": cost_naive,
        "cost_usd_corrected": cost_corrected,
        "budget_cap_usd": BUDGET_CAP,
        "fits_budget_with_margin": cost_corrected < BUDGET_CAP * 0.6,
        "margin_usd": BUDGET_CAP - cost_corrected,
        "hardware": "4090 (projected)",
        "note": "PROJECTION ONLY — not a spend. No GRPO run.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=DEFAULT_ROLLING_WINDOW)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sha = git_sha()
    print(f"SHA={sha}  W={args.window}  ($0, no GRPO)", flush=True)
    ep = run_persistence_episode(seed=args.seed, window=args.window)
    # Use mean over all weeks (matches Check-7 "mean tokens/role/week") —
    # early weeks are shorter; steady W8 max is reported separately for ctx fit.
    mean_tok = float(ep["tokens_per_prompt_mean_all_weeks"])
    # Also compute budget with steady-state W8 mean (conservative upper for spend)
    steady_mean = ep["tokens_per_prompt_w8_mean"] or mean_tok
    budget_mean = project_grpo_budget(mean_tok)
    budget_steady = project_grpo_budget(float(steady_mean))

    # Naive ~$100 used the audit's ~600 tok/week estimate; recompute that baseline
    # for the "naive vs corrected" narrative.
    budget_audit_600 = project_grpo_budget(600.0)

    out = {
        "sha": sha,
        "window_default": DEFAULT_ROLLING_WINDOW,
        "window_used": args.window,
        "spend_usd": 0,
        "grpo": False,
        "episode": ep,
        "budget_from_measured_mean_all_weeks": budget_mean,
        "budget_from_measured_steady_w8_mean": budget_steady,
        "budget_from_audit_600_tok_estimate": budget_audit_600,
        "qwen_context_tokens": QWEN_CTX,
        "resampling_source": (
            "artifacts/diagnostics/llm_text_io.md final pooled parse-fail "
            f"p={LLM_TEXT_IO_PARSE_FAIL_RATE} ⇒ 1/(1-p)={RESAMPLING_FACTOR:.2f}"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "persistence_ok": ep["persistence_ok"],
        "leak_hits": ep["leak_hits"],
        "tokens_per_prompt_w8_mean": ep["tokens_per_prompt_w8_mean"],
        "tokens_per_prompt_w8_max": ep["tokens_per_prompt_w8_max"],
        "tokens_mean_all_weeks": mean_tok,
        "cost_usd_naive_audit600": round(budget_audit_600["cost_usd_naive"], 2),
        "cost_usd_corrected_measured_mean": round(budget_mean["cost_usd_corrected"], 2),
        "cost_usd_corrected_steady_w8": round(budget_steady["cost_usd_corrected"], 2),
        "resampling_factor": RESAMPLING_FACTOR,
        "fits_32k": ep["fits_qwen_32k_with_margin"],
        "fits_250": budget_steady["fits_budget_with_margin"],
        "wrote": str(OUT_JSON),
    }, indent=2))


if __name__ == "__main__":
    main()
