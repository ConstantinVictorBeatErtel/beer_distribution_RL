#!/usr/bin/env python3
"""Re-measure Check-5 parse-fail with grammar-constrained decoding ($0 local).

Uses product ``beer_distribution_rl.agents.llm``. No GRPO. No paid spend.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from beer_distribution_rl.agents.baselines import base_stock_order
from beer_distribution_rl.agents.ippo.trainer import IPPOConfig, build_env_config
from beer_distribution_rl.agents.llm import (
    AgentMemory,
    ConstrainedOrderDecoder,
    WeekRecord,
    observe_local,
    prompt_leak_report,
    serialize_prompt,
)
from beer_distribution_rl.env.core import BeerGameCore, Role


# Check 5 (preflight) post-hoc regex rates — same caps / seed / Y×A×prop×AR(1).
CHECK5_BEFORE: dict[str, dict[str, Any]] = {
    "inf": {
        "capacity_mult": None,
        "parse_attempts": 324,
        "parse_failures": 96,
        "parse_failure_rate": 0.2962962962962963,
        "by_role": None,  # not stratified in Check 5
    },
    "1.0mu": {
        "capacity_mult": 1.0,
        "parse_attempts": 352,
        "parse_failures": 138,
        "parse_failure_rate": 0.39204545454545453,
        "by_role": None,
    },
    "0.8mu": {
        "capacity_mult": 0.8,
        "parse_attempts": 344,
        "parse_failures": 126,
        "parse_failure_rate": 0.36627906976744184,
        "by_role": None,
    },
}


def _cap_label(capacity_mult: float | None) -> str:
    if capacity_mult is None:
        return "inf"
    if abs(capacity_mult - 1.0) < 1e-9:
        return "1.0mu"
    if abs(capacity_mult - 0.8) < 1e-9:
        return "0.8mu"
    return f"{capacity_mult}mu"


def run_basestock_episode(cfg: IPPOConfig, S: int = 30) -> float:
    env_cfg = replace(build_env_config(cfg), signaling_enabled=False, regime="A")
    core = BeerGameCore(env_cfg)
    core.reset(cfg.seed)
    levels = {r: S for r in core.roles}
    total = 0.0
    done = False
    while not done:
        orders = {
            r: base_stock_order(core._states[r], levels[r], order_cap=env_cfg.order_cap)
            for r in core.roles
        }
        _, _, done, info = core.step(orders)
        total += float(info.system_cost)
    return total


def round_trip_examples(seed: int = 0) -> list[dict[str, Any]]:
    """Three real logged observations → prompt → constrained parse."""
    cfg = IPPOConfig(
        regime="A",
        topology="y",
        capacity_mult=None,
        rationing="proportional",
        demand="ar1",
        seed=seed,
        horizon=52,
    )
    env_cfg = replace(build_env_config(cfg), signaling_enabled=False, regime="A")
    core = BeerGameCore(env_cfg)
    core.reset(seed)
    role = core.roles[0]
    mem = AgentMemory(role=role, role_name=core.role_names[role])
    out: list[dict[str, Any]] = []
    for week in range(3):
        obs = observe_local(core, role)
        prompt = serialize_prompt(
            mem,
            obs,
            order_cap=env_cfg.order_cap,
            holding=0.5,
            backlog_cost=1.0,
        )
        raw = '{"delta": 1}'
        from beer_distribution_rl.agents.llm.parser import parse_order_from_delta_text

        parsed = parse_order_from_delta_text(
            raw, int(obs["last_demand_or_order"]), order_cap=env_cfg.order_cap
        )
        out.append(
            {
                "week": week,
                "role": core.role_names[role],
                "obs_fields": {
                    "inventory": obs["inventory"],
                    "backlog": obs["backlog"],
                    "on_order": obs["on_order"],
                    "last_demand_or_order": obs["last_demand_or_order"],
                    "last_shipment_received": obs["last_shipment_received"],
                    "last_order_placed": obs["last_order_placed"],
                    "ship_pipeline": obs["ship_pipeline"],
                    "order_pipeline": obs["order_pipeline"],
                },
                "prompt": prompt,
                "raw_model_output": raw,
                "parsed_order": parsed,
                "leak_issues": prompt_leak_report(prompt, role, core),
            }
        )
        orders = {
            r: max(0, min(env_cfg.order_cap, int(core._states[r].last_demand_or_order)))
            for r in core.roles
        }
        orders[role] = int(parsed or 0)
        ship_before = int(core._states[role].last_shipment_received)
        _, _, _, info = core.step(orders)
        st = core._states[role]
        mem.append(
            WeekRecord(
                week=week,
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
    return out


def run_llm_episode(
    *,
    capacity_mult: float | None,
    seed: int,
    model: str,
    host: str,
    constrained: bool,
    max_parse_retries: int = 3,
    horizon: int = 52,
) -> dict[str, Any]:
    cfg = IPPOConfig(
        regime="A",
        topology="y",
        capacity_mult=capacity_mult,
        rationing="proportional",
        demand="ar1",
        seed=seed,
        horizon=horizon,
    )
    env_cfg = replace(build_env_config(cfg), signaling_enabled=False, regime="A")
    core = BeerGameCore(env_cfg)
    core.reset(seed)

    decoder = ConstrainedOrderDecoder(
        model=model,
        host=host,
        delta_max=8,
        order_cap=env_cfg.order_cap,
        max_parse_retries=max_parse_retries,
        constrained=constrained,
    )
    memories = {
        r: AgentMemory(role=r, role_name=core.role_names[r]) for r in core.roles
    }
    # Stratified counters: role and capacity×role
    by_role: dict[str, dict[str, int]] = {
        core.role_names[r]: {"attempts": 0, "failures": 0} for r in core.roles
    }
    leak_hits = 0
    token_est = 0
    system_cost = 0.0
    valid_weeks = 0
    fallback_weeks = 0
    done = False
    t0 = time.time()
    cap_key = _cap_label(capacity_mult)

    while not done:
        orders: dict[Role, int] = {}
        pre_obs = {r: observe_local(core, r) for r in core.roles}
        week_t = int(core.t)
        week_fallback = 0
        for r in core.roles:
            name = core.role_names[r]
            holding = float(core._cost(r).holding)
            bcost = float(core._cost(r).backlog)
            prompt = serialize_prompt(
                memories[r],
                pre_obs[r],
                order_cap=env_cfg.order_cap,
                holding=holding,
                backlog_cost=bcost,
                delta_max=8,
            )
            # Legacy Check-5 prompt shape when unconstrained (ORDER: absolute).
            if not constrained:
                prompt = _legacy_order_prompt(
                    memories[r],
                    pre_obs[r],
                    order_cap=env_cfg.order_cap,
                    holding=holding,
                    backlog_cost=bcost,
                )
            token_est += max(1, len(prompt) // 4)
            if prompt_leak_report(prompt, r, core):
                leak_hits += 1
            stats_key = f"{cap_key}|{name}"
            result = decoder.sample_order(
                prompt,
                int(pre_obs[r]["last_demand_or_order"]),
                stats_key=stats_key,
            )
            # Per-attempt accounting mirrors decoder.stats (failed retries count).
            by_role[name]["attempts"] += result.n_attempts
            if result.parse_ok:
                by_role[name]["failures"] += max(0, result.n_attempts - 1)
            else:
                by_role[name]["failures"] += result.n_attempts
            if result.used_fallback:
                week_fallback += 1
            orders[r] = result.order

        ship_before = {r: int(core._states[r].last_shipment_received) for r in core.roles}
        _, _, done, info = core.step(orders)
        system_cost += float(info.system_cost)
        valid_weeks += 1
        if week_fallback:
            fallback_weeks += 1
        if week_t % 4 == 0 or done:
            print(
                f"  week {week_t:02d}/{env_cfg.horizon} cost_so_far={system_cost:.1f} "
                f"parse_fail_rate={decoder.stats.rate:.3f} "
                f"week_fallback={week_fallback}",
                flush=True,
            )
        for r in core.roles:
            st = core._states[r]
            memories[r].append(
                WeekRecord(
                    week=core.t - 1,
                    demand_or_incoming=int(pre_obs[r]["last_demand_or_order"]),
                    ship_in=ship_before[r],
                    ordered=int(orders[r]),
                    alloc_recv=int(st.last_shipment_received),
                    inventory=int(st.inventory),
                    backlog=int(st.backlog),
                    on_order=int(st.on_order),
                    ship_pipeline=list(pre_obs[r]["ship_pipeline"]),
                    order_pipeline=list(pre_obs[r]["order_pipeline"]),
                    local_cost=float(info.local_costs[r]),
                )
            )

    # Persistence check
    m0 = memories[core.roles[0]]
    persistence_ok = False
    if len(m0.history) >= 1:
        fake = AgentMemory(role=m0.role, role_name=m0.role_name, history=m0.history[:1])
        p = serialize_prompt(
            fake,
            observe_local(core, m0.role),
            order_cap=env_cfg.order_cap,
            holding=0.5,
            backlog_cost=1.0,
        )
        persistence_ok = f"ordered={m0.history[0].ordered}" in p

    bs = run_basestock_episode(cfg, S=30)
    by_role_rates = {
        k: v["failures"] / max(v["attempts"], 1) for k, v in by_role.items()
    }
    return {
        "capacity_mult": capacity_mult,
        "cap_label": cap_key,
        "constrained": constrained,
        "seed": seed,
        "horizon": env_cfg.horizon,
        "n_roles": len(core.roles),
        "system_cost_llm": system_cost,
        "system_cost_basestock_S30": bs,
        "cost_ratio_vs_basestock": system_cost / bs if bs > 0 else None,
        "parse_stats": decoder.stats.as_dict(),
        "by_role": {
            k: {**v, "rate": by_role_rates[k]} for k, v in by_role.items()
        },
        "valid_order_weeks": valid_weeks,
        "weeks_with_fallback": fallback_weeks,
        "leak_prompt_hits": leak_hits,
        "persistence_ok": persistence_ok,
        "est_prompt_tokens_episode": token_est,
        "wall_sec": time.time() - t0,
        "resampling_multiplier": decoder.stats.resampling_multiplier(),
    }


def _legacy_order_prompt(
    memory: AgentMemory,
    obs: dict[str, Any],
    *,
    order_cap: int,
    holding: float,
    backlog_cost: float,
) -> str:
    """Check-5 absolute ORDER: prompt (for before/after legacy re-run)."""
    lines = [
        f"You are the {memory.role_name} in a beer distribution supply chain.",
        "Goal: minimize YOUR local cost = holding*inventory + backlog*backlog_units.",
        f"Your costs: holding={holding}, backlog={backlog_cost}.",
        f"Each week order an integer quantity in [0, {order_cap}] from your upstream supplier.",
        "You do NOT see other agents' private inventories or true consumer demand "
        "(unless you are a retailer facing customers).",
        "Your only action is the order. Do not broadcast signals.",
        "",
        "Output grammar (strict): reply with exactly one line matching:",
        "ORDER: <integer>",
        "Example valid reply:",
        "ORDER: 7",
        "No other text.",
        "",
        f"Current week t={obs['t']} (0-indexed before step; upcoming decision).",
        f"inventory={obs['inventory']}",
        f"backlog={obs['backlog']}",
        f"on_order={obs['on_order']}",
        f"inventory_position={obs['inventory_position']}",
        f"last_demand_or_order={obs['last_demand_or_order']}",
        f"last_shipment_received={obs['last_shipment_received']}",
        f"last_order_placed={obs['last_order_placed']}",
        f"ship_pipeline={obs['ship_pipeline']}",
        f"order_pipeline={obs['order_pipeline']}",
        "",
        "Own history (prior weeks):",
    ]
    if not memory.history:
        lines.append("(none — first week)")
    else:
        for rec in memory.history:
            lines.append(
                f"  week={rec.week}: demand_or_incoming={rec.demand_or_incoming}, "
                f"ship_in={rec.ship_in}, ordered={rec.ordered}, "
                f"alloc_recv={rec.alloc_recv}, "
                f"inv={rec.inventory}, backlog={rec.backlog}, cost={rec.local_cost:.2f}"
            )
    lines.append("")
    lines.append("ORDER:")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--horizon", type=int, default=52)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/diagnostics/llm_text_io_smoke.json"),
    )
    ap.add_argument("--roundtrip-only", action="store_true")
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="Also re-run Check-5 unconstrained ORDER: regex path for by-role before.",
    )
    ap.add_argument("--constrained-only", action="store_true", default=False)
    args = ap.parse_args()

    examples = round_trip_examples(args.seed)
    if args.roundtrip_only:
        print(json.dumps({"round_trip_examples": examples}, indent=2)[:8000])
        return

    caps: list[float | None] = [None, 1.0, 0.8]
    constrained_eps = []
    for cap in caps:
        print(
            f"=== CONSTRAINED cap={cap} horizon={args.horizon} ===",
            flush=True,
        )
        ep = run_llm_episode(
            capacity_mult=cap,
            seed=args.seed,
            model=args.model,
            host=args.host,
            constrained=True,
            horizon=args.horizon,
        )
        constrained_eps.append(ep)
        slim = {k: v for k, v in ep.items() if k != "parse_stats"}
        print(json.dumps(slim, indent=2), flush=True)
        print("parse_stats:", json.dumps(ep["parse_stats"], indent=2), flush=True)

    legacy_eps: list[dict[str, Any]] = []
    if args.legacy:
        for cap in caps:
            print(f"=== LEGACY(regex) cap={cap} horizon={args.horizon} ===", flush=True)
            ep = run_llm_episode(
                capacity_mult=cap,
                seed=args.seed,
                model=args.model,
                host=args.host,
                constrained=False,
                horizon=args.horizon,
            )
            legacy_eps.append(ep)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in ep.items()
                        if k in ("cap_label", "by_role", "parse_stats", "wall_sec")
                    },
                    indent=2,
                ),
                flush=True,
            )

    payload = {
        "model": args.model,
        "branch_base_sha_note": "see artifacts/diagnostics/llm_text_io.md",
        "check5_before": CHECK5_BEFORE,
        "round_trip_examples": [
            {k: v for k, v in ex.items() if k != "prompt"} for ex in examples
        ],
        "round_trip_prompts": [ex["prompt"] for ex in examples],
        "constrained_episodes": constrained_eps,
        "legacy_episodes": legacy_eps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
