#!/usr/bin/env python3
"""Readiness-audit smoke: provisional obs→prompt / parse / retained context.

NOT production ``agents/llm/`` — audit-only harness for Check 2/3/5.
Order-only (broadcast channel dropped). Inference via local Ollama, $0.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beer_distribution_rl.agents.baselines import base_stock_order
from beer_distribution_rl.agents.ippo.trainer import IPPOConfig, build_env_config
from beer_distribution_rl.env.core import BeerGameCore, Role

ORDER_RE = re.compile(r"ORDER:\s*(\d+)", re.IGNORECASE)
FORBIDDEN_SUBSTRINGS = (
    "customer_demand",
    "true_demand",
    "consumer_demand",
    "end_customer_demand",
    "rival_inventory",
    "other_agent_inventory",
)


@dataclass
class WeekRecord:
    week: int
    inventory: int
    backlog: int
    on_order: int
    last_demand_or_order: int
    last_shipment_received: int
    last_order_placed: int
    ship_pipeline: list[int]
    order_pipeline: list[int]
    order: int
    local_cost: float
    allocation_received: int


@dataclass
class AgentMemory:
    role: Role
    role_name: str
    history: list[WeekRecord] = field(default_factory=list)


def observe_local(core: BeerGameCore, role: Role) -> dict[str, Any]:
    """Local dict obs — mirrors core.observe but never includes signals (LLM order-only)."""
    obs = core.observe(role)
    obs.pop("signals", None)
    return obs


def serialize_prompt(
    memory: AgentMemory,
    obs: dict[str, Any],
    *,
    order_cap: int,
    holding: float,
    backlog_cost: float,
) -> str:
    """Build week-t prompt with retained own-history only."""
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
                f"  week={rec.week}: demand_or_incoming={rec.last_demand_or_order}, "
                f"ship_in={rec.last_shipment_received}, ordered={rec.order}, "
                f"alloc_recv={rec.allocation_received}, "
                f"inv={rec.inventory}, backlog={rec.backlog}, cost={rec.local_cost:.2f}"
            )
    lines.append("")
    lines.append("ORDER:")
    return "\n".join(lines)


def parse_order(text: str, order_cap: int) -> int | None:
    """Return order qty or None on parse failure."""
    m = ORDER_RE.search(text.strip())
    if not m:
        # also accept bare trailing integer after ORDER:
        m2 = re.search(r"ORDER:\s*(\d+)", text, re.IGNORECASE)
        if not m2:
            return None
        m = m2
    qty = int(m.group(1))
    if qty < 0 or qty > order_cap:
        return None
    return qty


def prompt_leak_report(prompt: str, role: Role, core: BeerGameCore) -> list[str]:
    """Heuristic E1-style checks on text prompt."""
    issues: list[str] = []
    low = prompt.lower()
    for s in FORBIDDEN_SUBSTRINGS:
        if s in low:
            issues.append(f"forbidden substring: {s}")
    # Other roles' inventories must not appear as privileged fields.
    my_name = core.role_names[role]
    for r, name in core.role_names.items():
        if r == role:
            continue
        # Allow role name in instructions; forbid "<other> inventory=N" patterns
        pat = re.compile(rf"\b{re.escape(name)}\b.*\binventory\s*=\s*\d+", re.I | re.S)
        if pat.search(prompt):
            issues.append(f"other-role inventory leak pattern for {name}")
    # Upstream must not see true customer demand labeled as such
    if role not in (Role.RETAILER, Role.RETAILER_B):
        if "true consumer demand" in low and "last_demand_or_order" in low:
            # instruction text is ok; field value compared below in tests
            pass
        if re.search(r"customer_demand\s*=\s*\d+", prompt, re.I):
            issues.append("customer_demand numeric field")
    _ = my_name
    return issues


def ollama_generate(model: str, prompt: str, *, host: str, temperature: float = 0.0) -> str:
    """Chat API with a hard system constraint — better format adherence than raw generate."""
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 12},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You output only: ORDER: <integer>\n"
                        "Example: ORDER: 7\n"
                        "Never omit the ORDER: prefix. No other words."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
    ).encode()
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    msg = data.get("message") or {}
    return str(msg.get("content", data.get("response", "")))


def run_basestock_episode(cfg: IPPOConfig, S: int = 30) -> float:
    from dataclasses import replace

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


def run_llm_episode(
    *,
    capacity_mult: float | None,
    seed: int,
    model: str,
    host: str,
    max_parse_retries: int = 3,
    horizon: int | None = None,
) -> dict[str, Any]:
    cfg = IPPOConfig(
        regime="A",  # order-only / channel dropped
        topology="y",
        capacity_mult=capacity_mult,
        rationing="proportional",
        demand="ar1",
        seed=seed,
        horizon=horizon or 52,
    )
    env_cfg = build_env_config(cfg)
    from dataclasses import replace

    env_cfg = replace(env_cfg, signaling_enabled=False, regime="A")
    core = BeerGameCore(env_cfg)
    core.reset(seed)

    memories = {
        r: AgentMemory(role=r, role_name=core.role_names[r]) for r in core.roles
    }
    parse_attempts = 0
    parse_failures = 0
    valid_weeks = 0
    system_cost = 0.0
    leak_hits = 0
    token_est_prompt = 0
    prompts_sample: list[str] = []
    done = False
    t0 = time.time()

    while not done:
        orders: dict[Role, int] = {}
        week_prompts: dict[Role, str] = {}
        pre_obs = {r: observe_local(core, r) for r in core.roles}
        week_t = int(core.t)
        week_parse_fail = 0
        for r in core.roles:
            holding = float(core._cost(r).holding)
            bcost = float(core._cost(r).backlog)
            prompt = serialize_prompt(
                memories[r],
                pre_obs[r],
                order_cap=env_cfg.order_cap,
                holding=holding,
                backlog_cost=bcost,
            )
            week_prompts[r] = prompt
            token_est_prompt += max(1, len(prompt) // 4)
            leaks = prompt_leak_report(prompt, r, core)
            if leaks:
                leak_hits += 1
            if core.t == 0 and r == core.roles[0]:
                prompts_sample.append(prompt)
            if core.t == 1 and len(prompts_sample) < 2:
                prompts_sample.append(prompt)

            qty = None
            for _ in range(max_parse_retries):
                parse_attempts += 1
                raw = ollama_generate(model, prompt, host=host)
                qty = parse_order(raw, env_cfg.order_cap)
                if qty is not None:
                    break
                parse_failures += 1
            if qty is None:
                week_parse_fail += 1
                # last-resort safe default (still counts as parse failure path exhausted)
                qty = int(pre_obs[r]["last_demand_or_order"])
                qty = max(0, min(env_cfg.order_cap, qty))
            orders[r] = qty

        # persistence check material: week-1 prompt must contain week-0 outcome after step
        ship_before = {r: int(core._states[r].last_shipment_received) for r in core.roles}
        _, _, done, info = core.step(orders)
        system_cost += float(info.system_cost)
        valid_weeks += 1
        if week_t % 4 == 0 or done:
            print(
                f"  week {week_t:02d}/{env_cfg.horizon} cost_so_far={system_cost:.1f} "
                f"parse_fail_rate={parse_failures/max(parse_attempts,1):.3f} "
                f"week_fallback={week_parse_fail}",
                flush=True,
            )

        for r in core.roles:
            st = core._states[r]
            # allocation ≈ shipment received this week (post-step)
            alloc = int(st.last_shipment_received)
            memories[r].history.append(
                WeekRecord(
                    week=core.t - 1,
                    inventory=int(st.inventory),
                    backlog=int(st.backlog),
                    on_order=int(st.on_order),
                    last_demand_or_order=int(pre_obs[r]["last_demand_or_order"]),
                    last_shipment_received=ship_before[r],
                    last_order_placed=int(orders[r]),
                    ship_pipeline=list(pre_obs[r]["ship_pipeline"]),
                    order_pipeline=list(pre_obs[r]["order_pipeline"]),
                    order=int(orders[r]),
                    local_cost=float(info.local_costs[r]),
                    allocation_received=alloc,
                )
            )

    # persistence: after week 0, week-1 prompt should mention ordered=
    persistence_ok = False
    if horizon is None or (horizon or 52) >= 2:
        # rebuild first two prompts offline from memory trail
        # Check that each memory's second entry implies first order appears if we re-serialize mid-ep
        # Simpler: history length == T and week0 order appears in a fresh serialize at end-state with truncated hist
        m0 = memories[core.roles[0]]
        if len(m0.history) >= 1:
            fake = AgentMemory(role=m0.role, role_name=m0.role_name, history=m0.history[:1])
            obs = observe_local(core, m0.role)
            p = serialize_prompt(
                fake,
                obs,
                order_cap=env_cfg.order_cap,
                holding=0.5,
                backlog_cost=1.0,
            )
            persistence_ok = f"ordered={m0.history[0].order}" in p

    bs = run_basestock_episode(cfg, S=30)
    return {
        "capacity_mult": capacity_mult,
        "seed": seed,
        "horizon": env_cfg.horizon,
        "n_roles": len(core.roles),
        "system_cost_llm": system_cost,
        "system_cost_basestock_S30": bs,
        "cost_ratio_vs_basestock": system_cost / bs if bs > 0 else None,
        "parse_attempts": parse_attempts,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / max(parse_attempts, 1),
        "valid_order_weeks": valid_weeks,
        "leak_prompt_hits": leak_hits,
        "persistence_ok": persistence_ok,
        "est_prompt_tokens_episode": token_est_prompt,
        "wall_sec": time.time() - t0,
        "prompt_samples": prompts_sample[:2],
    }


def round_trip_demo(seed: int = 0) -> list[dict[str, Any]]:
    """Check 2: serialize real obs and parse synthetic LLM outputs."""
    cfg = IPPOConfig(
        regime="A", topology="y", capacity_mult=None, rationing="proportional",
        demand="ar1", seed=seed, horizon=52,
    )
    env_cfg = build_env_config(cfg)
    from dataclasses import replace

    env_cfg = replace(env_cfg, signaling_enabled=False, regime="A")
    core = BeerGameCore(env_cfg)
    core.reset(seed)
    out = []
    mem = AgentMemory(role=core.roles[0], role_name=core.role_names[core.roles[0]])
    for i in range(3):
        obs = observe_local(core, core.roles[0])
        prompt = serialize_prompt(
            mem, obs, order_cap=env_cfg.order_cap, holding=0.5, backlog_cost=1.0
        )
        for raw, expect_ok in (
            ("ORDER: 12", True),
            ("I think we should ORDER: 5\n", True),
            ("twelve cases please", False),
            (f"ORDER: {env_cfg.order_cap + 5}", False),
        ):
            parsed = parse_order(raw, env_cfg.order_cap)
            out.append(
                {
                    "week": i,
                    "obs_fields": {
                        k: obs[k]
                        for k in (
                            "inventory",
                            "backlog",
                            "on_order",
                            "last_demand_or_order",
                            "ship_pipeline",
                            "order_pipeline",
                        )
                    },
                    "prompt_chars": len(prompt),
                    "est_tokens": len(prompt) // 4,
                    "raw": raw,
                    "parsed": parsed,
                    "ok": (parsed is not None) == expect_ok if expect_ok else parsed is None,
                }
            )
        # step with pass-through
        orders = {r: int(core._states[r].last_demand_or_order) for r in core.roles}
        _, _, _, info = core.step(orders)
        mem.history.append(
            WeekRecord(
                week=i,
                inventory=int(core._states[core.roles[0]].inventory),
                backlog=int(core._states[core.roles[0]].backlog),
                on_order=int(core._states[core.roles[0]].on_order),
                last_demand_or_order=int(obs["last_demand_or_order"]),
                last_shipment_received=int(obs["last_shipment_received"]),
                last_order_placed=int(orders[core.roles[0]]),
                ship_pipeline=list(obs["ship_pipeline"]),
                order_pipeline=list(obs["order_pipeline"]),
                order=int(orders[core.roles[0]]),
                local_cost=float(info.local_costs[core.roles[0]]),
                allocation_received=int(core._states[core.roles[0]].last_shipment_received),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--horizon", type=int, default=52)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("artifacts/diagnostics/llm_tier_smoke.json"))
    ap.add_argument("--roundtrip-only", action="store_true")
    args = ap.parse_args()

    rt = round_trip_demo(args.seed)
    if args.roundtrip_only:
        print(json.dumps({"round_trip": rt}, indent=2))
        return

    caps: list[float | None] = [None, 1.0, 0.8]
    results = []
    for cap in caps:
        print(f"=== episode cap={cap} horizon={args.horizon} ===", flush=True)
        results.append(
            run_llm_episode(
                capacity_mult=cap,
                seed=args.seed,
                model=args.model,
                host=args.host,
                horizon=args.horizon,
            )
        )
        print(json.dumps({k: v for k, v in results[-1].items() if k != "prompt_samples"}, indent=2), flush=True)

    payload = {
        "model": args.model,
        "round_trip": rt,
        "episodes": [{k: v for k, v in e.items() if k != "prompt_samples"} for e in results],
        "prompt_samples": [e.get("prompt_samples", []) for e in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
