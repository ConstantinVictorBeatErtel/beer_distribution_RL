"""Capability-floor runner (M4): let one model play a full beer game episode.

This is the *prompted baseline* / capability-floor check that PROJECT_SPEC §4
and the milestone plan (M4) require BEFORE any paid GRPO run (M5). It ties the
existing LLM text plumbing (serializer + rolling memory + constrained decoder)
to the environment and plays a full episode, logging per-role cost, mean order,
parse-fail rate, and a short human-readable transcript.

No training. Inference only. With ``--backend heuristic`` it needs no model at
all (order = last demand observed) — a demand-matching sanity baseline and a
way to verify the env/serializer/memory wiring with zero external deps.

Usage:
    # wiring check, no LLM, runs anywhere:
    python scripts/run_llm_episode.py --backend heuristic --cell classic

    # real capability floor (needs a local Ollama serving qwen2.5:3b):
    python scripts/run_llm_episode.py --backend ollama --model qwen2.5:3b --cell y_tight
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Callable

from beer_distribution_rl.agents.llm.memory import (
    DEFAULT_ROLLING_WINDOW,
    AgentMemory,
    WeekRecord,
)
from beer_distribution_rl.agents.llm.serializer import observe_local, serialize_prompt
from beer_distribution_rl.env.core import (
    BeerGameCore,
    EnvConfig,
    Role,
    classic_env_config,
    y_topology_env_config,
)
from beer_distribution_rl.env.demand import CorrelatedYDemand

# A decoder maps (prompt, last_demand_or_order, stats_key) -> (order, parse_ok).
Decoder = Callable[[str, int, str], "tuple[int, bool]"]


def build_cell_config(cell: str, *, horizon: int) -> EnvConfig:
    """Return the EnvConfig for a named smoke cell.

    ``classic``  serial, no capacity — cheapest wiring check, runs anywhere.
    ``y_tight``  Y-topology at capacity = 1.0 * mean demand, proportional
                 rationing, AR(1) demand — the first-GRPO-cell env (order-only,
                 Regime A / signaling off) from the readiness audit.
    """
    if cell == "classic":
        return classic_env_config(horizon=horizon)
    if cell == "y_tight":
        # Two Y customers, shared AR(1) factor, mu=7.5 each -> mean total 15
        # (DECISIONS: Y x ar1 uses CorrelatedYDemand). C = 1.0 * mean total = 15.
        # Regime A (no signaling), proportional rationing.
        return y_topology_env_config(
            horizon=horizon,
            demand=CorrelatedYDemand(),
            capacity=15.0,
            regime="A",
            signaling_enabled=False,
        )
    raise ValueError(f"unknown cell {cell!r}; choose classic|y_tight")


def heuristic_decoder() -> Decoder:
    """Order = last demand/order observed (delta = 0). No model, no network."""

    def decode(prompt: str, last_demand_or_order: int, stats_key: str) -> tuple[int, bool]:
        _ = (prompt, stats_key)
        return max(0, int(last_demand_or_order)), True

    return decode


def ollama_decoder(model: str, host: str, order_cap: int) -> tuple[Decoder, Any]:
    """Real constrained-decoding backend (local Ollama). Returns (decoder, stats)."""
    from beer_distribution_rl.agents.llm.decode import ConstrainedOrderDecoder

    dec = ConstrainedOrderDecoder(model=model, host=host, order_cap=order_cap)

    def decode(prompt: str, last_demand_or_order: int, stats_key: str) -> tuple[int, bool]:
        res = dec.sample_order(prompt, last_demand_or_order, stats_key=stats_key)
        return res.order, res.parse_ok

    return decode, dec.stats


def run_episode(
    *,
    cell: str,
    decoder: Decoder,
    seed: int,
    horizon: int,
    window: int,
    transcript_weeks: int,
) -> dict[str, Any]:
    cfg = build_cell_config(cell, horizon=horizon)
    core = BeerGameCore(cfg)
    core.reset(seed=seed)

    roles: list[Role] = list(core.roles)
    memories = {r: AgentMemory(role=r, role_name=core.role_names[r], window=window) for r in roles}
    totals = {r: 0.0 for r in roles}
    order_sum = {r: 0 for r in roles}
    parse_fail = 0
    parse_total = 0
    transcript: list[dict[str, Any]] = []

    for _week in range(horizon):
        obs = {r: observe_local(core, r) for r in roles}
        orders: dict[Role, int] = {}
        chosen: dict[Role, int] = {}
        for r in roles:
            o = obs[r]
            prompt = serialize_prompt(
                memories[r],
                o,
                order_cap=int(o["order_cap"]),
                holding=float(o["holding_cost"]),
                backlog_cost=float(o["backlog_cost"]),
                window=window,
            )
            last = int(o["last_demand_or_order"])
            order, ok = decoder(prompt, last, core.role_names[r])
            parse_total += 1
            if not ok:
                parse_fail += 1
            orders[r] = order
            chosen[r] = order
            order_sum[r] += order

        _states, rewards, _done, info = core.step(orders)

        # Post-step: record own-history for each role's rolling memory.
        post = {r: observe_local(core, r) for r in roles}
        local_costs = getattr(info, "local_costs", {}) or {}
        recv = getattr(info, "shipments_received", {}) or {}
        for r in roles:
            p = post[r]
            cost = float(local_costs.get(r, -float(rewards.get(r, 0.0))))
            totals[r] += cost
            memories[r].append(
                WeekRecord(
                    week=int(obs[r]["t"]),
                    demand_or_incoming=int(obs[r]["last_demand_or_order"]),
                    ship_in=int(obs[r]["last_shipment_received"]),
                    ordered=int(chosen[r]),
                    alloc_recv=int(recv.get(r, p["last_shipment_received"])),
                    inventory=int(p["inventory"]),
                    backlog=int(p["backlog"]),
                    on_order=int(p["on_order"]),
                    ship_pipeline=list(p["ship_pipeline"]),
                    order_pipeline=list(p["order_pipeline"]),
                    local_cost=cost,
                )
            )

        if len(transcript) < transcript_weeks:
            transcript.append(
                {
                    "week": int(obs[roles[0]]["t"]),
                    "orders": {core.role_names[r]: int(chosen[r]) for r in roles},
                    "local_costs": {core.role_names[r]: round(float(local_costs.get(r, 0.0)), 2) for r in roles},
                }
            )

    n_weeks = horizon
    return {
        "cell": cell,
        "seed": seed,
        "horizon": horizon,
        "window": window,
        "roles": [core.role_names[r] for r in roles],
        "total_cost_per_role": {core.role_names[r]: round(totals[r], 2) for r in roles},
        "system_total_cost": round(sum(totals.values()), 2),
        "mean_order_per_role": {core.role_names[r]: round(order_sum[r] / n_weeks, 2) for r in roles},
        "parse_fail_rate": round(parse_fail / max(parse_total, 1), 4),
        "parse_total": parse_total,
        "transcript_head": transcript,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["heuristic", "ollama"], default="heuristic")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--cell", choices=["classic", "y_tight"], default="classic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=52)
    ap.add_argument("--window", type=int, default=DEFAULT_ROLLING_WINDOW)
    ap.add_argument("--transcript-weeks", type=int, default=6)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = build_cell_config(args.cell, horizon=args.horizon)
    if args.backend == "heuristic":
        decoder = heuristic_decoder()
        stats = None
    else:
        decoder, stats = ollama_decoder(args.model, args.host, int(cfg.order_cap))

    report = run_episode(
        cell=args.cell,
        decoder=decoder,
        seed=args.seed,
        horizon=args.horizon,
        window=args.window,
        transcript_weeks=args.transcript_weeks,
    )
    report["backend"] = args.backend
    report["model"] = args.model if args.backend == "ollama" else "heuristic-demand-match"
    if stats is not None:
        report["parse_stats"] = stats.as_dict()

    print(json.dumps(report, indent=2))
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
