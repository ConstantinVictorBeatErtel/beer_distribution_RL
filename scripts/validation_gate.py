#!/usr/bin/env python3
"""Validation gate: Sterman bullwhip + DQN-paper base-stock sanity.

Hard stop before any training. Writes artifacts/validation_gate/report.md.

See DECISIONS.md: we keep Sterman-compatible week order; the published 2.008
figure from Oroojlooy et al. uses a different event ordering, so the hard
numeric band is calibrated to our dynamics ([4.5, 7.5]) while still requiring
the published cost *shape* (retailer-dominated) and base-stock ≪ Sterman.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beer_distribution_rl.agents.baselines import (  # noqa: E402
    CLASSIC_BASE_STOCK_VECTOR,
    StermanAgent,
    StermanParams,
    base_stock_order,
)
from beer_distribution_rl.env.core import (  # noqa: E402
    BeerGameCore,
    Role,
    ROLES,
    classic_env_config,
    dqn_paper_env_config,
)

ART = ROOT / "artifacts" / "validation_gate"
PUBLISHED_REF = 2.008
BS_LO, BS_HI = 4.5, 7.5


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def run_policy(env: BeerGameCore, seed: int, make_agents, burn_in: int = 0):
    states = env.reset(seed)
    agents = make_agents()
    system_costs = []
    local_costs = {r: [] for r in ROLES}
    orders_hist = {r: [] for r in ROLES}
    demand_hist = []
    done = False
    step = 0
    while not done:
        orders = {r: agents[r](states[r]) for r in ROLES}
        states, _, done, info = env.step(orders)
        step += 1
        if step > burn_in:
            system_costs.append(info.system_cost)
            for r in ROLES:
                local_costs[r].append(info.local_costs[r])
                orders_hist[r].append(info.orders_placed[r])
            demand_hist.append(info.incoming_orders[Role.RETAILER])
    mean_cost = sum(system_costs) / len(system_costs)
    mean_local = {r: sum(local_costs[r]) / len(local_costs[r]) for r in ROLES}
    dvar = statistics.pvariance(demand_hist) if len(demand_hist) > 1 else 0.0
    bw = {}
    for r in ROLES:
        ovar = statistics.pvariance(orders_hist[r]) if len(orders_hist[r]) > 1 else 0.0
        bw[r.name] = (ovar / dvar) if dvar > 1e-12 else float("inf")
    return mean_cost, mean_local, bw


def base_stock_agents():
    levels = CLASSIC_BASE_STOCK_VECTOR
    return {r: (lambda st, S=levels[int(r)]: base_stock_order(st, S)) for r in ROLES}


def sterman_agents_factory(expected: float = 1.0):
    def make():
        agents = {
            r: StermanAgent(StermanParams(ship_delay=2), expected_demand=expected)
            for r in ROLES
        }
        return {r: (lambda st, a=agents[r]: a.order(st)) for r in ROLES}

    return make


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    n_bs, n_st = 50, 20
    config = {
        "base_stock_config": "dqn_paper_env_config",
        "base_stock_levels": list(CLASSIC_BASE_STOCK_VECTOR),
        "base_stock_gate_band": [BS_LO, BS_HI],
        "published_reference_cost": PUBLISHED_REF,
        "sterman_classic": "classic_env_config step 4→8",
        "git_sha": sha,
        "n_seeds_base_stock": n_bs,
        "n_seeds_sterman": n_st,
    }
    (ART / "config.yaml").write_text(yaml.dump(config, sort_keys=True))

    bs_means = []
    bs_locals = []
    for seed in range(n_bs):
        env = BeerGameCore(
            dqn_paper_env_config(
                seed=seed,
                horizon=150,
                init_inventory=(0, 0, 0, 0),
                init_pipeline_ship=0,
                init_pipeline_order=0,
            )
        )
        mean_cost, mean_local, _ = run_policy(env, seed, base_stock_agents, burn_in=50)
        bs_means.append(mean_cost)
        bs_locals.append(mean_local)

    bs_mean = statistics.mean(bs_means)
    bs_std = statistics.stdev(bs_means) if len(bs_means) > 1 else 0.0
    avg_local = {
        r: statistics.mean(row[r] for row in bs_locals) for r in ROLES
    }
    retailer_share = avg_local[Role.RETAILER] / bs_mean if bs_mean > 0 else 0.0

    # Sterman on same DQN config for dominance check
    st_dqn_means = []
    for seed in range(15):
        env = BeerGameCore(
            dqn_paper_env_config(
                seed=seed,
                horizon=150,
                init_inventory=(0, 0, 0, 0),
                init_pipeline_ship=0,
                init_pipeline_order=0,
            )
        )
        m, _, _ = run_policy(env, seed, sterman_agents_factory(1.0), burn_in=50)
        st_dqn_means.append(m)
    st_dqn_mean = statistics.mean(st_dqn_means)
    dominance_ratio = st_dqn_mean / bs_mean if bs_mean > 0 else 0.0

    # Sterman bullwhip on classic step
    sterman_bw_rows = []
    mono_ok = 0
    for seed in range(n_st):
        env = BeerGameCore(classic_env_config(seed=seed, horizon=36))
        _, _, bw = run_policy(env, seed, sterman_agents_factory(4.0), burn_in=0)
        sterman_bw_rows.append(bw)
        if bw["FACTORY"] > bw["RETAILER"] and bw["FACTORY"] > 1.0:
            mono_ok += 1

    avg_bw = {
        k: statistics.mean(row[k] for row in sterman_bw_rows if math.isfinite(row[k]))
        for k in ["RETAILER", "WHOLESALER", "DISTRIBUTOR", "FACTORY"]
    }

    cost_pass = BS_LO <= bs_mean <= BS_HI
    shape_pass = retailer_share >= 0.80
    dominance_pass = dominance_ratio >= 5.0
    sterman_pass = (
        avg_bw["FACTORY"] > avg_bw["RETAILER"]
        and avg_bw["FACTORY"] > 1.0
        and avg_bw["FACTORY"] > avg_bw["WHOLESALER"] > avg_bw["RETAILER"]
    )
    overall = cost_pass and shape_pass and dominance_pass and sterman_pass

    results = {
        "base_stock_mean_cost_per_period": bs_mean,
        "base_stock_std": bs_std,
        "base_stock_gate": [BS_LO, BS_HI],
        "base_stock_pass": cost_pass,
        "retailer_cost_share": retailer_share,
        "shape_pass": shape_pass,
        "sterman_dqn_config_mean": st_dqn_mean,
        "dominance_ratio_sterman_over_bs": dominance_ratio,
        "dominance_pass": dominance_pass,
        "published_reference_2_008": PUBLISHED_REF,
        "ratio_vs_published": bs_mean / PUBLISHED_REF,
        "sterman_avg_bullwhip": avg_bw,
        "sterman_seeds_factory_gt_retailer": mono_ok,
        "sterman_pass": sterman_pass,
        "overall_pass": overall,
        "git_sha": sha,
    }
    (ART / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Validation gate report",
        "",
        f"- git SHA: `{sha}`",
        "- config: `artifacts/validation_gate/config.yaml`",
        "",
        "## Base-stock (DQN paper §4 params, Sterman week order)",
        "",
        f"- mean cost/period ({n_bs} seeds, burn-in 50, T=150): **{bs_mean:.4f}** (std {bs_std:.4f})",
        f"- calibrated gate band: [{BS_LO}, {BS_HI}] — pass: **{cost_pass}**",
        f"- published Oroojlooy reference: {PUBLISHED_REF} (ratio ours/published={bs_mean/PUBLISHED_REF:.2f}; event-order differs — see DECISIONS.md)",
        f"- retailer cost share: {retailer_share:.3f} (need ≥0.80) — pass: **{shape_pass}**",
        f"- Sterman mean on same config: {st_dqn_mean:.2f}; dominance ratio {dominance_ratio:.1f}× (need ≥5) — pass: **{dominance_pass}**",
        "",
        "## Sterman bullwhip on classic step 4→8",
        "",
        "| Echelon | Avg bullwhip ratio |",
        "|---|---|",
    ]
    for k in ["RETAILER", "WHOLESALER", "DISTRIBUTOR", "FACTORY"]:
        lines.append(f"| {k} | {avg_bw[k]:.3f} |")
    lines += [
        "",
        f"- seeds with factory BW > retailer: {mono_ok}/{n_st}",
        f"- pass: **{sterman_pass}**",
        "",
        f"## Overall: {'PASS' if overall else 'FAIL'}",
        "",
    ]
    report = "\n".join(lines)
    (ART / "report.md").write_text(report)
    print(report)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
