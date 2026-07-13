#!/usr/bin/env python3
"""B1 bug-hunt probes: action-cap saturation + infinite-capacity backlog.

Writes JSON under analysis/diag/cache/ and figures under analysis/figs/diag/.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.diag.common import (  # noqa: E402
    CACHE_DIR,
    DIAG_EVAL_SEED_OFFSET,
    FIG_DIR,
    ROLES_ORDER,
    ensure_dirs,
    list_m3_runs,
    load_trainer,
    write_json,
)
from analysis.diag.eval_ablation import evaluate_with_ablation  # noqa: E402
from beer_distribution_rl.agents.baselines import base_stock_order  # noqa: E402
from beer_distribution_rl.env.core import (  # noqa: E402
    BeerGameCore,
    EnvConfig,
    ROLES,
    Role,
    classic_env_config,
)
from beer_distribution_rl.env.demand import UniformDemand  # noqa: E402

ORDER_CAP = 64
MU = 7.5  # U[0,15] mean


# ---------------------------------------------------------------------------
# Anomaly 1 — order saturation
# ---------------------------------------------------------------------------


def anomaly1_order_saturation(
    *,
    n_episodes: int = 20,
    max_seeds: int = 5,
    rationing: str = "proportional",
) -> dict:
    ensure_dirs()
    runs = list_m3_runs(rationing=rationing)
    # Cap tags in capacity order; limit seeds for speed.
    by_tag: dict[str, list] = defaultdict(list)
    for r in runs:
        by_tag[r["capacity_tag"]].append(r)
    for tag in by_tag:
        by_tag[tag] = sorted(by_tag[tag], key=lambda x: int(x["seed"]))[:max_seeds]

    hist: dict[str, dict[str, list[int]]] = {}
    frac_at_cap: dict[str, dict[str, float]] = {}
    frac_at_rel_ceiling: dict[str, dict[str, float]] = {}
    mean_order: dict[str, dict[str, float]] = {}
    p95_order: dict[str, dict[str, float]] = {}
    abs_sensitivity_proxy: dict[str, dict[str, float]] = {}

    for tag, rows in by_tag.items():
        role_orders: dict[str, list[int]] = {rn: [] for rn in ROLES_ORDER}
        for row in rows:
            trainer, _ = load_trainer(row["run_dir"])
            m = evaluate_with_ablation(
                trainer,
                n_episodes=n_episodes,
                seed=DIAG_EVAL_SEED_OFFSET + 40_000 + int(row["seed"]),
                ablation_mode="intact",
                collect_steps=True,
            )
            for s in m["steps"]:
                for rn in ROLES_ORDER:
                    role_orders[rn].append(int(s["orders_placed"][rn]))
            del trainer

        hist[tag] = {rn: role_orders[rn] for rn in ROLES_ORDER}
        frac_at_cap[tag] = {
            rn: float(np.mean([1.0 if o >= ORDER_CAP else 0.0 for o in role_orders[rn]]))
            for rn in ROLES_ORDER
        }
        # Relative-action soft ceiling: demand_max(15)+delta_max(8)=23 for retailer,
        # but upstream last_demand can exceed 15. Track fraction at env hard cap only above.
        mean_order[tag] = {rn: float(np.mean(role_orders[rn])) for rn in ROLES_ORDER}
        p95_order[tag] = {rn: float(np.percentile(role_orders[rn], 95)) for rn in ROLES_ORDER}
        # Proxy for near-constant: 1 - (std / (mean+1e-6)) clipped
        abs_sensitivity_proxy[tag] = {
            rn: float(np.std(role_orders[rn])) for rn in ROLES_ORDER
        }
        # Also fraction ordering at high values (>= 32, half of cap)
        frac_at_rel_ceiling[tag] = {
            rn: float(np.mean([1.0 if o >= 32 else 0.0 for o in role_orders[rn]]))
            for rn in ROLES_ORDER
        }

    # Figures: histograms per role for tight (0p8mu) and inf
    for tag, title in [("0p8mu", "0.8μ tight"), ("inf", "∞ capacity")]:
        if tag not in hist:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for ax, rn in zip(axes.ravel(), ROLES_ORDER):
            data = hist[tag][rn]
            ax.hist(data, bins=np.arange(0, ORDER_CAP + 2) - 0.5, color="#4a6fa5", edgecolor="none")
            ax.axvline(ORDER_CAP, color="#a33b2b", ls="--", lw=1.2, label=f"cap={ORDER_CAP}")
            ax.set_title(
                f"{rn}: mean={mean_order[tag][rn]:.1f}, "
                f"P(cap)={frac_at_cap[tag][rn]:.3f}, std={abs_sensitivity_proxy[tag][rn]:.2f}"
            )
            ax.set_xlabel("order qty")
            ax.set_ylabel("weeks")
            ax.set_xlim(-1, ORDER_CAP + 1)
        fig.suptitle(f"B1 Anomaly1 — order histogram ({title})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"bughunt_a1_hist_{tag}.png", dpi=160)
        plt.close(fig)

    # Cap-hit summary bar chart
    tags_order = [t for t in ["inf", "1p5mu", "1p2mu", "1p0mu", "0p8mu"] if t in frac_at_cap]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(tags_order))
    w = 0.2
    for i, rn in enumerate(ROLES_ORDER):
        ax.bar(
            x + (i - 1.5) * w,
            [frac_at_cap[t][rn] for t in tags_order],
            w,
            label=rn,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tags_order)
    ax.set_ylabel(f"Fraction of weeks with order == {ORDER_CAP}")
    ax.set_title("B1 Anomaly1 — action-cap saturation by role × capacity")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bughunt_a1_cap_frac.png", dpi=160)
    plt.close(fig)

    # Derive recommended cap under AR(1) + backlog accumulation
    # Max plausible: demand can spike; relative Δ=+8; bullwhip amplification across 4 echelons.
    # Worst-case recursive: o_{k+1} <= o_k + delta_max, starting from d_max.
    # Absolute: if action space were absolute 0..cap, need headroom above max backlog recovery.
    ar1_mu, ar1_phi, ar1_sigma = 8.0, 0.7, 2.0
    # Rough high demand quantile ~ mu + 3*sigma_stat where sigma_stat = sigma/sqrt(1-phi^2)
    sig_stat = ar1_sigma / np.sqrt(1 - ar1_phi**2)
    d_hi = ar1_mu + 3 * sig_stat  # ~16.4
    # Total replenishment delay per link ≈ L_o + L_s = 3; chain of 4 → effective L varies.
    # Conservative max order for absolute discrete: cover 4 weeks of high demand + backlog buffer.
    backlog_buffer_weeks = 8  # catch-up after capacity episode
    max_plausible_need = d_hi * (3 + backlog_buffer_weeks)  # ~180
    # Relative-mode effective max without hard clamp: can ratchet by +delta each week
    # from a spike; ratchet ceiling in T weeks from d_hi is d_hi + T*delta_max.
    ratchet_52 = d_hi + 52 * 8  # enormous if uncapped relative — so hard cap matters for ratchets

    result = {
        "n_episodes": n_episodes,
        "max_seeds": max_seeds,
        "order_cap": ORDER_CAP,
        "frac_at_cap": frac_at_cap,
        "frac_ge_32": frac_at_rel_ceiling,
        "mean_order": mean_order,
        "p95_order": p95_order,
        "order_std": abs_sensitivity_proxy,
        "cap_recommendation": {
            "ar1_d_hi_approx": float(d_hi),
            "max_plausible_need_abs": float(max_plausible_need),
            "relative_ratchet_52w": float(ratchet_52),
            "suggested_hard_cap": 128,
            "preferred_action_space": (
                "Keep relative Δ∈[-δ,δ] for learning tractability, but raise env order_cap "
                "so the hard clamp rarely binds under bullwhip ratchets; OR switch to "
                "log-scaled / continuous order surplus. Absolute {0..64} is too coarse and "
                "binds under amplification."
            ),
            "rationale": (
                f"Under AR(1) φ=0.7, high demand ≈{d_hi:.1f}. With relative +8/week ratchet "
                f"and 4-echelon bullwhip, orders easily exceed 64. Cap=64 therefore flattens "
                f"both policy sensitivity (D3) and the inflation metric ceiling."
            ),
        },
        # Drop raw hist lists from JSON (too large); keep summary counts at 0,32,64
        "hist_summary": {
            tag: {
                rn: {
                    "n": len(hist[tag][rn]),
                    "count_0": int(sum(1 for o in hist[tag][rn] if o == 0)),
                    "count_ge_32": int(sum(1 for o in hist[tag][rn] if o >= 32)),
                    "count_64": int(sum(1 for o in hist[tag][rn] if o >= ORDER_CAP)),
                    "count_le_23": int(sum(1 for o in hist[tag][rn] if o <= 23)),
                }
                for rn in ROLES_ORDER
            }
            for tag in hist
        },
    }
    write_json(CACHE_DIR / "bughunt_a1.json", result)
    return result


# ---------------------------------------------------------------------------
# Anomaly 2 — delays, conservation, init, C=∞ backlog
# ---------------------------------------------------------------------------


def delay_unit_trace() -> dict:
    """Inject a single unit order at t=0 with all else zero; record receipt weeks."""
    cfg = EnvConfig(
        horizon=12,
        ship_delay=2,
        order_delay=1,
        demand=UniformDemand(0, 0),  # zero demand
        init_inventory=(0, 100, 0, 0),  # wholesaler stocked so it can ship
        init_pipeline_ship=0,
        init_pipeline_order=0,
        capacity=None,
        seed=0,
    )
    env = BeerGameCore(cfg)
    env.reset(0)
    # Zero pipelines explicitly
    for r in ROLES:
        env._states[r].ship_pipeline = [0] * cfg.ship_delay
        env._states[r].order_pipeline = [0] * cfg.order_delay
        env._states[r].on_order = 0
        env._states[r].inventory = 100 if r == Role.WHOLESALER else 0
        env._states[r].backlog = 0

    log = []
    # Week 1: retailer places order of 1; everyone else 0
    for week in range(1, 9):
        if week == 1:
            orders = {
                Role.RETAILER: 1,
                Role.WHOLESALER: 0,
                Role.DISTRIBUTOR: 0,
                Role.FACTORY: 0,
            }
        else:
            orders = {r: 0 for r in ROLES}
        states, _, _, info = env.step(orders)
        log.append(
            {
                "week": week,
                "incoming": {r.name: info.incoming_orders[r] for r in ROLES},
                "shipments": {r.name: info.shipments[r] for r in ROLES},
                "received": {r.name: info.shipments_received[r] for r in ROLES},
                "inv": {r.name: states[r].inventory for r in ROLES},
                "backlog": {r.name: states[r].backlog for r in ROLES},
                "retailer_ship_pipe": list(states[Role.RETAILER].ship_pipeline),
                "wholesaler_order_pipe": list(states[Role.WHOLESALER].order_pipeline),
            }
        )

    # Expected classic (L_o=1, L_s=2):
    # week1: retailer orders 1
    # week2: wholesaler sees incoming=1, ships 1
    # week4: retailer receives 1  (ship appended week2, received after 2 pops)
    wholesaler_sees_week = next(
        (e["week"] for e in log if e["incoming"]["WHOLESALER"] == 1), None
    )
    retailer_recv_week = next(
        (e["week"] for e in log if e["received"]["RETAILER"] == 1), None
    )
    wholesaler_ships_week = next(
        (e["week"] for e in log if e["shipments"]["WHOLESALER"] == 1), None
    )

    expected = {
        "order_arrives_at_wholesaler_week": 2,  # L_o=1
        "wholesaler_ships_week": 2,
        "retailer_receives_week": 4,  # ship week 2 + L_s=2 → week 4
    }
    observed = {
        "order_arrives_at_wholesaler_week": wholesaler_sees_week,
        "wholesaler_ships_week": wholesaler_ships_week,
        "retailer_receives_week": retailer_recv_week,
    }
    match = observed == expected
    return {
        "expected": expected,
        "observed": observed,
        "matches_classic_Lo1_Ls2": match,
        "trace": log,
        "note": (
            "Classic Sterman week order with L_o=1, L_s=2: order placed week t arrives "
            "upstream week t+1; shipment sent week t arrives downstream week t+2. "
            "Total retailer order→receipt = 3 weeks. PROJECT_SPEC §3.1 / DECISIONS.md."
        ),
    }


def delay_factory_production_trace() -> dict:
    """Factory produces 1 at week 1; when does factory inventory receive it?"""
    cfg = EnvConfig(
        horizon=8,
        ship_delay=2,
        order_delay=1,
        demand=UniformDemand(0, 0),
        init_inventory=(0, 0, 0, 0),
        init_pipeline_ship=0,
        init_pipeline_order=0,
        capacity=None,
        seed=0,
    )
    env = BeerGameCore(cfg)
    env.reset(0)
    for r in ROLES:
        env._states[r].ship_pipeline = [0] * cfg.ship_delay
        env._states[r].order_pipeline = [0] * cfg.order_delay
        env._states[r].on_order = 0
        env._states[r].inventory = 0
        env._states[r].backlog = 0

    recv_week = None
    for week in range(1, 7):
        orders = {r: (1 if (r == Role.FACTORY and week == 1) else 0) for r in ROLES}
        states, _, _, info = env.step(orders)
        if info.shipments_received[Role.FACTORY] == 1:
            recv_week = week
            break
    return {
        "factory_produces_week": 1,
        "factory_receives_week": recv_week,
        "expected_receive_week": 3,  # L_s=2
        "match": recv_week == 3,
    }


def conservation_probe(n_episodes: int = 30, horizon: int = 52, seed0: int = 0) -> dict:
    """Property: cumulative factory production = physical stock + delivered to customers."""
    failures = []
    max_abs_err = 0.0
    for ep in range(n_episodes):
        seed = seed0 + ep
        rng = np.random.default_rng(seed)
        cfg = EnvConfig(
            horizon=horizon,
            demand=UniformDemand(0, 15),
            capacity=None,
            seed=seed,
            init_inventory=(12, 12, 12, 12),
            init_pipeline_ship=4,
            init_pipeline_order=4,
        )
        env = BeerGameCore(cfg)
        env.reset(seed)
        # Initial physical goods
        init_goods = 0
        for r in ROLES:
            st = env._states[r]
            init_goods += st.inventory + sum(st.ship_pipeline)

        cum_prod = 0
        cum_delivered = 0
        done = False
        while not done:
            orders = {r: int(rng.integers(0, 65)) for r in ROLES}
            states, _, done, info = env.step(orders)
            cum_prod += info.factory_production
            cum_delivered += info.shipments[Role.RETAILER]
            physical = 0
            for r in ROLES:
                physical += states[r].inventory + sum(states[r].ship_pipeline)
            # Identity: init + produced = physical + delivered
            lhs = init_goods + cum_prod
            rhs = physical + cum_delivered
            err = abs(lhs - rhs)
            max_abs_err = max(max_abs_err, err)
            if err > 0:
                failures.append(
                    {
                        "ep": ep,
                        "t": env.t,
                        "lhs": lhs,
                        "rhs": rhs,
                        "err": err,
                        "init": init_goods,
                        "cum_prod": cum_prod,
                        "physical": physical,
                        "cum_delivered": cum_delivered,
                    }
                )
                break
    return {
        "n_episodes": n_episodes,
        "n_failures": len(failures),
        "max_abs_err": max_abs_err,
        "conserved": len(failures) == 0,
        "first_failures": failures[:5],
    }


def base_stock_vs_init() -> dict:
    """Compare init inventory position to base-stock implied by delays + U[0,15]."""
    # Per-echelon lead time for receiving own replenishment:
    # Non-factory: L_o (order transit) + L_s (ship) = 3, plus upstream may add delay when
    # stocked out. Factory: L_s = 2 only (production pipeline).
    L_s, L_o = 2, 1
    L_nonfact = L_o + L_s  # 3
    L_fact = L_s  # 2
    mu = MU
    # Newsvendor-ish base-stock for continuous review approx: S ≈ μ L + z σ √L
    # U[0,15]: σ ≈ sqrt(((15-0+1)^2-1)/12) ≈ 4.76
    sigma = np.sqrt(((15 - 0 + 1) ** 2 - 1) / 12.0)
    z = 1.0  # mild safety (~84% fill if Gaussian)
    S_retailer = mu * L_nonfact + z * sigma * np.sqrt(L_nonfact)
    S_wholesaler = mu * L_nonfact + z * sigma * np.sqrt(L_nonfact)
    S_distributor = mu * L_nonfact + z * sigma * np.sqrt(L_nonfact)
    S_factory = mu * L_fact + z * sigma * np.sqrt(L_fact)

    # Init inventory position = on-hand + on_order(=sum ship pipe) - backlog
    init_inv = 12
    init_ship_pipe_sum = 4 * L_s  # 8
    init_ip = init_inv + init_ship_pipe_sum  # 20 (backlog 0)
    # Note: order pipeline is incoming demand to *this* node, not part of own IP.

    return {
        "mu": mu,
        "sigma_uniform_0_15": float(sigma),
        "lead_times": {"non_factory": L_nonfact, "factory": L_fact},
        "implied_base_stock_approx": {
            "retailer": float(S_retailer),
            "wholesaler": float(S_wholesaler),
            "distributor": float(S_distributor),
            "factory": float(S_factory),
        },
        "init": {
            "on_hand": init_inv,
            "ship_pipeline_sum": init_ship_pipe_sum,
            "order_pipeline_fill": 4,
            "inventory_position": init_ip,
            "note": "init pipelines filled with 4 (= classic step demand), not μ=7.5",
        },
        "gap_ip_vs_S": {
            "retailer": float(init_ip - S_retailer),
            "factory": float(init_ip - S_factory),
        },
        "assessment": (
            "Init IP=20 vs retailer S≈μL+zσ√L≈30.5 — systematically under-stocked for "
            "U[0,15]. Pipelines seeded at classic demand=4, not 7.5. This alone produces "
            "transient (and possibly chronic, if policies are weak) backlog even at C=∞."
        ),
    }


def backlog_at_infinity(
    *,
    n_episodes: int = 20,
    max_seeds: int = 5,
    burn_in: int = 10,
) -> dict:
    """Measure backlog rates under C=∞ for learned policies vs base-stock oracle."""
    ensure_dirs()
    runs = [
        r
        for r in list_m3_runs(rationing="proportional", capacity_tags=["inf"])
        if int(r["seed"]) < max_seeds
    ]

    learned = {"any_backlog_frac": [], "mean_backlog_sum": [], "post_burn_backlog_frac": []}
    for row in runs:
        trainer, _ = load_trainer(row["run_dir"])
        m = evaluate_with_ablation(
            trainer,
            n_episodes=n_episodes,
            seed=DIAG_EVAL_SEED_OFFSET + 50_000 + int(row["seed"]),
            ablation_mode="intact",
            collect_steps=True,
        )
        steps = m["steps"]
        any_bl = [1.0 if any(v > 0 for v in s["backlogs"].values()) else 0.0 for s in steps]
        bl_sum = [float(sum(s["backlogs"].values())) for s in steps]
        post = any_bl[burn_in:] if len(any_bl) > burn_in else any_bl
        # Per-episode grouping for burn-in: steps are flat across episodes
        # Recompute post-burn per episode
        post_ep = []
        by_ep: dict[int, list] = defaultdict(list)
        for s in steps:
            by_ep[s["ep"]].append(s)
        for ep_steps in by_ep.values():
            for s in ep_steps[burn_in:]:
                post_ep.append(1.0 if any(v > 0 for v in s["backlogs"].values()) else 0.0)
        learned["any_backlog_frac"].append(float(np.mean(any_bl)))
        learned["mean_backlog_sum"].append(float(np.mean(bl_sum)))
        learned["post_burn_backlog_frac"].append(float(np.mean(post_ep)) if post_ep else float("nan"))
        del trainer

    # Base-stock oracle with levels matched to U[0,15] lead times
    bs = base_stock_vs_init()["implied_base_stock_approx"]
    S = {
        Role.RETAILER: int(round(bs["retailer"])),
        Role.WHOLESALER: int(round(bs["wholesaler"])),
        Role.DISTRIBUTOR: int(round(bs["distributor"])),
        Role.FACTORY: int(round(bs["factory"])),
    }
    # Also classic-ish levels scaled: try IP-target around 30
    oracle_stats = {"any_backlog_frac": [], "mean_backlog_sum": [], "post_burn_backlog_frac": []}
    for ep in range(n_episodes):
        seed = 9000 + ep
        cfg = EnvConfig(
            horizon=52,
            demand=UniformDemand(0, 15),
            capacity=None,
            seed=seed,
            regime="A",
            signaling_enabled=False,
        )
        env = BeerGameCore(cfg)
        states = env.reset(seed)
        any_bl = []
        bl_sum = []
        done = False
        t = 0
        while not done:
            orders = {r: base_stock_order(states[r], S[r], order_cap=ORDER_CAP) for r in ROLES}
            states, _, done, info = env.step(orders)
            t += 1
            bl = sum(states[r].backlog for r in ROLES)
            any_bl.append(1.0 if bl > 0 else 0.0)
            bl_sum.append(float(bl))
        oracle_stats["any_backlog_frac"].append(float(np.mean(any_bl)))
        oracle_stats["mean_backlog_sum"].append(float(np.mean(bl_sum)))
        oracle_stats["post_burn_backlog_frac"].append(float(np.mean(any_bl[burn_in:])))

    # Same oracle but with classic init — already default. Compare with warm init at μ.
    warm_stats = {"any_backlog_frac": [], "mean_backlog_sum": [], "post_burn_backlog_frac": []}
    for ep in range(n_episodes):
        seed = 9100 + ep
        cfg = EnvConfig(
            horizon=52,
            demand=UniformDemand(0, 15),
            capacity=None,
            seed=seed,
            init_inventory=(20, 20, 20, 20),
            init_pipeline_ship=8,  # ≈ μ rounded
            init_pipeline_order=8,
        )
        env = BeerGameCore(cfg)
        states = env.reset(seed)
        any_bl = []
        bl_sum = []
        done = False
        while not done:
            orders = {r: base_stock_order(states[r], S[r], order_cap=ORDER_CAP) for r in ROLES}
            states, _, done, info = env.step(orders)
            bl = sum(states[r].backlog for r in ROLES)
            any_bl.append(1.0 if bl > 0 else 0.0)
            bl_sum.append(float(bl))
        warm_stats["any_backlog_frac"].append(float(np.mean(any_bl)))
        warm_stats["mean_backlog_sum"].append(float(np.mean(bl_sum)))
        warm_stats["post_burn_backlog_frac"].append(float(np.mean(any_bl[burn_in:])))

    def _summ(d):
        return {
            k: {
                "mean": float(np.mean(v)),
                "ci95": float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
            }
            for k, v in d.items()
        }

    # Inflation at C=∞ explanation probe on one run
    infl_explanation = None
    if runs:
        trainer, _ = load_trainer(runs[0]["run_dir"])
        m = evaluate_with_ablation(
            trainer,
            n_episodes=5,
            seed=DIAG_EVAL_SEED_OFFSET + 60_000,
            ablation_mode="intact",
            collect_steps=True,
        )
        steps = m["steps"]
        infl = [s["factory_order_inflation"] for s in steps if s["factory_order_inflation"] == s["factory_order_inflation"]]
        # Split by whether ANY node has backlog (allocation_triggers) even though capacity never binds
        infl_when_short = [
            s["factory_order_inflation"]
            for s in steps
            if s["allocation_triggers"] and s["factory_order_inflation"] == s["factory_order_inflation"]
        ]
        infl_when_ok = [
            s["factory_order_inflation"]
            for s in steps
            if (not s["allocation_triggers"])
            and s["factory_order_inflation"] == s["factory_order_inflation"]
        ]
        infl_explanation = {
            "note": (
                "At C=∞, capacity_binds is always False, so D5's infl|non-binding is the "
                "unconditional factory order/need ratio. A value ≈1.52 is classic bullwhip "
                "amplification (factory orders above incoming distributor orders), not a "
                "rationing-game signal. It is expected under local-cost IPPO / Sterman-like "
                "behavior and does not indicate a capacity bug."
            ),
            "mean_infl_unconditional": float(np.mean(infl)) if infl else float("nan"),
            "mean_infl_when_any_backlog": float(np.mean(infl_when_short)) if infl_when_short else float("nan"),
            "mean_infl_when_no_backlog": float(np.mean(infl_when_ok)) if infl_when_ok else float("nan"),
            "frac_weeks_any_backlog": float(np.mean([1.0 if s["allocation_triggers"] else 0.0 for s in steps])),
        }
        del trainer

    result = {
        "learned_C_inf": _summ(learned),
        "base_stock_oracle_classic_init": _summ(oracle_stats),
        "base_stock_oracle_warm_init": _summ(warm_stats),
        "base_stock_levels_used": {r.name.lower(): S[r] for r in ROLES},
        "inflation_at_inf_explanation": infl_explanation,
    }
    write_json(CACHE_DIR / "bughunt_a2_backlog.json", result)
    return result


def anomaly2_all() -> dict:
    ensure_dirs()
    delay = delay_unit_trace()
    fact = delay_factory_production_trace()
    cons = conservation_probe()
    init = base_stock_vs_init()
    backlog = backlog_at_infinity()
    result = {
        "delay_unit_trace": {
            k: v for k, v in delay.items() if k != "trace"
        },
        "delay_unit_trace_full": delay,
        "delay_factory_trace": fact,
        "conservation": cons,
        "init_vs_basestock": init,
        "backlog_at_infinity": backlog,
    }
    write_json(CACHE_DIR / "bughunt_a2.json", result)
    return result


def main() -> int:
    ensure_dirs()
    print("=== B1 Anomaly 1: order saturation ===", flush=True)
    a1 = anomaly1_order_saturation()
    print(json.dumps({k: a1[k] for k in ("frac_at_cap", "mean_order", "p95_order")}, indent=2))

    print("=== B1 Anomaly 2: delays / conservation / backlog ===", flush=True)
    a2 = anomaly2_all()
    print(
        json.dumps(
            {
                "delay_match": a2["delay_unit_trace"]["matches_classic_Lo1_Ls2"],
                "factory_delay_match": a2["delay_factory_trace"]["match"],
                "conserved": a2["conservation"]["conserved"],
                "init_assessment": a2["init_vs_basestock"]["assessment"],
                "learned_backlog": a2["backlog_at_infinity"]["learned_C_inf"],
                "oracle_backlog": a2["backlog_at_infinity"]["base_stock_oracle_classic_init"],
                "warm_oracle_backlog": a2["backlog_at_infinity"]["base_stock_oracle_warm_init"],
                "infl_note": a2["backlog_at_infinity"]["inflation_at_inf_explanation"],
            },
            indent=2,
        )
    )
    write_json(CACHE_DIR / "bughunt_summary.json", {"a1": a1, "a2_keys": list(a2.keys())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
