# B1 Bug Hunt — Anomalies D3/D5

**Status:** complete (blocking gate for env v1.1)  
**Probe script:** `python -m analysis.diag.bughunt_b1`  
**Figures:** `analysis/figs/diag/bughunt_a1_*.png`  
**Caches:** `analysis/diag/cache/bughunt_*.json`

Non-negotiable constraints were respected: no reward shaping, no honesty enforcement, no cross-role parameter sharing. This is an environment correctness audit only.

---

## Executive verdict

| Anomaly | Bug? | Invalidate M3 / Tier-1? |
|---|---|---|
| **A1** policies saturate at action cap 64 | **No** — cap hit rate = 0% for learned IPPO across all capacities | **No.** D3 sensitivity collapse is not cap-induced. |
| **A2** chronic backlog at C=∞ | **Partial.** Delays and goods conservation are correct. A real **`on_order` init undercount** (−4 permanent bias) was found and fixed. Physical backlog at C=∞ is mostly demand variance + classic-init understocking for U[0,15], not a delay bug. | **Do not discard Tier-1.** No delay bug. Physical costs / D2–D6 structural conclusions stand. Re-train after the `on_order` fix for clean v1.1 numbers (obs feature was biased). |

**Plain statement required by the brief:** a delay off-by-one was **not** confirmed. Prior Tier-1 results must **not** be discarded on delay grounds.

---

## Anomaly 1 — action-cap saturation / D3 sensitivity collapse

### Reproduction

Re-ran intact eval on frozen M3 Regime-B proportional checkpoints (5 seeds × 5 caps × 20 episodes). Histograms of decoded order quantities per role:

- `analysis/figs/diag/bughunt_a1_hist_0p8mu.png`
- `analysis/figs/diag/bughunt_a1_hist_inf.png`
- `analysis/figs/diag/bughunt_a1_cap_frac.png`

### Result: cap is **not** binding for learned policies

Fraction of weeks with `order == 64`, per role × capacity (all zeros):

| Cap | retailer | wholesaler | distributor | factory |
|---|---:|---:|---:|---:|
| ∞ | 0.00 | 0.00 | 0.00 | 0.00 |
| 1.5μ | 0.00 | 0.00 | 0.00 | 0.00 |
| 1.2μ | 0.00 | 0.00 | 0.00 | 0.00 |
| 1.0μ | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.8μ | 0.00 | 0.00 | 0.00 | 0.00 |

Orders sit well below the hard clamp:

| Cap | factory mean | factory P95 |
|---|---:|---:|
| ∞ | 8.1 | 25 |
| 0.8μ | 10.9 | 27 |

Training uses **relative** actions `order = clip(last_demand + Δ, 0, 64)` with `Δ ∈ [-8, 8]` (`DECISIONS.md`). Retailer demand ≤ 15 ⇒ retailer orders ≤ 23 unless upstream amplification raises `last_demand_or_order`. Learned policies never ratchet to 64.

### Root cause of D3 collapse (not a bug)

D3’s absolute sensitivities fall to ~0.11 / ~0.05 at 0.8μ while the ratio rises. That is consistent with a **near-constant relative policy** (small Jacobian / peaked discrete head), not with pegging at the env cap:

- Absolute orders still vary with incoming demand (order std ≈ 6–9 at 0.8μ).
- Inventory/signal sweeps often fail to flip the argmax `Δ` token ⇒ near-zero |Δ order| in the probe.
- Tight capacity (C = 0.8μ ≈ 6.2) makes local cost landscapes harsh; policies learn low-gain responses.

Muted inflation 0.22 → 0.28 is likewise **not** explained by cap flattening. On a **serial** chain, proportional rationing is identity fill for a single claimant — shortage *gaming* has no rival to steal from. Relative `Δ_max=8` also limits one-step inflation. Those are design/structural limits, not action-cap bugs.

### Cap recommendation (non-blocking)

Under planned AR(1) φ≈0.7 (high demand ≈ μ + 3σ_stat ≈ 16) plus bullwhip ratchets, absolute orders can exceed 64 (Sterman baseline already hits P95 = 64 on U[0,15]). Suggested for v1.1:

- Keep relative `Δ` for tractable learning.
- Raise env `order_cap` to **128** (or higher) so the hard clamp rarely binds under amplification.
- Prefer that over a continuous/log action space for Tier-1 comparability; log-surplus is optional if absolute discrete becomes too large for LLM tokenisation later.

**Fix:** none required for A1 (no bug). Cap raise is a v1.1 calibration choice, not a silent contaminant of M3.

**M3 invalidation:** no.

---

## Anomaly 2 — chronic backlog at C=∞ / D5 shortfall 0.70

### Suspects (in brief order)

#### 1. Pipeline / inventory initialization — **calibration issue, not a logic bug**

Classic init (DECISIONS): on-hand = 12, pipelines filled with **4** (classic step demand).

| Quantity | Value |
|---|---:|
| U[0,15] mean μ | 7.5 |
| Per-link replenishment delay (L_o+L_s) | 3 |
| Init inventory position (on-hand + ship pipe) | 20 |
| Approx installation base-stock S ≈ μL + zσ√L (z=1) | ~30 |

Init is systematically understocked for training demand. Pipelines seeded at classic demand 4, not μ = 7.5. This raises transient (and, with weak policies, ongoing) shortfall rates. **Recommend for v1.1:** scale `init_pipeline_*` and/or `init_inventory` to the demand process mean (and document).

#### 2. Off-by-one in ship / order delay — **not a bug**

Explicit single-unit trace (all else zero; wholesaler stocked so it can ship):

| Event | Expected (L_o=1, L_s=2) | Observed |
|---|---:|---:|
| Retailer orders 1 | week 1 | week 1 |
| Wholesaler sees order | week 2 | week 2 |
| Wholesaler ships | week 2 | week 2 |
| Retailer receives | week 4 | week 4 |

Factory production of 1 at week 1 is received at week 3 (= L_s=2). Matches PROJECT_SPEC / Sterman week order in `DECISIONS.md`. Regression: `test_delay_unit_trace_classic_beer_game`, `test_factory_production_delay`.

#### 3. Backlog accounting / conservation — **physical goods conserved**

Property check over random policies:  
`init_goods + cumulative_production = physical (inv + ship pipes) + delivered_to_customers`  
holds with zero failures. Regression: `test_property_goods_conservation`.

#### 4. Real bug found: **`on_order` init undercount**

**Reproduction.** After `reset()`, for every non-factory role:

```text
on_order = sum(ship_pipeline)           # was 8
# correct:
on_order = sum(ship_pipeline) + sum(upstream.order_pipeline) + upstream.backlog  # 12
```

Factory was already correct (`on_order = sum(ship_pipeline)`).

**Symptom.** Bias is a **permanent −4** for retailer/wholesaler/distributor for the entire episode (does not self-heal under normal play). Confirmed under constant and random order policies.

**Impact.**

- Physical transition, costs, shipments, backlogs: **unaffected** (`on_order` is not used in the fill equations).
- `RoleState.inventory_position()` and IPPO obs feature `on_order`: **biased low by 4**.
- Base-stock baselines over-order early (IP understated).

**Root cause.** `BeerGameCore.reset()` counted goods already in the ship pipeline but omitted orders still in the upstream **order-delay** pipeline (information in transit). Step-time updates (`on_order += placed`, `on_order -= received`) are consistent; only init was wrong. Retailer `order_pipeline` was also seeded with `init_pipeline_order` even though retailer incoming is customer demand — left a stale obs feature; now zeroed at init.

**Fix.** Applied in `beer_distribution_rl/env/core.py` `reset()`:

- Set `on_order` from ship pipe + upstream outstanding after all role states exist.
- Retailer `order_pipeline` init → zeros.

**Regression tests.**

- `test_on_order_init_includes_order_delay_pipeline`
- `test_property_on_order_invariant` (hypothesis)
- delay + conservation tests above

All `tests/` pass after the fix.

### Why D5 still shows ~70% allocation-shortfall at C=∞ (not a delay bug)

D5’s `allocation_triggers` = `any(role.backlog > 0)` — a high-sensitivity union over four nodes.

| Policy (C=∞, U[0,15]) | frac any-backlog | notes |
|---|---:|---|
| Learned IPPO (5 seeds) | ~0.74 | matches D5 |
| Pass-through | ~0.72 | |
| Base-stock z=1 (S≈30) | ~0.78 | |
| Base-stock z=2 (S≈38) | ~0.46 | |
| Base-stock z=3 (S≈46) | ~0.20 | backlog *can* be rare with enough safety stock |

So unlimited factory capacity does **not** imply rare backlog under highly variable demand and installation (not echelon) policies. Learned agents hold moderate inventory and look similar to pass-through / mild base-stock — mediocre control, not broken physics.

### Odd `infl|non-binding ≈ 1.52` at C=∞ — **not a bug**

At C=∞, `capacity_binds` is always false. D5’s `infl|non-binding` is therefore the **unconditional** mean of `factory_order / factory_incoming`. A ratio ≈ 1.5 is ordinary bullwhip amplification under local-cost agents, not evidence of rationing games or a capacity bug. When capacity *does* bind (tight caps), the same metric jumps to ~2.4–2.8 on binding weeks — the interesting contrast.

---

## M3 / Tier-1 contamination statement

| Claim | Contaminated? |
|---|---|
| Physical costs, bullwhip, backlog rates, D5 bind fractions | **No** (dynamics unchanged by `on_order` fix) |
| D2 signal ablation / babbling | **No** |
| D4 share-rate indifference | **No** |
| D6 demand uninformative | **No** |
| D3 listener sensitivity magnitudes | Mildly (obs includes biased `on_order`); qualitative “low absolute sensitivity at tight cap” still holds via A1 |
| Base-stock baseline numbers that use `inventory_position()` | **Yes** — re-run after fix |
| IPPO policies trained with biased `on_order` obs | **Mildly** — constant offset is largely absorbable by net biases; still **re-train for v1.1** before claiming new phase-diagram numbers |

**Do not discard all prior Tier-1 results.** There is no delay bug. Discard/recompute only quantities that depend on `on_order` / `inventory_position` baselines, and plan a clean retrain on the fixed env for v1.1 + AR(1) before M4.

---

## Fixes shipped

1. `BeerGameCore.reset()` — correct `on_order` init; zero retailer order pipeline.
2. Regression tests in `tests/test_core.py` (delay trace, conservation, on_order invariant).

## Hand-off to E1/E2 / env v1.1 (out of scope for B1, recorded for the queue)

- Scale init inventories/pipelines to demand mean (addresses C=∞ shortfall inflation).
- Consider `order_cap=128` with relative actions retained.
- Serial topology cannot stress-test multi-claimant shortage gaming — Y-topology remains necessary for P3.

---

*B1 complete. Env unblocked for E1/E2 once this report is accepted.*
