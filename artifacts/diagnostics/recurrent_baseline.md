# Recurrent IPPO baseline (memory-matched MLP)

**Branch tip SHA:** `c56fc82f4def8d76e37611bd043cc9917bd9e7e7`  
**Branched from main (`git rev-parse HEAD` at branch creation):** `061aa59235397b7360c32a01cf4f98add0dd503a`

## Architecture choice

**Chose: GRU over per-week local observations** (not stacked-history concatenation).

Rationale:
- Matches the planned LLM's full T=52 own-history retention without blowing up the observation dimension (W×obs_dim).
- Fits the existing R1 runner (YAML + seed + git SHA, vec envs, matched-deterministic greedy eval) with hidden-state carry + reset on done.
- Single-step BPTT with stored (detached) input hiddens keeps the CleanRL-style shuffled minibatch update intact.

Policy: `RecurrentActorCritic` — GRU(obs_dim→128) → separate 2×256 actor/critic MLPs. One module per role; no parameter sharing.

## Shared information set (apples-to-apples vs LLM)

Both the recurrent MLP and the planned order-only LLM see **own history only** (E1 no-leak). Per-week content aligned with Check 3 structured history:

| Check 3 history field | IPPO local obs / GRU input |
|---|---|
| `demand_or_incoming` | `last_demand_or_order` |
| `ship_in` / `alloc_recv` | `last_shipment_received` |
| `ordered` | `last_order_placed` |
| `inv`, `backlog` | `inventory`, `backlog` |
| `cost` | recoverable via `h`,`b` coeffs in obs × inv/backlog |

Plus pipelines, `on_order`, `t/horizon`. **Never** rival private inventories or privileged `customer_demand`/`true_demand` for upstream agents. Cheap-talk board is off (Regime A). Rewards remain strictly local (no system term, no honesty reward).

## Markovian vs recurrent cost (Regime A × prop × AR(1), 10 seeds)

Matched-deterministic `final_eval` (`n_episodes≥20` at train end). Markovian n=60 cells; recurrent n=60 headline cells from `artifacts/runs/ippo/tier1_v11` vs `.../recurrent_baseline` (plus 20 Y×uniform recurrent cells for gaming rule contrast).

| Topo | Cap | Markovian (mean±CI95) | Recurrent (mean±CI95) | Δ (rec−mark) |
|---|---|---:|---:|---:|
| serial | ∞ | 29.4±1.4 | 60.4±13.9 | +31.0 |
| serial | 1.0μ | 627.7±52.7 | 921.9±101.9 | +294.2 |
| serial | 0.8μ | 528.3±116.5 | 934.7±147.5 | +406.4 |
| y | ∞ | 95.2±2.7 | 150.6±29.5 | +55.5 |
| y | 1.0μ | 1458.2±80.3 | 1388.6±119.5 | −69.6 |
| y | 0.8μ | 1352.3±39.6 | 1585.6±77.5 | +233.2 |

**Memory-only finding:** Under the matched R1 budget (400k steps, ~49 PPO updates), giving the MLP a GRU **does not reduce cost** — five of six headline cells get *worse* (mean |Δ|≈182; only Y×1.0μ is slightly better within CI overlap). History alone is **not** a free lunch for this IPPO setup; the recurrent run is still the correct memory-matched reference for later LLM comparisons (any LLM edge must beat *this*, not the Markovian MLP).

## Shortage-gaming recheck (recurrent vs Markovian, Regime A × Y × AR(1))

Order inflation gap vs base-stock S=30; matched-deterministic re-roll (`greedy=True`, seed+10000, 20 eps).

| Arch | Cap | Rationing | Gap (order−S*) | Ratio | Mean order | Frac@128 |
|---|---|---|---:|---:|---:|---:|
| markovian | ∞ | proportional | 4.5±0.4 | 4.2±0.3 | 7.7±0.0 | 0.0±0.0 |
| markovian | 1.0μ | proportional | 14.3±0.8 | 14.3±0.7 | 14.9±0.5 | 0.0±0.0 |
| markovian | 1.0μ | uniform | −0.5±9.1 | 7.3±2.3 | 10.5±1.3 | 0.0±0.0 |
| markovian | 0.8μ | proportional | 12.1±0.7 | 12.6±0.6 | 13.8±0.4 | 0.0±0.0 |
| markovian | 0.8μ | uniform | −9.6±10.5 | 5.8±1.7 | 9.5±1.1 | 0.0±0.0 |
| recurrent | ∞ | proportional | 6.7±1.9 | 7.5±0.9 | 8.7±0.5 | 0.0±0.0 |
| recurrent | 1.0μ | proportional | 15.0±0.3 | 14.9±0.3 | 15.2±0.3 | 0.0±0.0 |
| recurrent | 1.0μ | uniform | 0.7±12.2 | 10.8±2.3 | 12.1±1.7 | 0.0±0.0 |
| recurrent | 0.8μ | proportional | 14.9±0.3 | 14.8±0.3 | 15.2±0.3 | 0.0±0.0 |
| recurrent | 0.8μ | uniform | −13.0±19.4 | 7.7±3.6 | 10.2±2.4 | 0.0±0.0 |

**Recurrent gaming verdict:** `supported` — scarcity Δ(0.8μ−∞) under prop = **+8.23** (ok=True); mean prop−uniform @ tight = **+21.11** (ok=True).  
Markovian reference scarcity Δ = +7.60. History alone does **not** suppress shortage gaming; scarcity response is similar and the prop>uniform rule contrast remains (if anything, larger under recurrent, though uniform CIs are wide). Phenomenon survives the architecture change.

## Non-negotiables (unchanged)

- One policy per role (A); independence asserted at init.
- Strictly local per-agent cost rewards; no system term; no honesty reward.
- Recurrence changes **memory**, not reward or information-leak rules.

## Artifacts

- Runs: `artifacts/runs/ippo/recurrent_baseline/` (80 cells, 1191s wall, 8 workers)
- Cache: `analysis/diag/cache/recurrent_baseline.json`
- Train log: `artifacts/runs/ippo/recurrent_baseline_train.log`
