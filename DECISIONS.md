# Design decisions

Logged defaults for the v1 research codebase. Change only with a dated note.

| Decision | Default | Rationale |
|---|---|---|
| Python | 3.11+, package `beer_distribution_rl` | Spec / modern typing |
| Layout | Code under `beer_distribution_rl/{env,agents}/` | Enables `import beer_distribution_rl.env` |
| Classic eval horizon | T=36 | DQN-paper compare (Oroojlooy et al.) |
| Train default horizon | T=52 | PROJECT_SPEC §3.1 |
| Lead times | ship delay L_s=2, order delay L_o=1 | PROJECT_SPEC §3.1 |
| Action | absolute non-neg int order ∈ {0…128} | v1.1 (2026-07-12): raised from 64 per B1; AR(1)+relative Δ ratchets rarely bind |
| Classic costs | h=0.5, b=1.0 uniform all roles | Classic beer-game / DQN config |
| Classic demand | step 4→8 at week 5 (weeks 1–4 demand=4, then 8) | Classic MIT step |
| Training demand (v1.1) | AR(1) φ=0.7, μ=7.5, σ=2.0 (default); also `uniform`, `regime_switch`, `classic_step` | D6: U[0,15] lag-1 R²≈0; AR(1) R²≈0.47 — channel needs something to carry |
| Initial state | on-hand=12, backlog=0, pipelines filled with 4s | DQN-repo convention |
| Base-stock levels (DQN gate) | S=[9,5,3,1] retailer→factory | Oroojlooy et al. arXiv:1708.05924 §4 |
| DQN paper gate config | demand U{0,1,2}, L_s=L_o=2, ch=[2,2,2,2], cp=[2,0,0,0], S=[9,5,3,1], init 0 | Oroojlooy et al. §4 parameters |
| Base-stock cost target (calibrated) | mean cost/period ∈ **[4.5, 7.5]** under our Sterman-compatible event order | Empirically ~5.9 with S=[9,5,3,1]; published 2.008 uses their order-before-receive semantics — we keep Sterman week order (DECISIONS) and treat 2.008 as external reference, not a hard match |
| Gate also requires | base-stock ≪ Sterman on same DQN config (≥5×); retailer cost share ≥80% | Shape matches published allocation [1.91,0.05,0.02,0.03] |
| Research classic config | step 4→8, h=0.5, b=1.0, L_s=2, L_o=1 | PROJECT_SPEC §3.1 |
| Inventory position | on-hand − backlog + on_order (on_order += placed, −= received) | Required for correct base-stock |
| Sterman params | α_s=α_sl=0.5, β=1.0, θ=0.36 (demand smoothing) | Sterman 1989 anchoring-and-adjustment; documented for reproducibility |
| Week event order | receive shipments → receive orders → fill/ship → place orders → accrue costs | Sterman-compatible |
| Factory under C=∞ | production order enters factory ship pipeline (lead L_s) | Unlimited external supply |
| M1 scope | classic core + baselines + signals/rationing + PettingZoo; no IPPO/GRPO | Validation gate before training |
| Test stack | pytest + hypothesis; env has zero ML deps | Hub-ready pure Python env |
| Perf target | >10k `BeerGameCore.step()`/sec on laptop | PROJECT_SPEC §6 |

## Emergence constraints (non-negotiable)

- Regimes A/B: one policy (or LoRA) per role; no shared parameters or critics.
- Regimes A/B rewards: strictly local costs; no system-cost term, no cooperation shaping, no honesty reward.
- Signaling: optional and unverified; honesty is measured, never rewarded.
- Regime C (shared system reward) exists only as reproduction of arXiv:2605.17036.

## M2 — IPPO

| Decision | Default | Rationale |
|---|---|---|
| Algorithm | Independent PPO (CleanRL-style), MLP 2×256 | Spec Tier 1; auditable; laptop-friendly |
| Parameter sharing | Forbidden in A/B; also forbidden in C for M2 | Emergence claim; C only shares the *reward*, not weights |
| Shared critic | Forbidden — one critic head per role | Spec non-negotiable |
| M2 regimes | A and C only on classic step demand | Gate: cost ballpark vs Sterman/base-stock |
| Regime B / capacity matrix | M3: Regime B × caps × rationing; relative claims; honesty measured only | Phase diagram before GPU spend |
| M3 train timesteps (sweep default) | 50k / cell, ≥10 seeds on headline B cells | Laptop-feasible; extend later |
| Serial rationing equivalence | On serial topology, prop/uniform/honesty-weighted coincide (one claimant) | P3 needs Y-topology; documented in M3_REPORT |
| M3 matrix completed | B × 5 caps × 2 ration × 10 seeds × 50k steps | Phase diagram v1; P1 weak, P2 mixed, P3 N/A |
| Action space | relative: order = clip(demand + Δ, 0, 128), Δ ∈ [-8, 8] | v1.1: cap 128 (B1); relative Δ retained for tractability |
| Reward scale | 0.1 (training only; logged costs unscaled) | Stabilizes value learning |
| Reproducibility | YAML + seed + git SHA in `run_meta.json` | Spec §6 |
| Demand info value | logged to `demand_info_value.json` / `run_meta.json` at train start | Paper justification for env v1.1 demand change |
| Boundary actions | `eval/frac_actions_at_cap`; warn if >5% | Healthy runs ≈0; saturation contaminates sensitivity |

## Tier-1 v1.1 matrix (Agent R1)

| Decision | Default | Rationale |
|---|---|---|
| Matrix axes | {A,B,C} × {serial,y} × {∞,1.2μ,1.0μ,0.8μ} × {prop,uniform,honesty_w} × {ar1,regime_switch} × ≥10 seeds | Close D1 (A vs B); P3 needs Y; informative demand |
| Dropped cap | 1.5μ removed from M3 grid | Concentrate on binding region; ∞ retained as slack anchor |
| Prune: C=∞ rationing | Keep **proportional only** at ∞ | Capacity never binds ⇒ rationing policies identical |
| Prune: serial rationing | Keep **proportional only** on serial | Single claimant ⇒ prop≡uniform≡honesty_weighted (DECISIONS M3) |
| Kept after prune | **840** cells (was 1440 full cartesian @ 10 seeds) | `prune_summary.json` written each run |
| Y × ar1 demand | `CorrelatedYDemand` (shared factor) | Rival broadcasts informative; matrix label stays `ar1` |
| Y × regime_switch | `TwinCustomerDemand(RegimeSwitch)` | Independent streams; ablation vs correlated factor |
| Cell parallelism | 8–16 processes (`ProcessPoolExecutor`) | Dominant speedup; CPU-bound MLPs |
| Vec envs / cell | `n_envs=64`, `rollout_steps=128` (batch 8192) | GPU idle otherwise; no JAX rewrite |
| Timesteps / cell | **400k** (was 50k in M3) | With vec envs, 50k ⇒ only ~6 PPO updates; 400k/(64×128)≈49 matches M3 learning budget |
| Device | cpu default; `cuda`/`mps` via CLI/Colab | Move nets to GPU only after vec envs |
| Resumability | skip cell if `final_eval.json` exists | Colab sessions die; checkpoint per cell |
| Bind event logs | `eval/frac_capacity_binds`, `eval/frac_allocation_triggers`, `week_events.json` | Closes D5 log gap (no checkpoint recompute) |
| Artifacts | `config.yaml` + `run_meta.json` (seed, git SHA) per run | Spec §6 |
| Runner | `scripts/run_tier1_matrix.py`; Colab: `notebooks/colab_tier1.ipynb` | Coffee-break target ≪15 min on Colab |
