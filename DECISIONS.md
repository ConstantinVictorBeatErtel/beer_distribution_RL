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

## B′ control / M4 gate (2026-07-15) — `exp/bprime-control`

| Decision | Default | Rationale |
|---|---|---|
| B′ train matrix | **Skipped** (0 h / $0) | Prompt gate: only train B′ if Prompt 2 ablation shows channel plausibly load-bearing |
| Prompt 2 result | Channel **not** load-bearing; shuffles inert under det. eval | B scarcity gap largely eval-mode confound; see `v11_ablation.md` / `v11_bprime.md` |
| Causal prior without B′ | Treat **B ≈ B′** as the supported framing | Babbling / negative-result writeup; do not claim information-flow advantage |
| **M4 LLM gate** | **Not cleared** | No demonstrated load-bearing cheap-talk channel; revisit B′/M4 only after a checkpoint fails shuffle-time |

## Recurrent IPPO baseline (2026-07-15) — `baseline/recurrent-ippo`

| Decision | Default | Rationale |
|---|---|---|
| LLM-vs-MLP comparison | **Memory-matched** | Recurrent GRU IPPO retains own T=52 history matching planned LLM context (Check 4); removes memory confound from language-prior claims |
| Recurrent architecture | GRU(obs→128) + 2×256 actor/critic per role | Preferred over frame-stack; fits R1 runner; E1 own-history only |
| Headline cells | Regime A × {serial,Y} × {∞,1.0μ,0.8μ} × prop × AR(1) × 10 seeds (+ Y×uniform for gaming) | Order-only setting matching LLM channel drop |
| Memory-only finding | GRU does **not** cut cost vs Markovian under matched budget; shortage gaming **survives** | See `artifacts/diagnostics/recurrent_baseline.md` |

## LLM text I/O (2026-07-15) — `feat/llm-text-io`

| Decision | Default | Rationale |
|---|---|---|
| Product path | `beer_distribution_rl/agents/llm/` | Clears readiness blocker 2; provisional smoke harness is not a substitute |
| Information set | Own history only (Check 3 / recurrent match); order-only (no cheap talk) | E1 no-leak in text; apples-to-apples vs GRU baseline |
| Action emit | Relative Δ ∈ [-8, 8] → clip(demand+Δ, 0, 128) | IPPO parity (DECISIONS M2 / Check 0) |
| Decoding | **JSON-schema constrained** (Ollama `format` / GBNF twin for vLLM); resample≤3 logged | Check 5 regex parse-fail 30–39% → **0%** uniform across cap×role; see `llm_text_io.md` |
| Resampling multiplier | **1.00** (= 1/(1−p) at p=0) | Feeds corrected GRPO budget; drop ~1.5× parse tax from Check 7 drafts |
| Spend on this branch | **$0** / no GRPO | Inference-only local Qwen2.5-3B smoke |

## LLM rolling context (2026-07-15) — `feat/llm-rolling-context`

| Decision | Default | Rationale |
|---|---|---|
| Context retention | **Rolling own-history window W=8** (`DEFAULT_ROLLING_WINDOW`) | Clears readiness blockers 3+7; full history fits 32k but GRPO full-hist ~$870 over $250; W=8 ≈$89 corrected |
| Ablation knob | `AgentMemory.window` / `serialize_prompt(window=)` | One-line W-sensitivity later (mirrors recurrent “does memory help?”) |
| Per-prompt tokens @ W=8 | **~538 mean / 540 max** (chars/4) | ≪ Qwen2.5-3B 32k; persistence PASS on T=52 |
| GRPO budget (projection) | **~$89** after ×**1/(1−p)=1.00** (p=0 from `llm_text_io.md`) | 9 cells × 50 upd × G=4 × measured W8 tokens × 4090 @$0.50/hr; fits $250 w/ margin; **$0 spent / no GRPO** |
| E1 | Own history only in prompt | No other-role private state; leak hits 0 |

## LLM-tier readiness v2 (2026-07-15) — `preflight/llm-tier-readiness-v2`

| Decision | Default | Rationale |
|---|---|---|
| Baseline SHA | `76ce8a1f978f39c88667b518240d5187879de13e` | `git rev-parse HEAD` at branch tip (= `main` after recurrent-baseline + text-I/O + rolling-context merges) |
| Blocker 2 (text I/O) | **GO** | Product serializer+parser; parse-fail **0%** uniform cap×role (`llm_text_io.md`) |
| Blocker 3 (context) | **GO** | Rolling W=8 productized; T=52 persistence PASS; ~538 tok ≪ 32k (`llm_rolling_context.md`) |
| Blocker 4 (memory) | **GO** | Recurrent IPPO baseline exists; shared own-history info set; LLM W=8 not strictly more memory than GRU (`recurrent_baseline.md`) |
| Blocker 7 (budget) | **GO** | Corrected W=8 × 1.00 resampling ≈ **$89** / 9 cells fits $250 (`LEDGER.md`) |
| Non-negotiables | **GO** | One LoRA/role; local-cost GRPO scalar; no B signaling; no C system reward |
| **Overall readiness verdict** | **READY-FOR-FIRST-GRPO-CELL** | All v1 blockers cleared; details: `artifacts/diagnostics/llm_tier_readiness_v2.md` |
| First cell (SPEC only) | Y × Regime A order-only × **1.0μ** × prop × AR(1) × **3 seeds** × Qwen2.5-3B × LoRA/role × W=8 × G=4 × 50 upd | Smallest probe of RL vs order-signal/shortage-gaming; est. **~$30** (3/9 of $89); kill if no RL signal / collapse / protocol breach |
| GRPO | **Not started** | $0 this audit; launch only after human review |
