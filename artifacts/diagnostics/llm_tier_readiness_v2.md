# LLM-tier readiness audit v2

**Branch:** `preflight/llm-tier-readiness-v2`  
**Baseline SHA:** `76ce8a1f978f39c88667b518240d5187879de13e` (`git rev-parse HEAD` at branch creation = `main` after merges of `baseline/recurrent-ippo`, `feat/llm-text-io`, `feat/llm-rolling-context`)  
**Date:** 2026-07-15  
**Scope:** Re-verify v1 blockers against merged product artifacts. **Inference / inspection only. No GRPO. $0 spent.**

**Prior audit:** `preflight/llm-tier-readiness` → `artifacts/diagnostics/llm_tier_readiness.md`  
**Overall then:** `BLOCKED-ON-{product-text-I/O, parse-fail productization, memory-confound protocol, GRPO budget plan}`

---

## Overall verdict

### **READY-FOR-FIRST-GRPO-CELL**

| Prior blocker | Re-verify | Verdict |
|---|---|---|
| 2 Text I/O | Product serializer + constrained parser; parse-fail 0% uniform | **GO** |
| 3 Context | Rolling W=8 productized; T=52 persistence; fits 32k | **GO** |
| 4 Memory confound | Recurrent IPPO baseline exists; shared own-history info set | **GO** |
| 7 Budget | Corrected W=8 × 1/(1−p)=1.00 ≈ **$89** for 9-cell lean plan | **GO** |
| Non-negotiables (port) | One LoRA/role; local cost reward; no B signaling; no C system reward | **GO** |

All former NO-GOs cleared by merged branches. This audit does **not** launch GRPO — it specs the first cell for human review.

---

## Blocker 2 — Text I/O — **GO**

**Cite:** `artifacts/diagnostics/llm_text_io.md` (branch `feat/llm-text-io`, merged).

| Requirement | Evidence |
|---|---|
| Serializer exists | `beer_distribution_rl/agents/llm/serializer.py` (+ `memory.py`) |
| Parser exists | `grammar.py`, `parser.py`, `decode.py` — JSON-schema / GBNF constrained Δ ∈ [-8, 8] |
| Round-trip clean | 3 real logged obs examples; unit tests `tests/test_llm_text_io.py` |
| E1 no-leak in text | Leak report empty on round-trips; own-history only; no `customer_demand` / `true_demand` / rival private state |
| Parse-fail ≪ ~5% | **0.0%** after schema constraint (0/780 pooled) |
| Uniform across capacity × role | All 15 cells (3 caps × 5 roles) at **0%**; max−min = 0 |

| Cap | Before (Check 5 regex) | After (schema) |
|---|---:|---:|
| ∞ | 29.6% | **0.0%** |
| 1.0μ | 39.2% | **0.0%** |
| 0.8μ | 36.6% | **0.0%** |

Resampling multiplier \(1/(1-p)\) = **1.00**. No scarcity-concentrated parse bias → no selection into shortage-gaming via resampling.

**NO-GO threshold not met.** → **GO**.

---

## Blocker 3 — Context — **GO**

**Cite:** `artifacts/diagnostics/llm_rolling_context.md` (branch `feat/llm-rolling-context`, merged).

| Requirement | Evidence |
|---|---|
| Rolling-W8 productized | `DEFAULT_ROLLING_WINDOW = 8` in `agents/llm/memory.py`; `serialize_prompt(..., window=)` |
| Persistence over T=52 | **PASS** (`persistence_misses=[]`); prior week present each step; window drop verified |
| Fits context | Steady W=8 mean **538** / max **540** tok ≪ Qwen2.5-3B **32 768** (<2%) |
| E1 | Leak hits **0**; own history only |

**NO-GO threshold not met.** → **GO**.

---

## Blocker 4 — Memory confound — **GO**

**Cite:** `artifacts/diagnostics/recurrent_baseline.md` (branch `baseline/recurrent-ippo`, merged).

**Memory-matched baseline exists:** GRU IPPO (`RecurrentActorCritic`, one module per role), Regime A headline cells under the same R1 budget.

**Shared information set (apples-to-apples):**

| Field (own history only) | Recurrent IPPO | Order-only LLM (W=8) |
|---|---|---|
| `demand_or_incoming` / `last_demand_or_order` | Yes | Yes |
| `ship_in` / `alloc_recv` / `last_shipment_received` | Yes | Yes |
| `ordered` / `last_order_placed` | Yes | Yes |
| `inv`, `backlog`, pipelines, `on_order` | Yes | Yes |
| Rival private inventories / privileged true demand | **No** | **No** |
| Cheap-talk board | **Off** (Regime A) | **Off** (order-only) |
| Reward | Strictly local cost | Strictly local cost (GRPO scalar) |

**Memory asymmetry check:** The v1 confound was LLM-with-history vs **Markovian** MLP (LLM strictly more). That is resolved: the comparison baseline is now recurrent. With productized **W=8**, the LLM’s *explicit* prompt window is shorter than the GRU’s full-episode hidden-state capacity — the LLM does **not** have strictly more memory than the baseline it will be compared against. (Form differs: exact text window vs compressed GRU state; field set matches.)

**NO-GO threshold not met.** → **GO**.

---

## Blocker 7 — Budget — **GO**

**Cite:** `LEDGER.md` projection row + `artifacts/diagnostics/llm_rolling_context.md`.

Lean schedule: **9 cells** (Y × {∞, 1.0μ, 0.8μ} × prop × 3 seeds) × T=52 × 5 roles × G=4 × **50 updates** × 4090 @ $0.50/hr.

| Line | Est. $ | vs $250 |
|---|---:|---|
| Corrected W=8 steady (538 tok) × resampling **1.00** | **~$89** | Fits (~$161 margin) |
| Naive 600 tok × 1.00 | ~$99 | Fits |

Prior full-history ~$870 and W=8@200-upd ~$395 NO-GOs are obsolete under the productized window + lean update count + p=0.

**NO-GO threshold not met.** → **GO**.

---

## Non-negotiables (GRPO port) — **GO**

| Constraint | Port requirement | Status |
|---|---|---|
| One LoRA adapter per role | Never share adapters / joint update across retailer_a/b/… | Spec’d; mirrors IPPO `policies[r]` |
| GRPO reward scalar | **Strictly local** per-agent cost (−local_cost); not system | Spec’d |
| No Regime-B signaling reused | Order-only; `signaling_enabled=False`; no claim/broadcast heads | Spec’d (Regime A / B-equivalent order stream) |
| No Regime-C system reward | Do not train the first cell under Regime C | Spec’d |

**NO-GO if any violated — none would be under the first-cell spec below.** → **GO**.

---

## First GRPO cell — SPEC ONLY (do not run)

Smallest justified probe: does RL pressure on local cost **create or erode** shortage-gaming / order-signal behavior under scarcity, without cheap talk?

### Cell definition

| Axis | Choice | Why |
|---|---|---|
| Design | **Regime B-equivalent order-only** (Regime **A** env; signaling off) | Channel dropped; phenomenon lives in orders |
| Topology | **Y** | Rival claimants; shortage-gaming supported on Y |
| Capacity | **1.0μ** | Binding scarcity without the most extreme 0.8μ cell; mid of the phase diagram |
| Rationing | **proportional** | Prop>uniform gaming contrast; prop is the headline rule |
| Demand | AR(1) / matrix `ar1` (CorrelatedYDemand) | Informative demand; Tier-1 parity |
| Seeds | **3** (0, 1, 2) | Minimal replication; 3/9 of the lean 9-cell plan |
| Model | **Qwen2.5-3B** | Check 5 / text-I/O smoke parity |
| Adapters | **One LoRA per role** (5 adapters) | Non-negotiable |
| Context | **W=8** rolling own-history | Product default; budget-feasible |
| Decode | JSON-schema / GBNF Δ ∈ [-8, 8] | Parse-fail ~0; factor 1.00 |
| Horizon / group | T=52, G=4, **50 updates** | LEDGER lean schedule |
| Reward | Per-agent **local cost** only | Non-negotiable |

**Not in this cell:** Regime B signaling, Regime C, honesty-weighted rationing, ∞ / 0.8μ caps, >3 seeds, W≠8 ablations.

### Estimated cost (from LEDGER / rolling-context arithmetic)

Full lean projection: **~$89** for 9 (cap × seed) cells.  
This first cell = **3 seeds × one cap** = **3/9** of that schedule:

| Quantity | Value |
|---|---|
| Est. cost | **~$30** (= $89 × 3/9) |
| Est. GPU-h | **~59** @ $0.50/hr |
| Remaining headroom vs $250 after cell | **~$220** (if actual ≈ est.) |
| This audit spend | **$0** |

Append a real LEDGER row **before** any paid launch; do not treat this projection as a spend.

### Success / kill criteria

**Primary question:** After GRPO, does order behavior under 1.0μ×prop×Y show RL-driven change in shortage-gaming / order-inflation relative to the prompted (pre-GRPO) floor?

| Outcome | Criterion |
|---|---|
| **SUCCESS (continue)** | Matched-deterministic eval shows a **clear directional move** in the order-inflation / scarcity-gaming metrics used in Tier-1 (e.g. gap vs base-stock S=30, mean order, scarcity contrast if ∞ control later) **and** local-cost learning is non-degenerate (episode cost not stuck at resample/fallback noise). Direction may be *create* (more gaming) or *erode* (less) — either answers the scientific question. |
| **SUCCESS (weak / extend)** | Cost improves vs prompted floor but gaming metrics flat within seed noise → optional +2 seeds or add ∞ / 0.8μ contrast **only after** review; do not expand blindly. |
| **KILL — no RL signal** | After 50 updates × 3 seeds: no reliable change in cost **and** no change in order-inflation metrics vs prompted baseline (within seed scatter) → stop; do not burn the remaining ~$220 on a wider matrix. |
| **KILL — collapse** | Parse-fail rises above ~5% under train decode, or >5% actions pin at order cap, or valid-order rate collapses → stop and fix productization before more spend. |
| **KILL — protocol violation** | Any shared LoRA across roles, system-cost reward, Regime-B signaling enabled, or Regime-C training → **invalid run**; discard; do not interpret. |
| **KILL — budget** | Actual path cost trending >~$50 for this 3-seed cell (≪ 2× est.) without early SUCCESS signal → pause for human review. |

**Comparisons allowed after SUCCESS:** vs memory-matched recurrent IPPO on the same Y×1.0μ×prop×AR(1) cells (`recurrent_baseline.md`) — same information set. Do **not** claim language-prior wins against the Markovian MLP alone.

---

## Deliverable checklist

- [x] SHA recorded (`76ce8a1f978f39c88667b518240d5187879de13e`)
- [x] GO/NO-GO per former blocker + non-negotiables
- [x] One overall verdict: **READY-FOR-FIRST-GRPO-CELL**
- [x] First-cell SPEC with cost (~$30) + kill criteria
- [x] `$0` this audit; GRPO **not** launched
- [ ] Launch — **awaiting human review**

---

## References

- v1 audit: `preflight/llm-tier-readiness` / `llm_tier_readiness.md` (if present on that branch)
- Text I/O: `artifacts/diagnostics/llm_text_io.md`
- Rolling context + budget: `artifacts/diagnostics/llm_rolling_context.md`, `LEDGER.md`
- Memory-matched baseline: `artifacts/diagnostics/recurrent_baseline.md`
- Emergence constraints: `DECISIONS.md`
