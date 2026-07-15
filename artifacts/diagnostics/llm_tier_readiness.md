# LLM-tier readiness audit

**Branch:** `preflight/llm-tier-readiness`  
**Baseline SHA:** `061aa59235397b7360c32a01cf4f98add0dd503a` (`git rev-parse HEAD` at branch tip = `main`)  
**Date:** 2026-07-15  
**Scope:** Inspection + $0 prompted-only smoke (local Ollama `qwen2.5:3b`). **No GRPO. No paid spend. No training code.**

**Design under audit (order-signal pivot):**
- Broadcast / cheap-talk channel is **DROPPED** for the LLM tier.
- LLM agents emit an **ORDER** only; rivals observe them only via the order stream + retained own context.
- Agents **retain context across weeks** (history in each week’s prompt).

---

## Overall verdict

### **BLOCKED-ON-{product-text-I/O, parse-fail productization, memory-confound protocol, GRPO budget plan}**

| Check | Verdict |
|---|---|
| 0 Orient | **GO** |
| 1 Phenomenon-exists (shortage gaming) | **GO** (supported on sibling branch; merge artifact to main) |
| 2 Text I/O layer | **NO-GO** (product `agents/llm/` absent; provisional audit harness only) |
| 3 Context retention | **NO-GO** (design + smoke OK; no production memory module) |
| 4 Memory confound | **GO** (documented; must not silently compare) |
| 5 Prompted capability floor | **GO-WITH-CAUTION** (valid orders every week; 3.2–4.4× base-stock; parse-fail 30–39%) |
| 6 Non-negotiables | **GO** (IPPO scaffolding OK; LLM must not reintroduce shared policy / system reward / verified signals) |
| 7 Budget projection | **NO-GO as-planned** (full-history GRPO exceeds $250; rolling window required) |

Shortage-gaming in Tier-1 is **supported**, so LLM-tier spend is **not** premature on the phenomenon gate. Readiness is still blocked on missing product text I/O, an explicit MLP-vs-LLM memory protocol, and a budget-feasible context strategy.

---

## Check 0 — Orient — **GO**

### IPPO observation (`beer_distribution_rl/agents/ippo/obs.py`)

`state_to_obs(state, role, core) → float32` vector. Local info only (no other roles’ true inventories).

| Block | Fields | Scaling |
|---|---|---|
| 0–5 | `inventory`, `backlog`, `on_order`, `last_demand_or_order`, `last_shipment_received`, `last_order_placed` | ÷20 |
| next `ship_delay` (2) | ship pipeline slots | ÷20 |
| next `order_delay` (1) | order pipeline slots | ÷20 |
| next 3 | `t/horizon`, `holding`, `backlog` cost coeffs | unscaled |
| if signaling | per board role: `[present, claimed_demand/20, claimed_inventory/20]` | — |

**Shapes:** serial A/C → **12**; serial B → **24**; Y A/C → **12**; Y B → **27**.

Core `BeerGameCore.observe()` exposes the same local dict fields (plus `inventory_position`, `order_cap`, …). E1: upstream never sees true consumer demand under a privileged key.

### Order action space

- Env: absolute discrete `{0,…,order_cap}` with `order_cap=128` (code/DECISIONS; spec still says 64).
- IPPO default: **relative** Δ ∈ `[-8,+8]` → 17 bins, decoded as `clip(last_demand_or_order + Δ, 0, 128)`.
- Regime B IPPO also has broadcast / claim heads — **out of scope for LLM tier** under the order-only design.

### Per-agent local cost

Each week: `local_cost[r] = h_r * inventory_r + b_r * backlog_r`.  
Regimes A/B: `reward[r] = −local_cost[r]`. Regime C: all get `−system_cost` (reproduction only). Classic `h=0.5`, `b=1.0`.

### Episode loop / horizon T

Sterman week in `BeerGameCore.step`: receive ships → receive orders / demand → fill/ship (ration) → place orders → (optional cheap talk) → accrue costs → `t += 1`; terminate when `t >= horizon`.

| Config | T |
|---|---|
| **Standard train / Tier-1 v1.1** | **52** |
| Classic / DQN eval configs | 36 |

**Standard episode length for this audit:** **T = 52 weeks**.

---

## Check 1 — Phenomenon-exists gate — **GO**

**Expected artifact:** `artifacts/diagnostics/shortage_gaming.md`  
**On this branch / `main`:** **absent** (not yet merged).  
**On sibling branch `analysis/shortage-gaming` (commit analyzing same baseline SHA `061aa592…`):** **present**.

**Verdict from that report:** **`supported`**.

> Inflation rises with tightness (Δ=8.80) and is higher under proportional than uniform (mean Δ=6.79).

Deciding numbers (B×Y×AR(1), matched-deterministic eval):
- Scarcity: gap(∞)=3.76 → gap(0.8μ)=12.55 (**Δ=+8.80**).
- Rule contrast: mean prop−uniform @ tight = **+6.79**.
- Serial no-rival control scarcity Δ=**−4.82** (does not mirror Y).
- A≈B on Y×prop ⇒ gaming does **not** need the broadcast channel (aligns with dropping cheap talk for LLM tier).

**Gate interpretation:** The order-signal phenomenon the LLM tier would study **is confirmed in Tier-1**. LLM-tier spend is **not** premature on this gate.  
**Follow-up:** merge `analysis/shortage-gaming` so `artifacts/diagnostics/shortage_gaming.md` lands on `main` before any paid LLM work.

---

## Check 2 — Text I/O layer — **NO-GO**

### Product status

| Spec item | Status |
|---|---|
| `beer_distribution_rl/agents/llm/` | **Missing** |
| obs → prompt serialization | **Missing in product** |
| LLM text → order parser + strict grammar | **Missing in product** |
| Parse-failure resampling | **Missing in product** |
| Logged parse-failure rate | **Missing in product** |

### Provisional audit harness (not a substitute)

`scripts/diag/llm_tier_smoke.py` implements a **provisional** order-only serialize/parse for this audit:
- Grammar: exactly one line `ORDER: <int>` with `0 ≤ int ≤ order_cap`.
- Resample up to 3 times on parse failure; then fall back to `last_demand_or_order` (logged as exhausted retries).
- Round-trip on real Y-topology obs: **12/12** parser cases OK.

**Example (week 0, `retailer_a`):**

```
inventory=12 backlog=0 on_order=12 inventory_position=24
last_demand_or_order=0 ship_pipeline=[4, 4] order_pipeline=[0]
→ strict output: ORDER: <integer>
```

**Example parse:**
| Raw | Parsed |
|---|---|
| `ORDER: 12` | 12 |
| `I think we should ORDER: 5` | 5 |
| `twelve cases please` | fail |
| `ORDER: 133` (cap 128) | fail |

### Spec for product `agents/llm/` (gaps)

1. **Prompt fields (own state only):** role name, `h`/`b`, `t`, `inventory`, `backlog`, `on_order`, `inventory_position`, `last_demand_or_order`, `last_shipment_received`, `last_order_placed`, `ship_pipeline`, `order_pipeline`, `order_cap`. **No** signals / broadcast. **No** other agents’ private state. **No** `customer_demand` / `true_demand` keys (E1).
2. **History block:** structured own-week records (see Check 3).
3. **Output grammar:** `^ORDER:\s*(\d+)\s*$` (optionally allow prefix noise then extract once); reject OOR.
4. **Resample** on failure (N=3 default); **log** per-role parse-failure rate every eval / GRPO update.
5. **No** signal / claim heads in the LLM action schema.

---

## Check 3 — Context retention — **NO-GO** (design+smoke only)

### (a) What is retained — **spec (provisional smoke implements structured state)**

| Option | Choice |
|---|---|
| Full free-form transcript | **No** (token blow-up; format drift) |
| **Structured own-history** | **Yes** — per week: demand/incoming seen, ship in, order placed, allocation received, ending inv/backlog, local cost |
| Running summary | Optional later if rolling window still too large |

### (b) Context-window budget vs Qwen2.5-3B (32k)

Estimates for **structured** history (role card ~220 tok + state ~90 tok + ~40 tok/week line):

| Quantity | Value |
|---|---|
| Tokens / week (prompt @ week 0) | ~310 |
| Tokens / week (prompt @ week 51, full history) | ~2,350 |
| Mean prompt tokens / role / week | ~1,330 |
| Episode prompt tokens (5 Y roles × 52) | ~3.5×10⁵ |
| Fits full-history in 32k for one prompt? | **Yes** (max ~2.4k ≪ 32k) |

**Smoke char/4 estimate** (~220k prompt-tok/episode across 5 roles) is consistent with ~850 mean tok/role/week including system chat wrapper.

**Conclusion:** Full-history prompts **fit** the context window for T=52. A rolling window is **not** required for context length — it **is** required for **budget** (Check 7).

### (c) Persistence — **verified in smoke**

`persistence_ok=true` on all three episodes: week-t+1 prompt contains week-t `ordered=…` from own history. Example from smoke (week 1 prompt fragment):

```
Own history (prior weeks):
  week=0: demand_or_incoming=0, ship_in=0, ordered=12, alloc_recv=4, inv=9, backlog=0, cost=4.50
```

### (d) No-leak — **holds in provisional text**

Smoke leak heuristic: **0 hits** across all prompts. Forbidden substrings (`customer_demand`, `true_demand`, …) absent; no other-role `inventory=` patterns. Product must re-run E1-style tests on the text serializer (mirror `tests/test_demand.py`).

---

## Check 4 — Memory confound — **GO** (document; do not fix here)

IPPO baseline is a **Markovian MLP** (pipelines in the flat obs only). LLM agents with retained history have **strictly more information**.

| Information at decision week t | IPPO MLP (current) | Prompted / GRPO LLM (proposed) |
|---|---|---|
| Current local state (inv, backlog, on_order, pipelines, last_*) | Yes (in obs vector) | Yes (in prompt) |
| Cost coeffs, t/horizon | Yes | Yes |
| Delayed cheap-talk board | Yes if Regime B | **No** (channel dropped) |
| Explicit multi-week trajectory of own orders / costs / fills | **No** (unless engineered into obs / GRU — not built) | **Yes** (retained history) |
| Rival private inventories / true consumer demand (upstream) | **No** | **No** |
| Rival orders as observable env consequences | Only via own incoming orders / fills | Same, plus own memory of past interactions |

**Implication for LLM-vs-MLP comparisons:** any cost / inflation gap may be **memory**, not language priors or GRPO.

**Honest options (pick one before headline claims):**
1. Give the MLP history / recurrence (frame stack or GRU) matched to the LLM context window; **or**
2. Frame the LLM edge as **partly memory-driven** and report a memory-ablated LLM (history off) as a control.

**Do not** silently publish asymmetric LLM-vs-MLP tables.

---

## Check 5 — Prompted capability floor — **GO-WITH-CAUTION**

**Setup:** local Ollama `qwen2.5:3b`, $0. Y-topology, Regime A (order-only), proportional rationing, AR(1), T=52, seed 0, caps `{∞, 1.0μ, 0.8μ}`. All 5 roles prompted. Artifact: `artifacts/diagnostics/llm_tier_smoke.json`.

| Cap | System cost (LLM) | Base-stock S=30 | Ratio | Parse-fail rate | Valid order every week | Leak hits | Persistence |
|---|---:|---:|---:|---:|---|---:|---|
| ∞ | 33636.5 | 10437.0 | **3.22×** | 0.296 | yes | 0 | yes |
| 1.0μ | 54551.0 | 12396.5 | **4.40×** | 0.392 | yes | 0 | yes |
| 0.8μ | 78767.0 | 21030.0 | **3.75×** | 0.366 | yes | 0 | yes |

**Interpretation:**
- Model **can play** (integer orders every week after resample/fallback; not random gibberish collapse).
- **Not** competitive with base-stock (3–4× worse). Parse-fail rises with context length (~0% early → ~30–40% late) — grammar adherence degrades as history grows.
- Floor is **high enough that GRPO is not a priori hopeless**, but productization must cut parse-fail (stricter decoding / constrained decoding / shorter history) before training spend.

If parse-fail stayed ~100% with constant fallback, this check would be **NO**. That did not occur.

---

## Check 6 — Non-negotiables survive the port — **GO** (with flags)

| Constraint | Current scaffolding | LLM-tier requirement |
|---|---|---|
| One policy / LoRA per role (A/B); no shared weights | IPPO: `policies[r]` + independence assert | **Keep:** one LoRA adapter per role; never share across retailer_a/b/… |
| Strictly local rewards (A/B) | `core.py` A/B → `−local_cost` | GRPO scalar = **per-agent local cost** (not system) |
| Signals unverified / optional | Regime B channel exists; unverified | **Channel dropped** for LLM — do not call `SignalingActorCritic` / signal heads |
| Regime C system reward | Exists for reproduction | **Do not** use C as the selfish-emergence GRPO setting |

**Violation risks in current repo if ported naively:**
1. Reusing Regime **B** `build_env_config` enables `signaling_enabled=True` — conflicts with order-only design.
2. Training on Regime **C** would smuggle system reward into GRPO (invalidates emergence claim).
3. A single shared LoRA / shared backbone updated as one policy across roles would violate the non-negotiable (May 2026 paper’s setup — explicitly what we are *not* doing).

---

## Check 7 — Budget projection (arithmetic only, $0 spent) — **NO-GO as-planned**

Assumptions (order-only LLM cells after channel drop):
- Cells: Y × `{∞, 1.0μ, 0.8μ}` × proportional × 3 seeds = **9** (mechanism honesty-weighted optional → 18).
- T=52, 5 roles, GRPO group size G=4, **200 updates/cell**.
- Mean prompt tokens/role/week ≈ 1330 (full history) or ≈ 600 (rolling W=8).
- Completion ≈ 8 tok (`ORDER: N`).
- Blended 4090 throughput ≈ 400 tok/s; rate **$0.50/hr** (mid of spec $0.30–0.70).

| Strategy | Tokens (9×200×G=4) | GPU-h | Est. $ | vs $250 |
|---|---:|---:|---:|---|
| Full history | ~2.5×10⁹ | ~1740 | **~$870** | **Over** (~3.5×) |
| Rolling W=8 | ~1.1×10⁹ | ~790 | **~$395** | **Over** |
| Rolling W=8, 50 updates | ~2.9×10⁸ | ~200 | **~$100** | **Fits w/ margin** |
| Full history, 50 updates | ~6.3×10⁸ | ~430 | **~$217** | **Tight fit** |
| Spec-like 18 cells × 200 × full hist | ~5.0×10⁹ | ~3480 | **~$1740** | Far over |

**Drivers:** retained context is **~54%** of token volume vs rolling-8 at 200 updates (full hist vs W=8). Context retention strategy **dominates** the budget.

**Verdict:** Full-history GRPO at a serious update count **does not fit** the $250 hard cap. A **rolling window (or summary) + lean update schedule** is required before any paid run. Projection appended to `LEDGER.md` (not a spend).

---

## Deliverable checklist

- [x] SHA recorded  
- [x] GO/NO-GO per check  
- [x] Info-asymmetry table (Check 4)  
- [x] Token / budget numbers (Checks 3b, 7)  
- [x] Overall verdict  
- [x] `LEDGER.md` projection row  
- [x] `DECISIONS.md` verdict line  
- [ ] Merge — **awaiting human review**

**Smoke artifacts (supporting, $0):**
- `scripts/diag/llm_tier_smoke.py`
- `artifacts/diagnostics/llm_tier_smoke.json`
