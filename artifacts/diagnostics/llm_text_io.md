# LLM text I/O (serializer + grammar-constrained parser)

**Branch:** `feat/llm-text-io`  
**Branched from (merge of `baseline/recurrent-ippo` into `main`):** `5c0f6852daab759c393dfa8b3b76961fae025a23` (`git rev-parse HEAD` at branch creation)  
**Date:** 2026-07-15  
**Spend:** **$0** (local Ollama `qwen2.5:3b` only). **No GRPO.**

Clears readiness **blocker 2 (text I/O)** and drives Check 5's 30–39% parse-failure rate to **0%**, uniform across capacity × role.

---

## Product module

`beer_distribution_rl/agents/llm/`

| Half | Module | Role |
|---|---|---|
| Serializer | `serializer.py`, `memory.py` | Own obs (+ retained own history) → prompt |
| Parser / decode | `grammar.py`, `parser.py`, `decode.py` | Constrained JSON Δ → absolute order |

Shared information set matches recurrent IPPO / Check 3 (E1 no-leak): own past orders, demand/incoming observed, allocations received, inventory, backlog, pipelines. Never other agents' private state; never privileged `customer_demand` / `true_demand` fields.

Action space (IPPO parity): relative **Δ ∈ [-8, 8]** → `order = clip(last_demand_or_order + Δ, 0, 128)`.

---

## Constrained-decoding approach

**Not** post-hoc regex as the primary path.

1. **JSON-schema constrained sampling** via Ollama chat `format=` (same class of guarantee as vLLM guided decoding / logits masking):
   ```json
   {"type":"object","properties":{"delta":{"type":"integer","minimum":-8,"maximum":8}},
    "required":["delta"],"additionalProperties":false}
   ```
2. Portable **GBNF** twin in `grammar.py` (`ORDER_DELTA_GBNF`) for vLLM / llama.cpp backends.
3. Map Δ → absolute order with `map_delta_to_order`.
4. **Resample** up to 3 times on parse failure; log per-attempt fail rate; demand-match fallback if exhausted.

Check 5 used free-form `ORDER: <int>` + regex → 30–39% fail as history grew. Schema constraint makes invalid tokens unreachable; failures become rare by construction.

**Grammar iteration:** none required after the first schema — acceptance bar met at 0% on the first constrained re-measure. Kept resample+logging as safety.

---

## Serializer round-trip (3 real logged observations)

Y-topology, Regime A, AR(1), seed 0, role `retailer_a`. Synthetic constrained output `{"delta": 1}` parsed back to absolute order. Leak report empty on all three.

### Example 1 — week 0

```
inventory=12 backlog=0 on_order=12 inventory_position=24
last_demand_or_order=0 last_shipment_received=0 last_order_placed=0
ship_pipeline=[4, 4] order_pipeline=[0]
Own history: (none — first week)
raw → {"delta": 1}  ⇒  order=1
```

### Example 2 — week 1

```
inventory=9 backlog=0 on_order=9 inventory_position=18
last_demand_or_order=7 last_shipment_received=4 last_order_placed=1
ship_pipeline=[4, 4] order_pipeline=[0]
Own history:
  week=0: demand_or_incoming=0, ship_in=0, ordered=1, alloc_recv=4,
          inv=9, backlog=0, on_order=9, ship_pipeline=[4, 4],
          order_pipeline=[0], cost=4.50
raw → {"delta": 1}  ⇒  order=8
```

### Example 3 — week 2

```
inventory=5 backlog=0 on_order=13 inventory_position=18
last_demand_or_order=8 last_shipment_received=4 last_order_placed=8
ship_pipeline=[4, 1] order_pipeline=[0]
Own history includes week=0 and week=1 ordered=… lines (persistence)
raw → {"delta": 1}  ⇒  order=9
```

Unit coverage: `tests/test_llm_text_io.py` (round-trip, E1 upstream no-leak, Δ mapping, schema/GBNF).

---

## Parse-failure re-measure (Check 5 protocol)

Same setup as Check 5: local `qwen2.5:3b`, Y × Regime A × proportional × AR(1), T=52, seed 0, caps `{∞, 1.0μ, 0.8μ}`, all 5 roles prompted.

**Before** = Check 5 post-hoc `ORDER:` regex (`artifacts/diagnostics/llm_tier_smoke.json` on `preflight/llm-tier-readiness`). Role stratification was not logged there.  
**After** = this branch, JSON-schema constrained Δ decoding (`artifacts/diagnostics/llm_text_io_smoke.json`).

### By capacity (overall)

| Cap | Before (regex) | After (schema) | After attempts | After fails |
|---|---:|---:|---:|---:|
| ∞ | **29.6%** (96/324) | **0.0%** | 260 | 0 |
| 1.0μ | **39.2%** (138/352) | **0.0%** | 260 | 0 |
| 0.8μ | **36.6%** (126/344) | **0.0%** | 260 | 0 |

### By capacity × role (after; before role N/A in Check 5)

| Cap \\ Role | retailer_a | retailer_b | wholesaler | distributor | factory |
|---|---:|---:|---:|---:|---:|
| ∞ | 0% (0/52) | 0% | 0% | 0% | 0% |
| 1.0μ | 0% | 0% | 0% | 0% | 0% |
| 0.8μ | 0% | 0% | 0% | 0% | 0% |

**Uniformity:** max−min rate across all 15 cells = **0**. No scarcity concentration → no selection bias into shortage-gaming via resampling.

**Acceptance bar:** parse-fail **≪ 5%** and uniform across capacity and role — **PASS**.

Leak hits: 0. Persistence: ok on all three episodes.

---

## Resampling cost multiplier

\[
\frac{1}{1-p},\quad p=\text{parse-failure rate}
\]

| Cap | Before \(p\) | Before \(1/(1-p)\) | After \(p\) | After \(1/(1-p)\) |
|---|---:|---:|---:|---:|
| ∞ | 0.296 | **1.42×** | 0.0 | **1.00×** |
| 1.0μ | 0.392 | **1.64×** | 0.0 | **1.00×** |
| 0.8μ | 0.366 | **1.58×** | 0.0 | **1.00×** |
| Pooled | ~0.35 | ~1.54× | **0.0** (0/780) | **1.00×** |

**Budget feed-forward:** use resampling multiplier **1.00** for the next GRPO budget correction (no parse-tax). Prior Check-7 plans that baked in ~1.5× parse overhead can drop that factor.

---

## Side note (not a gate)

Prompted system cost vs base-stock S=30 under constrained decode: ∞ 1.92×, 1.0μ 1.61×, 0.8μ 0.95× (smoke only; policy quality is out of scope for this branch).

---

## Artifacts

- Smoke JSON: `artifacts/diagnostics/llm_text_io_smoke.json`
- Smoke log: `artifacts/diagnostics/llm_text_io_smoke.log`
- Tests: `tests/test_llm_text_io.py`
- Runner: `scripts/diag/llm_text_io_smoke.py`
