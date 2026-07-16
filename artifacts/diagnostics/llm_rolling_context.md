# LLM rolling-context memory

**Branch:** `feat/llm-rolling-context`  
**Branched from (`main` after `feat/llm-text-io` merge):** `4debc15dfb5f7ac20adbb0e54d0c0442d4f54699` (`git rev-parse HEAD` at branch creation)  
**Date:** 2026-07-15  
**Spend:** **$0**. **No GRPO.** Inference-only / arithmetic only (local env rollouts + token estimates).

Clears readiness **blocker 3 (context)** and **blocker 7 (budget)** via productized rolling-window own-history.

---

## Product module

| Piece | Location | Role |
|---|---|---|
| Default `W` | `DEFAULT_ROLLING_WINDOW = 8` in `agents/llm/memory.py` | One-line ablation knob |
| Memory | `AgentMemory.window` + `windowed_history()` | Retains full episode; prompts see last W |
| Serializer | `serialize_prompt(..., window=None)` | Emits rolling own history only |

Each week's prompt includes the last **W** weeks of the agent's **own** structured history (orders, demand/incoming observed, allocations, inventory, backlog, pipelines, local cost). Never other agents' private state (E1).

**Ablation-ready:** change `DEFAULT_ROLLING_WINDOW` (or construct `AgentMemory(..., window=W)` / pass `window=` to `serialize_prompt`) for a later W-sensitivity check — mirrors the recurrent-baseline “does more memory help?” question.

---

## Per-prompt token count at W=8

Measured on Y × Regime A × prop × AR(1) × seed 0, role `retailer_a`, T=52, base-stock S=30 fill-in (no LLM calls). Estimate = `len(prompt)//4` (Check 3 / readiness parity).

| Quantity | Value |
|---|---:|
| Prompt @ week 0 (empty history) | **236** tok |
| Prompt @ steady W=8 (mean) | **538** tok |
| Prompt @ steady W=8 (max) | **540** tok |
| Mean over all 52 weeks | **510** tok |
| Qwen2.5-3B context | 32 768 |
| Fits with margin? | **Yes** (max 540 ≪ 32k; <2% of context) |

Prior Check-3 full-history max was ~2.4k tok/week; rolling W=8 caps prompts near ~0.5k.

---

## Persistence test (T=52, W=8)

Protocol: for every week `t ∈ [1, 51]`, the week-`t` prompt must contain week `t−1`'s structured outcome (`week=…`, `ordered=…`, `demand_or_incoming=…`, `alloc_recv=…`, `backlog=…`). After the window fills, weeks older than W must be absent.

| Check | Result |
|---|---|
| Persistence across full T=52 | **PASS** (`persistence_misses=[]`) |
| Window drop (week 0 absent at end) | **PASS** |
| E1 leak hits | **0** |

Unit coverage: `tests/test_llm_rolling_context.py`. Runner: `scripts/diag/llm_rolling_context.py` → `artifacts/diagnostics/llm_rolling_context.json`.

Example prompt tail at steady W=8:

```
Own history (rolling last W=8 weeks):
  week=N: demand_or_incoming=…, ship_in=…, ordered=…, alloc_recv=…,
          inv=…, backlog=…, on_order=…, ship_pipeline=…, order_pipeline=…, cost=…
  …
```

---

## Corrected GRPO budget (blocker 7) — projection only, $0 spent

Plan (same as readiness Check 7 lean schedule):  
**9 cells** (Y × {∞, 1.0μ, 0.8μ} × prop × 3 seeds) × **T=52** × **5 roles** × **G=4** rollouts/update × **50 updates** × 4090 @ **400 tok/s**, **$0.50/hr**.

Resampling factor from final parse-fail in `artifacts/diagnostics/llm_text_io.md`:

\[
\frac{1}{1-p},\quad p = 0\ \text{(0/780 pooled, schema-constrained)} \implies \mathbf{1.00}
\]

| Line | tokens/week | Tokens (pre-resample) | × 1/(1−p) | GPU-h | Est. $ | vs $250 |
|---|---:|---:|---:|---:|---:|---|
| Naive audit (Check 7) | 600 | 2.85×10⁸ | **×1.00** | 198 | **~$99** | Fits |
| Measured mean (all weeks) | 510 | 2.43×10⁸ | **×1.00** | 168 | **~$84** | Fits |
| Measured steady W=8 mean | 538 | 2.56×10⁸ | **×1.00** | 177 | **~$89** | Fits w/ margin |

**Corrected figure (report):** **~$89** (measured W=8 steady mean × resampling factor **1.00**).  
The readiness “naive ~$100” used a 600-tok estimate; productized prompts are slightly leaner (~538). Because constrained decoding drove **p→0**, resampling does **not** inflate spend (factor **1.00** exactly). For reference, the old Check-5 parse-fail (~35%, factor ~1.54×) would have pushed ~$99 → ~$152 — that tax is cleared by `feat/llm-text-io`.

**Fits $250 with margin:** yes (~$161 headroom at the corrected ~$89 line).

---

## Artifacts

- JSON: `artifacts/diagnostics/llm_rolling_context.json`
- Tests: `tests/test_llm_rolling_context.py`
- Runner: `scripts/diag/llm_rolling_context.py`
- Ledger projection row appended (not a spend)
