# GPU / compute ledger

Hard budget cap: **$250**. Append a row before every paid GPU job.

| Date | Job | Hardware | Est. hours | Rate ($/hr) | Est. cost ($) | Actual ($) | Notes |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | Tier 1 is laptop-only; no GPU spend yet |
| 2026-07-15 | exp/bprime-control B×B′ matrix | — (not started) | 0 | 0 | 0 | **0** | **Gate block:** Prompt 2 ablation ⇒ channel not load-bearing; B′ train skipped. Planned slice was {B,B′}×{serial,y}×{∞,1.0μ,0.8μ}×prop×AR1×10seeds @400k. See `artifacts/diagnostics/v11_bprime.md`. |
| 2026-07-15 | **projection only** — LLM GRPO rolling-W8 (order-only, 9 cells × 50 upd × G=4) | 4090 @ $0.50/hr | ~177 (corrected) | 0.50 | **~$89** | **0** | **Not a spend.** Branch `feat/llm-rolling-context` SHA `4debc15…`. tokens/week≈538 (measured W=8 steady) × T=52 × 5 roles × G=4 × 50 upd × 9 cells. Resampling factor **1/(1−p)=1.00** from `llm_text_io.md` final p=0. Naive audit ~$100 @ 600 tok × 1.00 ≈ $99; corrected measured **~$89**. Fits $250 w/ ~$161 margin. See `artifacts/diagnostics/llm_rolling_context.md`. |
