# v11 B′ control — **NOT RUN** (ablation gate blocked)

**Causal matrix skipped.** Prompt instruction: do not train B′ until the Prompt 2 ablation shows the channel is *plausibly load-bearing*. It does not.

## Gate evidence (Prompt 2 — eval-only ablation)

Frozen Tier-1 v11 B checkpoints; deterministic (argmax) eval; listener signal board ablated at inference (zero / shuffle-agent / shuffle-time). Rubric: channel is load-bearing only if **B collapses toward A under shuffle** (esp. shuffle-across-time).

| Topology | Cap | A | B intact | B shuffle-time | Δ(shuffle−intact) | Rubric call |
|---|---|---:|---:|---:|---:|---|
| serial | 1.0μ | 623±49 | 555±131 | 574±139 | +19 | **artifact** — shuffles inert; no reliable B edge |
| serial | 0.8μ | 535±115 | 475±117 | 490±141 | +15 | **artifact** — shuffles inert |
| y | 1.0μ | 1443±103 | 1421±93 | 1444±101 | +23 | **artifact** — shuffles inert |
| y | 0.8μ | 1345±48 | 1355±88 | 1383±100 | +28 | **artifact** — shuffles inert |

Plain-language Prompt 2 verdict: **channel is not load-bearing.** Matched deterministic eval also shows the logged stochastic A−B scarcity gap is mostly an eval-mode confound (A greedy vs B stochastic), not information flow.

Corroboration from Prompt 3 (`analysis/v11-signal-content`): retailer broadcasts are **babbling** — I(claim;d₊₁)≈0.07 bits (~14% of truthful lag-1 ceiling) and held-out decoder R²≈0.07. No lied-but-decodable code to scramble.

## Planned matrix (not executed)

`{B, B′} × {serial, y} × {∞, 1.0μ, 0.8μ} × proportional × AR(1) × 10 seeds`, 400k steps, R1 runner conventions. B′ = identical architecture to B except listener signal inputs permanently scramble-across-time **during training**.

| Cell count if run | ~120 train cells |
|---|---|
| Wall / $ | see `LEDGER.md` — **$0 / 0 h** (gate block) |

## B-vs-B′ table

*Empty — no B′ checkpoints.* Interpreting B vs B′ without a load-bearing channel would not isolate “information flow”; it would re-measure architecture/exploration under a known babbling equilibrium.

## Figure

*Not generated.* No costs to plot.

## Causal verdict (counterfactual framing only)

| If we had run… | Interpretation |
|---|---|
| B < B′ under scarcity (CIs exclude 0) | Clean claim: information flow (not architecture) drives the advantage |
| B ≈ B′ | Headline dies; babbling / negative-result framing |

**Given the gate:** treat the second row as the **prior** already supported by Prompt 2+3. Do **not** spend compute to rediscover babbling via B′. Revisit B′ only if a future checkpoint fails the shuffle-time rubric (Prompt 2 criterion 2).

## M4 LLM gate

**Not cleared.** See `DECISIONS.md`. Tier-1 has not demonstrated a load-bearing cheap-talk channel; M4 prompted LLM baselines remain blocked on the communication claim.
