# Current state

**Date:** 2026-07-17 · **Spend so far:** $0 of $250 budget

One-paragraph version: the classical-RL half of this project (Tier 1) is
finished and produced a real result — trained agents rediscover shortage
gaming under tight capacity, a 2026 replication of a 1997 human behavior.
The LLM half (Tier 2) just ran its first free test, and the test **failed
the way the project plan says to watch for**: Qwen2.5-3B does not yet play
the game coherently enough to be worth training. Nothing paid has been
spent. The next step is a diagnosis-and-retry loop on the free tier, not a
GPU purchase.

---

## What we built

1. **The game engine** (`beer_distribution_rl/env/`) — a fast, tested
   simulator of the Beer Distribution Game, extended with a capacity-limited
   factory, order-inflating rationing rules, an optional (lie-able) cheap-talk
   channel, and a two-retailer "Y" topology. Validated against published
   results before any training.
2. **Classical RL training (Tier 1)** — independent PPO, one policy per role,
   run across an 840-cell matrix of regimes × capacity × rationing × seeds,
   plus a memory-equipped (GRU) variant. All CPU/Colab, effectively free.
3. **LLM plumbing** — code that turns a game observation into a text prompt,
   forces the model's reply into a strict, parseable schema, and keeps a
   rolling memory of the agent's own past weeks.
4. **NEW this session — `scripts/run_llm_episode.py` + `notebooks/colab_llm_smoke.ipynb`**
   — the piece that was missing: code that actually sits a model down and
   makes it play a full 52-week game, end to end, then reports how it did.
   This is the free "capability floor" check the project plan requires
   *before* any paid training run. Runs on Colab's free GPU tier.

## What works

- Environment: solid, validated, fast, deterministic.
- Tier-1 training pipeline: works end to end, matrix runner + Colab notebook.
- **Headline Tier-1 finding holds up:** on the Y-topology, when capacity gets
  tight and rationing is proportional, trained agents inflate their orders to
  grab a bigger share — the classic 1997 bullwhip behavior, reproduced in a
  2026 RL agent. This survived a later audit that caught and fixed an
  evaluation bug (some early regimes were being scored in different modes).
- LLM text plumbing: 0% parse failures — the model's replies always come back
  in a format the code can read.
- The new episode runner: verified against both a no-model baseline and a
  fake model server before ever touching Colab, so the game-logic side is
  trustworthy.

## What doesn't work

- **Cheap-talk channel is a dud.** Agents that could broadcast claims just
  produced noise; removing the channel changes nothing. The project pivoted
  away from "does honest signaling emerge?" toward the order stream itself,
  which is where the real finding above lives.
- **Honesty-weighted rationing didn't restore truthfulness.** Agents stopped
  broadcasting instead of learning to tell the truth.
- **New finding today — Qwen2.5-3B fails the capability floor.** First
  Colab run, 52-week episodes, 0% parse failures (the model always replies
  in valid format), but it collapses to near-zero orders instead of playing
  sensibly:

  | Setting | Qwen2.5-3B mean order/role | Cost vs. naive "order what you sold" baseline |
  |---|---|---|
  | Classic serial chain | 0.38, 0.08, 0.0, 0.0 | **3.5× worse** |
  | Y-topology, tight capacity | 1.88, 1.77, 0.87, 0.31, 0.02 | ~same cost, but via backlog spiraling instead of real ordering |

  In the classic run the model orders almost nothing from week 1 onward while
  backlog and cost climb every week — it isn't reasoning about inventory, it's
  defaulting toward "order 0." This is a **valid, useful negative result**,
  not a bug: the project's own plan calls this out by name ("if the base
  model can't play coherently zero-shot, results at that size are
  uninterpretable — move up a size") and says to stop before spending money
  if this happens. So we stopped, as designed. $0 spent.

## What's next

1. **Don't train yet.** Training a model that can't play the game coherently
   would waste the budget and produce uninterpretable results.
2. **Retry cheaply, still on the free tier, in this order:**
   - Try `qwen2.5:7b` (bigger, still free on Colab, still $0) — the project
     plan's own prescribed first move when a capability floor fails.
   - Try prompt tweaks (a worked example in the prompt, a stronger nudge
     against ordering zero) before assuming size is the only lever.
   - Try Gemma 4 (an "American open model" alternative) at a comparable size,
     for comparison.
3. **Re-run the same free capability-floor notebook** against whichever
   change is tried, and look for the same two things: parse-fail rate near 0
   (already true) and orders that track demand instead of collapsing to 0.
4. **Only after a model clears the floor** does the paid step happen: a
   ~$30, 3-seed GRPO run on the Y-topology/tight-capacity cell, fully specced
   already in `artifacts/diagnostics/llm_tier_readiness_v2.md`, with kill
   criteria written down in advance.
5. Ledger discipline continues: `LEDGER.md` gets a row before any paid GPU
   time, no exceptions.

## Where things live

- Research plan: `PROJECT_SPEC.md`
- Every locked-in default and why: `DECISIONS.md`
- Spend tracking: `LEDGER.md`
- Free capability-floor test: `notebooks/colab_llm_smoke.ipynb` (Colab, $0)
- Paid-run spec (not yet launched): `artifacts/diagnostics/llm_tier_readiness_v2.md`
