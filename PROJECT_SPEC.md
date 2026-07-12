# Emergent Cooperation and Deception in Supply Chain Agents

**Working title:** *Do Self-Interested LLM Agents Rediscover the Bullwhip — and Learn to Game It?*

**Author:** Consti Ertel · **Status:** Spec v1.0 (July 2026) · **Target:** arXiv preprint + Prime Intellect Environments Hub + project website writeup

---

## 1. One-Paragraph Summary

We train independent, self-interested agents — classical MARL policies and small open-weight LLMs post-trained with GRPO — to play an extended Beer Distribution Game. Unlike prior work (notably the May 2026 GRPO beer game paper, arXiv:2605.17036, which trains a *shared* model on a *system-level* reward), each agent here has its own parameters and is rewarded **only on its own local costs**. We add (a) an optional, unenforced information-sharing channel and (b) a capacity-constrained factory with proportional rationing. The research questions: does honest information sharing *emerge* from self-interest when capacity is slack, and does *order inflation / shortage gaming* (Lee, Padmanabhan & Whang 1997's canonical bullwhip cause) emerge when capacity is tight? The headline artifact is a **phase diagram**: capacity tightness vs. learned signaling honesty. A mechanism-design second act tests whether honesty-weighted allocation rules restore truthful equilibria.

## 2. Positioning vs. Prior Work

| Work | Agents | Reward | Communication | Our delta |
|---|---|---|---|---|
| Oroojlooy et al. 2017 (DQN beer game) | Single RL agent, others fixed | Local + shaped | None | Multi-agent, all learning |
| Long et al. 2025 / infotheorylab demo | Prompted LLMs, no training | N/A | Info-sharing toggle (imposed) | We *train*; sharing is a learned action |
| arXiv:2605.17036 (GRPO beer game, May 2026) | Shared base LLM | **System-level** | None | **Separate policies, selfish rewards, optional cheap talk, capacity rationing, mechanism design** |

The May 2026 paper is our reproduction baseline (Regime C below). Our contribution lives in Regimes A/B and the capacity/mechanism experiments. No published work asks whether cooperation *emerges* (rather than being imposed via reward or orchestrator), or whether trained LLM agents rediscover shortage gaming.

## 3. Environment Specification

### 3.1 Core (faithful to classic beer game — validates against published results)
- 4 roles in a serial chain: Retailer → Wholesaler → Distributor → Factory.
- Discrete weeks, T = 52 (train), T = 36 (classic eval config for comparison).
- Per-role state: on-hand inventory, backlog, incoming shipments pipeline (2-week ship delay), incoming order (1-week order delay).
- Action: non-negative integer order quantity to upstream (cap at 64 for tractable action space; LLM agents emit an integer token sequence, parsed with a strict grammar + reject/resample on parse failure).
- Costs: holding h, backlog b per unit-week. Classic: h=0.5, b=1.0 uniform.
- Demand processes: (i) classic step 4→8, (ii) stationary uniform U[0,15], (iii) AR(1) with regime shift. Train on (ii), evaluate on all three (generalization check).
- Reward (per agent, per week): −(h_i · inventory_i + b_i · backlog_i). **Never** system cost, except Regime C.

### 3.2 Extensions (the contribution)
1. **Asymmetric costs:** downstream backlog penalties ≫ upstream (e.g., retailer b=2.0, factory b=0.5; holding inverted). Creates genuine incentive divergence.
2. **Signaling channel (cheap talk):** each week, each agent may broadcast a structured signal: `{claimed_demand: int | null, claimed_inventory: int | null}`. Broadcasting is optional, free, and **unverified** — agents may lie. All agents observe all broadcasts with 1-week delay. Honesty is *measured* (|claim − truth|), never rewarded directly.
3. **Capacity + rationing:** factory production capped at C units/week. When upstream orders exceed available supply at any node, allocation is **proportional to order size** (the classic rationing rule that incentivizes order inflation). C is the swept parameter: C ∈ {∞, 1.5·μ_demand, 1.2·μ, 1.0·μ, 0.8·μ}.
4. **(Stretch) Y-topology:** two competing retailers under one wholesaler. Only if core results land early.
5. **Mechanism-design act:** replace proportional rationing with (a) uniform allocation, (b) honesty-weighted allocation (allocation weight = exponential moving average of past signal accuracy). Question: does truthful signaling become the learned equilibrium?

### 3.3 Regimes (the experiment axes)
- **Regime A:** selfish rewards, no signaling channel (pure baseline).
- **Regime B:** selfish rewards, optional signaling (emergence test — the core contribution).
- **Regime C:** shared system reward (reproduction of arXiv:2605.17036 setup; sanity anchor).

Full matrix: {A, B, C} × {capacity levels} × {proportional, uniform, honesty-weighted rationing} × seeds. Not all cells run at the LLM tier — see §5.

## 4. Agents & Training

### Tier 1 — Classical MARL (statistical backbone, near-zero cost)
- Independent PPO (IPPO): one small policy per role (MLP, 2×256, or tiny GRU for the pipeline memory). **No parameter sharing, no shared critic** — parameter sharing smuggles in coordination and invalidates the emergence claim.
- Framework: PettingZoo ParallelEnv API + CleanRL-style IPPO (self-contained, auditable) or RLlib if multi-run orchestration is worth the dependency weight.
- Seeds: **≥10 per cell** across the full matrix. This tier produces the phase diagram.
- Baselines: base-stock policy (per-role, optimized via grid search), Sterman anchoring-and-adjustment heuristic, random. Report DQN-paper costs for the classic config as external anchor.

### Tier 2 — LLM agents (headline, targeted cells only)
- **Workhorse:** Qwen2.5-3B-Instruct, one LoRA adapter per role (4 adapters, one base model in memory — fits 24GB with 4-bit base). GRPO via TRL or verl; Unsloth if single-GPU memory is tight.
- **Capability-floor check:** report prompted-only (zero-training) performance of the same checkpoint first. If the base model can't play coherently zero-shot, results at that size are uninterpretable — move up a size.
- **Size ablation on headline cells:** repeat tight-capacity Regime B at Qwen2.5-7B (one A100 run) to defuse "your model was just too small."
- **Poolside Laguna XS 2.1 (33B-A3B, open weights):** include as **inference-only** prompted agent in all baseline evaluations (local vLLM or free OpenRouter tier). Rationale: coding-specialized model as economic agent is an interesting data point and a distribution/visibility opportunity. **Do not** make it the fine-tuning workhorse — architecture is <2 weeks old, GRPO tooling support unverified, 33B total params needs 80GB-class GPU for training. Promote to one GRPO run *only if* TRL/verl support is confirmed working, as a stretch goal.
- LLM-tier cells: Regime B × {C=∞, C=1.0μ, C=0.8μ} × {proportional, honesty-weighted}, 3 seeds each. Regime C at classic config, 3 seeds (reproduction). Everything else stays Tier 1.
- Prompting: role card (identity, costs, observation), few-shot format examples, structured output schema. Same template across models. Log every raw generation.

## 5. Metrics & Headline Artifacts

- **Bullwhip ratio:** Var(orders at echelon k) / Var(consumer demand), per echelon.
- **Costs:** per-agent and system, vs. base-stock/Sterman/random and vs. published numbers.
- **Sharing rate:** fraction of weeks each agent broadcasts.
- **Honesty score:** −mean |claimed − true|, normalized; plus a binary "inflation detector" (orders > 1.5× true need during rationing weeks).
- **Headline figure:** phase diagram — x: capacity tightness, y: honesty score, one curve per rationing mechanism, Tier-1 with CIs across ≥10 seeds, Tier-2 points overlaid.
- **Qualitative exhibit:** one free-form natural-language message experiment (LLM tier, single cell) — transcripts of agents hedging/misrepresenting under scarcity. Screenshot material, explicitly non-quantitative.

**Falsifiable predictions (register before running):** (P1) Under slack capacity, Regime B agents learn to share and honesty is high; Regime B system cost approaches Regime C. (P2) Under tight capacity + proportional rationing, order inflation emerges and honesty collapses. (P3) Honesty-weighted allocation restores truthful signaling. If P2 fails, that is *also* publishable ("trained LLM agents do not game rationing where humans do") — write it either way.

## 6. Engineering Plan

```
repo/
  env/            # pure-Python beer game core, no framework deps, 100% unit-tested
    core.py       # state transition, costs, delays  (property-based tests: conservation of goods)
    rationing.py  # allocation mechanisms
    signals.py    # cheap-talk channel
    wrappers.py   # PettingZoo ParallelEnv; Gymnasium single-agent wrapper for debugging
  agents/
    baselines.py  # base-stock, Sterman, random
    ippo/         # Tier 1 training
    llm/          # prompt templates, output grammar/parser, GRPO configs (TRL + verl)
  experiments/    # one YAML per matrix cell; every run: config + git SHA + seed logged
  analysis/       # phase diagram, bullwhip plots, honesty scoring
  paper/
```

- Env must be dependency-free and fast (>10k steps/sec) — it is also the Prime Intellect Environments Hub deliverable.
- **Validation gate (hard):** before any training, the env must reproduce (i) Sterman-heuristic bullwhip amplification qualitatively and (ii) base-stock costs matching the DQN paper's reported baseline within tolerance on the classic config. No training until this gate passes.
- Determinism: seeded RNG everywhere; golden-trajectory regression tests.
- Compute: Tier 1 on laptop/Colab. Tier 2 on Runpod/Vast spot — 4090 ($0.30–0.70/hr) for 3B runs, one A100-80GB session for the 7B ablation. **Budget cap: $250.** Track spend in a ledger file.
- LLM rollout efficiency: batch all 4 roles' generations per week; vLLM for rollout, TRL for updates (or verl's colocated mode).

## 7. Milestones (nights-and-weekends pacing)

| # | Deliverable | Gate |
|---|---|---|
| M1 (wk 1–2) | Env core + tests + baselines | Validation gate passes |
| M2 (wk 3–4) | IPPO Regime A/C on classic config | Matches published cost ballpark |
| M3 (wk 5–6) | Full Tier-1 matrix, phase diagram v1 | P1/P2 visible or refuted in MARL |
| M4 (wk 7–8) | LLM prompted baselines (Qwen + Laguna XS 2.1) | Capability floor documented |
| M5 (wk 9–11) | GRPO runs on headline cells | Tier-2 points on phase diagram |
| M6 (wk 12) | Mechanism-design cells + writeup + Hub publication | Preprint draft |

Kill criteria: if M2 slips past week 6, cut the Y-topology and free-form-language extras and ship the core. A finished modest result beats an abandoned ambitious one.

## 8. Risks

1. **MARL seed noise** — mitigated by Tier 1 carrying the statistics (≥10 seeds) and Tier 2 carrying only the demonstration.
2. **Small-LLM capability floor** — mitigated by prompted-only checks + 7B ablation.
3. **Emergence is boring (sharing trivially dominates)** — mitigated by asymmetric costs + capacity rationing, which manufacture genuine tension; and by pre-registered predictions making a null result publishable.
4. **Laguna tooling immaturity** — contained by inference-only default.
5. **Scope creep** — Y-topology and role-randomization are explicitly out of scope for v1.

## 9. Publication & Distribution

- Environment → Prime Intellect Environments Hub (standalone value).
- Preprint → arXiv (cs.MA / cs.LG), positioned against arXiv:2605.17036.
- Writeup → constiertel.com; framing: agents rediscovering a 1997 human behavior under 2026 training methods ("same river, only faster" — but let the site's frame do that work; the paper itself stays understated and empirical).
- X/LinkedIn thread anchored on the phase diagram + one deception transcript; tag poolside if Laguna results are included.
