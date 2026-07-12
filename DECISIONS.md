# Design decisions

Logged defaults for the v1 research codebase. Change only with a dated note.

| Decision | Default | Rationale |
|---|---|---|
| Python | 3.11+, package `beer_distribution_rl` | Spec / modern typing |
| Layout | Code under `beer_distribution_rl/{env,agents}/` | Enables `import beer_distribution_rl.env` |
| Classic eval horizon | T=36 | DQN-paper compare (Oroojlooy et al.) |
| Train default horizon | T=52 | PROJECT_SPEC §3.1 |
| Lead times | ship delay L_s=2, order delay L_o=1 | PROJECT_SPEC §3.1 |
| Action | absolute non-neg int order ∈ {0…64} | Tractable discrete action space |
| Classic costs | h=0.5, b=1.0 uniform all roles | Classic beer-game / DQN config |
| Classic demand | step 4→8 at week 5 (weeks 1–4 demand=4, then 8) | Classic MIT step |
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
