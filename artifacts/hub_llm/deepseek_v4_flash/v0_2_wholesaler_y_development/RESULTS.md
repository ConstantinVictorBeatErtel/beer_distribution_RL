# DeepSeek V4 Flash — Y Wholesaler Development Evaluation

Environment v0.2.0, `t5-strategic-y-v2`, wholesaler role, three public
development seeds, one rollout per seed, temperature 0, sequential execution.
These are development results, not a held-out benchmark claim.

| Seed | Model cost | Paired base-stock cost | Reward | Fill rate | Bullwhip |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,245.5 | 1,015.5 | 0.449 | 0.485 | 4.251 |
| 1 | 866.0 | 475.0 | 0.354 | 0.605 | 7.938 |
| 2 | 1,224.0 | 1,061.5 | 0.464 | 0.336 | 10.433 |
| **Mean ± SD** | **1,111.8 ± 213.2** | **850.7 ± 326.1** | **0.423 ± 0.060** | **0.475 ± 0.135** | **7.541 ± 3.110** |

All 108 actions used the required tool correctly and all episodes completed. The
model lost to its same-seed adaptive base-stock reference on every seed. This is a
valid negative result: the wholesaler task is learnable by a cheap heuristic but
not solved by this prompted model evaluation.

The three recorded traces used 113,734 input and 4,752 output tokens. Across all
five successful v0.2 smoke/development rollouts run during this correction pass,
recorded usage was 187,974 input and 7,920 output tokens, approximately $0.029 at
the observed Akash prices. Provider billing may include failed or retried calls.

Temperature zero did not make provider generation byte-identical: a separate seed
0 smoke scored 0.451 rather than 0.449. Environment replay remains deterministic
because every validated action and transition is recorded; model generation
repeatability is a separate property.
