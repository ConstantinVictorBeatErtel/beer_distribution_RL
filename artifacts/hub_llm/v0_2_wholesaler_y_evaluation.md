# v0.2 Tier-5 Y-wholesaler evaluation

All model evaluations used the same `0.2.0` environment, wholesaler role,
temperature-zero request setting, sequential execution, and no test seeds.
The development split uses seeds 0–2; validation uses seeds 0–4. Values are
mean ± population standard deviation across episodes. Lower local cost is
better; the base-stock reward is exactly 0.500 by construction.

## Headline comparison

| Split | Policy/model | Local cost | Reward |
|---|---|---:|---:|
| Development | Uniform random | 1,927.5 ± 71.9 | 0.299 ± 0.085 |
| Development | Adaptive base-stock | 850.7 ± 326.1 | 0.500 ± 0.000 |
| Development | DeepSeek V4 Flash | 1,111.8 ± 213.2 | 0.423 ± 0.060 |
| Development | Qwen3.6-35B-A3B | 1,054.8 ± 343.5 | 0.442 ± 0.022 |
| Validation | Uniform random | 2,338.3 ± 270.1 | 0.290 ± 0.099 |
| Validation | Adaptive base-stock | 1,013.6 ± 432.4 | 0.500 ± 0.000 |
| Validation | DeepSeek V4 Flash | 1,418.9 ± 176.6 | 0.396 ± 0.108 |
| Validation | Qwen3.6-35B-A3B | 1,386.5 ± 348.0 | 0.403 ± 0.077 |

Neither hosted model beat the same-seed adaptive base-stock reference on the
validation headline mean. This is a valid negative result, not a reason to
tune on validation seeds.

## Validation controls

| Variant | Policy/model | Local cost |
|---|---|---:|
| Base-rival | Uniform random | 2,877.6 ± 715.0 |
| Base-rival | Adaptive base-stock | 1,123.8 ± 332.3 |
| Base-rival | DeepSeek V4 Flash | 1,983.2 ± 743.2 |
| Base-rival | Qwen3.6-35B-A3B | 1,738.0 ± 514.0 |
| Uniform-rationing | Uniform random | 2,338.3 ± 270.1 |
| Uniform-rationing | Adaptive base-stock | 1,013.6 ± 432.4 |
| Uniform-rationing | DeepSeek V4 Flash | 1,606.0 ± 567.4 |
| Uniform-rationing | Qwen3.6-35B-A3B | 1,511.0 ± 324.2 |

## Provider/model variance

Qwen3.6 was repeated five times on fixed development seed 0. Costs were
`1,274, 1,132, 1,237, 1,287, 1,528`, giving `1,291.6 ± 145.5`; reward was
`0.442 ± 0.027`. The paired base-stock cost was constant at `1,015.5`.
All 180 Qwen validation actions and all 108 DeepSeek validation actions were
protocol-clean and all episodes completed.

The first Qwen probe with `max_tokens=96` failed to reach the tool because the
model spent its completion budget on reasoning. It is retained as a separate
capability/configuration failure, not pooled with the valid evaluation. The
valid Qwen run used `max_tokens=512` and `reasoning_effort="none"`.

## Training gate (specification only; not launched)

The old 3B proposal is superseded: the repository records that Qwen2.5-3B did
not clear the zero-shot capability floor. If training is later approved, the
smallest defensible probe is:

| Parameter | Proposed value |
|---|---|
| Model | Qwen2.5-7B-Instruct, 4-bit QLoRA |
| Role/task | Tier-5 Y wholesaler only; headline capacity 22 |
| Reward | Controlled wholesaler local-cost reward only |
| Adapter | One wholesaler adapter; scripted counterparties; no shared adapter |
| Rollout group | G=4 |
| Updates | 50 maximum; inspect at 10 and 20 |
| Learning rate | 5e-6 |
| Context | Own-history window W=8; current environment horizon T=36 |
| Hardware estimate | Colab Pro L4/T4, NF4, checkpointing, 8-bit optimizer; target 14–16 GB |
| Runtime/budget | Roughly 6–10 hours; confirm actual Colab quota/cost before launch |

Kill the run for OOM/restarts, protocol failure above 5%, persistent cap/zero
ordering collapse, or no improvement in local cost/order diagnostics by update
20. Any shared adapter, system-cost reward, or information/signaling change
invalidates the experiment. This cell remains unlaunched pending explicit
approval.
