# Y-Network Wholesaler Variant

The Tier 5 wholesaler evaluation is a complementary view of the same Beer
Distribution Game, not a separate environment. It uses the same simulator,
scenario seeds, scripted counterparties, grader, reward, and strict
`place_order(quantity)` interface as the retailer evaluation. The taskset merely
selects `controlled_roles = ["wholesaler"]`.

## Why it is harder

The retailer sees exogenous customer demand and can often learn a stable ordering
rule directly. The wholesaler instead serves two retailers whose orders are
endogenous. One retailer can order aggressively, both react to shortages, and the
factory has finite capacity. The wholesaler must therefore manage delayed supply
while separating genuine demand changes from strategic or amplified downstream
orders. This more directly tests bullwhip control and coordination under scarcity.

The variant should remain complementary rather than replace the retailer task.
Retailer and wholesaler results measure different information positions, so they
should be reported in separate rows rather than averaged into one headline score.

## Smoke evaluation

The checked-in `eval_akash_wholesaler_y_smoke.toml` selects one Tier 5 development
seed, one wholesaler task, and concurrency one. It spends no credits until the
evaluator is explicitly run with an `AKASH_API_KEY` in the local environment.

For broader runs, increase `seed_limit` only after the one-seed trace is protocol
clean. Keep `tiers = [5]` because the purpose of this variant is specifically the
two-retailer Y topology.

The wholesaler task is versioned with the canonical scenario. Current results must
use environment v0.2.0 / `t5-strategic-y-v2`; earlier v1 traces used the superseded
base-stock timing convention and are diagnostic only.
