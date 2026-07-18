# Beer Distribution Agent Environment — Difficulty Ladder

**Status:** Stage 4 approved, 2026-07-18
**Depends on:** [`ENVIRONMENT_SPEC.md`](ENVIRONMENT_SPEC.md) and
[`REWARD_SPEC.md`](REWARD_SPEC.md), approved 2026-07-18
**Scope:** Five order-only tiers with matched capability contrasts

## 1. Design rule

Difficulty is defined by environment properties, not by whether a particular
model happens to score poorly. Each tier introduces one principal capability and
keeps earlier parameters fixed where possible. When a topology change makes a
single adjacent comparison impossible, the tier includes explicit mechanism
controls.

The ladder is not a claim that one scalar captures every kind of difficulty.
Results remain stratified by tier and role. A model can be strong at stochastic
control and weak at strategic scarcity, and the report should show that shape.

## 2. Parameters shared by all tiers

| Parameter | Value |
|---|---|
| Decision horizon | 36 operational weeks |
| Settlement | 3 weeks (`order_delay + shipment_delay`) plus the approved terminal exposure charge |
| Order delay | 1 week |
| Shipment delay | 2 weeks |
| Order bounds | Integer `[0, 128]` |
| Holding cost | 0.5 per ending-inventory unit per week, every role |
| Backlog cost | 1.0 per ending-backlog unit per week, every role |
| Initial inventory | 12 units per role |
| Initial shipment pipeline | `[4, 4]` per role |
| Initial downstream-order pipeline | `[4]` per claimant; grader-private |
| Initial backlog | 0 |
| Strategic channel | None; orders only |
| Objective | Minimize the controlled role's local total cost |
| Model-visible history | Current observation plus 8 prior own-role records |
| Counterparty default | `adaptive_base_stock_v2`, defined below |
| Rationing default | Proportional; irrelevant on the serial single-claimant topology |
| Action validation and grading | The approved Stage 2 and Stage 3 contracts |

Capacities are explicit integers or `null`. No tier derives capacity from samples
or from the episode seed.

### Default scripted counterparty

`adaptive_base_stock_v2` is deterministic and uses only the role's permitted local
observation. Let `D_t` be its newly arrived local demand/order and let
`L = order_delay + shipment_delay = 3`:

```text
forecast_0 = scenario's public pre-change mean for that role
forecast_t = 0.75 * forecast_(t-1) + 0.25 * D_t
target_t = ceil(L * forecast_t)
order_t = min(128, max(0, target_t - inventory_position_t))
```

Serial roles initialize `forecast_0` to the single customer-stream mean. In Y,
retailers initialize to 7.5 and wholesaler/distributor/factory initialize to the
total mean 15. The observation is emitted after current demand is fulfilled. A
regression test shows that an order placed after week 1 demand arrives at the
start of week 4, so the target covers exactly `L=3` decision intervals. The v1
formula added another review period and is superseded. The policy is not told a
hidden shock time or post-shock mean.

This policy and its coefficients are frozen before model evaluation. Calibration
may reject a scenario as degenerate, but may not tune the policy on test seeds or
against a frontier model's results.

## 3. Tier summary

| Tier | Name | Principal change | Capability isolated |
|---|---|---|---|
| 1 | Steady control | Constant demand, full shipment notices, serial chain | Protocol use and stable delayed replenishment |
| 2 | Stochastic forecasting | Replace constant demand with stationary AR(1) | Filtering persistent noise without overreaction |
| 3 | Hidden regime shift | Add a seeded, undisclosed mean change | Online change detection and recovery |
| 4 | Pipeline uncertainty | Hide exact inbound-shipment timing | Belief-state maintenance from bounded history |
| 5 | Strategic scarcity | Y topology, explicit capacity, proportional rationing, aggressive rival | Robust local control under a competing claimant and manipulable allocation |

## 4. Tier definitions

### Tier 1 — Steady control

```yaml
scenario_id: t1-steady-serial-v2
topology: serial
demand:
  process: constant_v1
  value: 8
capacity: null
observation_mode: shipment_notices
counterparties: adaptive_base_stock_v2
```

`constant_v1` returns exactly 8 for every operational week and consumes no RNG.
The prompt states that customer demand is constant at 8. Exact dispatched inbound
shipments and aggregate `on_order` are visible; delayed downstream orders that
have not arrived remain hidden.

This is not intended to be challenging forecasting. It tests whether an agent can
follow the tool protocol, reason about a three-week replenishment delay, and reach
a stable ordering pattern without creating inventory or backlog oscillation.
Classical bullwhip is undefined here, so normalized order volatility is the
stability metric.

### Tier 2 — Stochastic forecasting

```yaml
scenario_id: t2-ar1-serial-v2
topology: serial
demand:
  process: ar1_v1
  mu: 7.5
  phi: 0.7
  innovation_sigma: 2.0
  x0: 7.5
  output: max(0, round(latent_state))
capacity: null
observation_mode: shipment_notices
counterparties: adaptive_base_stock_v2
```

Only the demand process changes from Tier 1. The prompt discloses that demand is a
stationary persistent stochastic process with long-run mean 7.5; it does not
disclose the seed or future samples.

This tests whether the agent distinguishes a persistent movement from one-week
noise and avoids amplifying that noise through a delayed system. The AR coefficient
is high enough to make recent history useful but below a near-random-walk regime.

### Tier 3 — Hidden regime shift

```yaml
scenario_id: t3-shift-serial-v2
topology: serial
demand:
  process: shifted_ar1_v1
  mu_before: 7.5
  phi: 0.7
  innovation_sigma: 2.0
  x0: 7.5
  shift_week_choices: [15, 19, 23]
  mu_after_choices: [4.0, 12.0]
capacity: null
observation_mode: shipment_notices
counterparties: adaptive_base_stock_v2
```

The shock namespace uses the master seed's 16-character hexadecimal string:

```text
shock_digest = SHA256("beer-agent-v1|" + master_seed_hex + "|mechanism/shock")
shift_week = [15, 19, 23][shock_digest_byte_0 mod 3]
mu_after = [4.0, 12.0][shock_digest_byte_1 mod 2]
```

The selections are therefore independent of demand-stream consumption. The
realized values are stored in canonical scenario JSON but are grader-private. The
demand RNG stream is otherwise the same as Tier 2, so a matched Tier 2/Tier 3 seed
has identical innovations and identical pre-shift demand.

The prompt says that one persistent demand-regime change may occur, but does not
reveal its direction, time, or magnitude. This prevents a calendar-memorized
always-increase response and tests detection plus controlled adaptation. Upward
and downward changes are both required: a policy that only builds safety stock
cannot pass both cheaply.

### Tier 4 — Pipeline uncertainty

```yaml
scenario_id: t4-partial-shift-serial-v2
topology: serial
demand: identical_to_t3_for_same_seed
capacity: null
observation_mode: aggregate_supply_line
counterparties: adaptive_base_stock_v2
```

Tier 4 reuses the exact Tier 3 realized shock and demand trajectory for a given
seed. The only environment change is observation mode:

- remove `inbound_shipment_pipeline` and any arrival-by-week detail;
- retain aggregate `on_order`, current receipt, inventory, backlog, incoming local
  demand/order, last order, and the fixed 8-record own-role history;
- never reveal delayed downstream orders or another role's state.

The hidden environment trajectory must remain identical for a fixed joint action
sequence in Tier 3 and Tier 4. This paired contrast isolates reconstructing likely
pipeline timing and maintaining a belief state. It does not increase stochasticity,
change counterparties, or tighten capacity.

### Tier 5 — Strategic scarcity

```yaml
scenario_id: t5-strategic-y-v2
topology: y
demand:
  process: correlated_y_ar1_v1
  per_retailer_mu: 7.5
  phi: 0.7
  common_innovation_sigma: 2.0
  idiosyncratic_sigma: 1.5
  common0: 0.0
capacity: 22
observation_mode: aggregate_supply_line
rationing: proportional_largest_remainder_v1
counterparties:
  scripted_retailers: scarcity_aggressive_v1
  scripted_upstream_roles: adaptive_base_stock_v2
```

Capacity 22 is identical across seeds. It leaves headroom above the theoretical
total customer-demand mean of 15 while still creating intermittent scarcity once
delayed orders propagate upstream. The value was calibrated only on development
and validation seeds. After correcting the v2 reference timing, capacity 22 binds
in 55.6% of validation base-rival weeks, inside the predeclared 10%--70% gate;
aggressive retailers have a 0% order-cap hit rate.
The allocation rule and capacity are public. Rival state, demand, orders, and
exact policy parameters remain private; the prompt says only that a scripted
rival may order aggressively under scarcity.

`scarcity_aggressive_v1` is a deterministic stress policy, not a claim of optimal
adversarial play:

```text
order_t = min(128, adaptive_base_stock_v2_order_t + 8)
```

When a retailer is controlled by the model, the other retailer uses this policy.
When wholesaler, distributor, or factory is controlled, both scripted retailers
use it. All scripted non-retailer roles use `adaptive_base_stock_v2`.

The tier tests whether the agent remains stable when another claimant persistently
inflates orders and proportional rationing rewards claim size. Low private cost
with a large rival/system externality is reported as strategic behavior, not
misreported as cooperative success.

#### Required Tier 5 mechanism controls

Two paired controls use the same seeds, demand, observations, and capacity:

1. `t5-control-base-rival`: replace `scarcity_aggressive_v1` with
   `adaptive_base_stock_v2`; keep proportional rationing.
2. `t5-control-uniform`: retain the aggressive rival but replace proportional
   allocation with `uniform_round_robin_v1`, capped by each claimant's request.

The headline Tier 5 score uses the strategic scenario, not an average over these
controls. At minimum, all heuristic baselines and shortlisted frontier models run
both controls. A strategic-robustness claim requires a response to the aggressive
rival and a different response when uniform allocation removes the marginal
benefit of order inflation.

## 5. What is public to the agent

| Information | T1 | T2 | T3 | T4 | T5 |
|---|---:|---:|---:|---:|---:|
| Cost coefficients, delays, horizon, order bounds | Yes | Yes | Yes | Yes | Yes |
| Exact factory capacity and rationing rule | n/a | n/a | n/a | n/a | Yes |
| Demand family | Constant | Persistent stationary | One possible persistent shift | Same as T3 | Persistent stochastic per retailer |
| Long-run/pre-change mean | 8 | 7.5 | 7.5 | 7.5 | 7.5 own / 15 total capacity |
| Exact shock time/direction | n/a | n/a | No | No | n/a |
| Exact inbound shipment schedule | Yes | Yes | Yes | No | No |
| Aggregate `on_order` | Yes | Yes | Yes | Yes | Yes |
| Rival policy formula or state | n/a | n/a | n/a | n/a | No |

The master seed, RNG states, future samples, and grader-private scenario fields are
never included in the model prompt. They remain in the audit trace and are
published with final results.

## 6. Role coverage and compute-aware evaluation

Every tier can generate a task for every role in its topology. The environment
package must support all of them even when a cheap screening run uses a subset.

### Core frontier screen

Run every candidate frontier model on all five headline tiers using the
customer-facing controlled role:

- serial tiers: retailer;
- Y tier: retailer A.

This preserves the same position across the main Tier 1–4 contrasts and makes the
Tier 5 role a direct competing claimant. With 10 test seeds and 36 decisions, the
screen is 1,800 model decisions per model.

### Role-complete evaluation

Run random and adaptive base-stock baselines for every role and all official
seeds; simulator-only cost is negligible. Run at least two predeclared frontier
models across all roles on five test seeds:

```text
(4 serial tiers * 4 roles + 1 Y tier * 5 roles) * 5 seeds * 36 weeks
= 3,780 model decisions per model
```

The role-complete table uses test indices `00000`–`00004` for every role,
including the retailer; it does not mix the retailer's ten-seed core estimate into
a five-seed role macro-average. Tier 5 mechanism controls run for retailer A and
wholesaler on those five seeds for shortlisted frontier models, and for every role
for cheap baselines.

### Promotion without hiding failures

Each candidate first runs the three development seeds. A model with less than 95%
first-attempt protocol-clean decisions is stopped before paid test evaluation; its
capability-floor failure and costs remain published. No cost threshold is used:
failing to beat base-stock is a result, not a reason to suppress a model.

Among protocol-capable candidates, the role-complete extension predeclares at
least one hosted model and one reproducible open-weight model using validation
results and expected evaluation cost. If more models fit the fixed budget, all may
advance. The selection rule and all validation rows are published before opening
test results.

The 52-week horizon is a non-headline stress extension for the best-performing
model and both cheap baselines only. It is not silently mixed with 36-week scores.

## 7. Seed splits

Seed IDs are derived without Python's randomized `hash()`:

```text
master_seed_hex(split, index) =
  first_16_hex_chars(
    SHA256("beer-agent-v1|" + split + "|" + five_digit_index)
  )
```

The canonical JSON stores this as a 16-character lowercase hexadecimal string.
Simulator adapters convert it to an unsigned 64-bit integer internally. Keeping it
as a string avoids loss of precision in JSON/JavaScript runtimes.

The fixed split sizes are:

| Split | Seeds per headline cell | Use |
|---|---:|---|
| Development | 3 | Public implementation and prompt debugging |
| Validation | 5 | Scenario calibration and predeclared model selection |
| Test | 10 | Core final evaluation; hidden until benchmark v1 is frozen |

The role-complete frontier evaluation uses test indices `00000`–`00004`; the core
screen uses all ten. Baselines run all ten plus an optional 100-seed baseline-only
stability study. All final test manifests are published after evaluation.

Shock parameters use a distinct `mechanism/shock` child namespace. Demand,
counterparty, and shock selection never consume from one another's RNG streams.

## 8. Environment-property calibration gates

These gates are evaluated with simulator policies on development and validation
seeds before the v1 manifest is frozen. They do not depend on frontier-model
performance.

1. Every tier produces finite cost and JSON-safe metrics for every role.
2. `adaptive_base_stock_v2` has lower median local cost than random ordering in
   every headline `(tier, role)` cell. A failing cell is not informative enough to
   publish without explanation or redesign.
3. Tier 1 demand variance is exactly zero and Tier 2 demand variance is positive.
4. Matched Tier 2/Tier 3 demand is byte-identical before the realized shift.
5. Matched Tier 3/Tier 4 hidden transitions are byte-identical for fixed actions;
   only the serialized observation differs.
6. Tier 3 includes at least one upward and one downward shift in every split. The
   v1 derivation above has been checked to satisfy this; generated manifests test
   the invariant rather than hand-picking outcomes.
7. Under the Tier 5 base-rival control, capacity binds in 10%–70% of operational
   weeks across validation seeds. Below that range scarcity is inert; above it the
   task risks becoming structurally infeasible.
8. Proportional and uniform Tier 5 controls produce different claimant allocations
   in at least 10% of shortage weeks under the same scripted actions.
9. `scarcity_aggressive_v1` does not hit the order cap in more than 25% of its
   decisions. Otherwise the stressor has collapsed into a cap artifact.
10. No tier's paired base-stock reference has zero total local cost.

If a proposed numeric parameter fails a gate, it is changed only using
development/validation evidence, recorded in `DECISIONS.md`, and released under a
new scenario version before any test or frontier-model run.

## 9. Claims the ladder can and cannot support

The matched contrasts support the following narrow claims:

- T1→T2: effect of persistent stochastic demand;
- T2→T3 before/after the hidden change: effect of non-stationarity;
- T3↔T4: effect of hiding exact shipment timing;
- Tier 5 headline↔controls: effect of an aggressive claimant and an allocation
  rule that rewards order size.

The ladder does not establish optimal control, general multi-agent cooperation,
human-like reasoning, or robustness to arbitrary adversaries. Tier 5 uses a fixed
scripted stress policy. True multi-model play and adaptive best-response opponents
remain later research modes.

## 10. Required ladder tests

1. Scenario serialization contains every parameter above and distinguishes public
   from grader-private fields.
2. Constant demand consumes no RNG and returns 8 for all operational weeks.
3. AR(1) samples follow the specified recurrence, rounding, and non-negative clamp.
4. Shock selections are deterministic, namespaced, and include both directions in
   development and validation manifests.
5. Tier 2 and Tier 3 paired innovations match before and after the shift; only the
   mean law changes.
6. Tier 3 and Tier 4 produce identical hidden transitions for fixed actions.
7. Aggregate observation mode never exposes shipment arrival slots.
8. Capacity is exactly 22 for every Tier 5 seed.
9. Aggressive counterpart orders equal `min(128, base_order + 8)`.
10. Swapping retailer A/B labels under a correspondingly swapped seed/action trace
    leaves Tier 5 metrics symmetric.
11. Tier 5 proportional and uniform controls differ only in allocation policy.
12. Generated split manifests have unique seeds and stable canonical hashes.
