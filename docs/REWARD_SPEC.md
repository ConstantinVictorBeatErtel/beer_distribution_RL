# Beer Distribution Agent Environment — Reward and Grading Specification

**Status:** Stage 3 approved, 2026-07-18
**Depends on:** [`ENVIRONMENT_SPEC.md`](ENVIRONMENT_SPEC.md), approved 2026-07-18
**Scope:** Order-only v1; one controlled role against scripted counterparties

## 1. Grading principle

The benchmark measures performance in a self-interested supply-chain role. Its
headline outcome is the controlled role's undiscounted local operating cost. The
Hub needs a scalar reward, so that cost is converted to a bounded monotonic score
using a frozen base-stock counterfactual on the same scenario and seed.

Service, bullwhip, total supply-chain cost, and protocol compliance are all
reported. They are not mixed into a hand-weighted scalar. Keeping the components
separate makes failure modes visible: a policy cannot offset poor service with a
smooth order stream, or hide damage to other roles behind low private cost.

The grader is deterministic code over the canonical transition trace. No language
model, regex over prose, or self-reported outcome participates in the primary
reward.

## 2. Cost accounting

For role `r` after fulfillment in operational week `t`:

```text
weekly_local_cost[r,t] =
    holding_cost[r] * ending_inventory[r,t]
  + backlog_cost[r] * ending_backlog[r,t]
```

All quantities are non-negative integers and cost coefficients are finite,
non-negative scenario constants. Costs are accumulated without discounting.

For the controlled role `a` over `H` decision weeks:

```text
operational_local_cost = sum(t=1..H, weekly_local_cost[a,t])
```

The same calculation over every role gives `operational_system_cost`.

### Settlement continuation

A visible finite horizon invites a last-week exploit: stop ordering because late
orders and backlog consequences fall just outside the episode. After week `H`, the
grader therefore runs a deterministic settlement continuation for

```text
settlement_weeks = order_delay + shipment_delay
```

No model calls occur in settlement. New exogenous customer demand is zero, every
role places zero new orders, and already-committed order and shipment pipelines
continue to advance. Holding and backlog costs continue to accrue. Settlement
uses no randomness.

After the continuation, the grader applies a one-period terminal exposure charge
using the controlled role's inventory position:

```text
terminal_inventory_position =
    ending_inventory + ending_on_order - ending_backlog

terminal_exposure_cost =
    holding_cost * max(terminal_inventory_position, 0)
  + backlog_cost * max(-terminal_inventory_position, 0)
```

This values excess stock already owned or committed and shortages not covered by
outstanding replenishment. It closes the remaining case where an oversized final
order is still recorded as an unfilled upstream commitment after the pipeline
tail. The same calculation is made independently for every role.

The primary cost is:

```text
local_total_cost =
    operational_local_cost
  + settlement_local_cost
  + terminal_exposure_cost
```

`system_total_cost` is defined analogously. Operational, settlement, and terminal
components are also reported separately. The settlement rule is part of
`ScenarioSpec` and may not vary by model.

Settlement is a valuation tail, not an extra gameplay phase. Service and bullwhip
metrics use only the `H` operational weeks.

## 3. Primary outcome and Hub scalar

### Frozen paired reference

For each evaluated episode, the evaluator runs the versioned base-stock policy as
the controlled role under the exact same `ScenarioSpec`, master seed, scripted
counterparties, and settlement rule. Demand and counterparty random streams are
therefore paired, but the reference rollout is an independent episode and cannot
share mutable state with the model rollout.

Let:

- `C_agent` be the model's `local_total_cost`;
- `C_base` be the paired base-stock policy's `local_total_cost`.

The reference policy parameters and version are frozen before test evaluation and
recorded in every result. Test seeds are not used to tune it.

### Cost score

For a protocol-clean completed episode with `C_base > 0`:

```text
cost_score = C_base / (C_base + C_agent)
```

Properties:

- range `(0, 1]` for finite non-negative costs;
- lower agent cost always produces a higher score;
- matching base-stock scores `0.5`;
- beating base-stock scores above `0.5`;
- zero agent cost scores `1.0`;
- no clipping hides differences between very good or very poor policies.

If `C_base == 0`, the score is `1.0` only when `C_agent == 0`, otherwise `0.0`.
Zero-cost reference cells should be rejected during scenario validation because
they carry little control signal.

The raw `local_total_cost`, not the normalized score, remains the headline
scientific outcome. The normalization exists for cross-cell Hub aggregation.

### Protocol gate

```text
episode_reward = protocol_clean * cost_score
```

`protocol_clean` is `1` only if every decision was accepted on its first attempt.
It is `0` if any protocol error occurred, including one repaired successfully.
This prevents an agent from buying extra inference calls through deliberate invalid
actions. Cost diagnostics may still be reported for a completed repaired episode,
but its official reward is zero.

A protocol-terminated episode also scores zero and its partial cost is labeled
incomplete rather than compared with completed costs. An `environment_error` has
no score and is excluded from model aggregates; it is counted and reported as an
infrastructure failure.

There is no separate formatting bonus, reasoning bonus, dense shaping term, or
system-cost term in the official reward.

## 4. Mandatory diagnostic metrics

Every completed episode records the following metrics. Diagnostics never alter
`episode_reward`.

### Cost and externality

- `operational_local_cost`
- `settlement_local_cost`
- `terminal_exposure_cost`
- `local_total_cost`
- `mean_weekly_local_cost = operational_local_cost / H`
- `system_total_cost`
- `other_roles_total_cost = system_total_cost - local_total_cost`
- paired base-stock versions of the same values
- `local_cost_ratio = C_agent / C_base` when `C_base > 0`
- `system_cost_ratio` against the paired base-stock system cost when defined

The local/system split is essential in the Y topology. Order inflation that wins a
larger proportional allocation may be rational under the disclosed local objective
while imposing costs on the rival or upstream roles. That is a result to expose,
not silently remove by changing the reward.

### Service level

For operational week `t` and downstream claimant `c`, let:

- `D_c,t` be newly arrived customer demand for a retailer's virtual customer or
  the newly arrived order from claimant `c` for an upstream role;
- `B_prev_c,t` be claimant `c`'s backlog before adding `D_c,t`;
- `Q_c,t` be units allocated and shipped to claimant `c` that week.

With backlog served before new demand, units filled immediately are:

```text
F_immediate_c,t = min(D_c,t, max(0, Q_c,t - B_prev_c,t))
```

The retailer has one virtual claimant. The Y wholesaler is calculated separately
for retailer A and retailer B before aggregation; aggregate backlog arithmetic is
not equivalent under multi-claimant rationing.

The canonical evaluator trace must therefore record per-claimant new demand,
backlog before new demand, and allocation. These fields remain grader-private and
must not be reconstructed from aggregate values after the fact.

The grader reports:

```text
immediate_fill_rate = sum(c,t, F_immediate_c,t) / sum(c,t, D_c,t)
cycle_service_level =
    mean(F_immediate_c,t == D_c,t for claimant-weeks where D_c,t > 0)
horizon_fulfillment =
    (sum(c,t, D_c,t) - ending_total_backlog_at_H) / sum(c,t, D_c,t)
```

If `sum(c,t, D_c,t) == 0`, the two unit-based rates are `null`, not `1.0`. Cycle
service is also `null` if there are no operational claimant-weeks with positive
demand. Terminal backlog at `H` is reported directly. Settlement shipments do not
retroactively improve operational service.

### Bullwhip and order stability

Let `O_t` be the controlled role's replenishment order and `D_t` its newly arrived
local demand/order. Metrics exclude a fixed startup window:

```text
metric_warmup_weeks = order_delay + shipment_delay
```

Using sample variance (`ddof=1`) on the remaining operational weeks:

```text
bullwhip_ratio = variance(O) / variance(D)
```

The ratio is `null` when fewer than two scored observations remain or
`variance(D) <= 1e-12`. It is never encoded as infinity. This matters for the
constant-demand tier, where the classical bullwhip ratio is mathematically
undefined.

The always-defined companion, when at least two scored orders remain, is:

```text
normalized_order_volatility =
    mean(abs(O_t - O_(t-1))) / max(mean(D_t), 1)
```

The grader also records order mean, sample variance, maximum, order-cap hit rate,
and the ratio `sum(O_t) / sum(D_t)` when total demand is positive. A low bullwhip
ratio is never interpreted without cost and service.

### Protocol and execution

- `protocol_clean`
- error count and error categories
- accepted decisions / expected decisions
- termination reason
- environment-error category, if any
- model calls and repair calls
- input/output tokens and latency when the provider exposes them

## 5. Aggregation and uncertainty

The atomic comparison is a paired `(scenario, controlled_role, seed)` bundle.
Reports must retain those identifiers and may not pool away role or difficulty.

For every model, tier, topology, and role, report:

- completed and protocol-clean episode counts;
- raw local and system cost mean, standard deviation, and median;
- mean and standard deviation of `episode_reward`;
- mean service and stability diagnostics with undefined counts;
- paired local-cost ratio to base-stock;
- per-seed rows in a machine-readable results file.

Confidence intervals use a paired bootstrap over seed bundles, so all policies and
roles for a seed are resampled together. The final benchmark macro-average gives
equal weight to each declared `(tier, topology, role)` cell; it does not let easy
or numerous cells dominate. Both the macro score and every component cell are
published.

Random and base-stock policies are ordinary evaluated policies in the results
table, not only hidden normalization machinery. Base-stock should score `0.5` in
its paired cells; failure of that invariant is a grader error.

## 6. Reasoning-quality rubric

The v1 Hub reward does **not** grade chain-of-thought or a model-written rationale.
The environment observes decisions, not the private reasoning that produced them.
A fluent explanation can be generated after the fact, judge models introduce
variance and bias, and requiring prose would increase evaluation cost while
changing the task.

For qualitative research only, an optional offline strategy-audit task may sample
completed traces and ask the evaluated model for a memo of at most 200 words. It is
a separate artifact and never enters the environment reward or leaderboard. A
blinded judge scores four dimensions from 0–2 and must cite trace week numbers:

1. **State grounding:** quantitative claims agree with the trace.
2. **Delay-aware causal account:** the memo distinguishes orders, shipments, and
   their lead times.
3. **Trade-off recognition:** it identifies inventory, backlog, service, and any
   local-versus-system externality relevant to the episode.
4. **Calibration:** it distinguishes supported conclusions from uncertainty and
   names at least one plausible alternative policy or failure cause.

The audit reports dimension scores, judge identity/version, and disagreements on
a dual-judged audit subset. It is described as an explanation-quality probe, not a
measurement of hidden reasoning.

## 7. Reward-hacking analysis

| Signal or surface | Plausible attack | Mitigation or interpretation |
|---|---|---|
| Finite-horizon local cost | Stop ordering or place oversized orders near the end so consequences arrive after termination | Deterministic no-demand/no-order settlement advances committed pipelines; a terminal inventory-position charge values remaining shortages and commitments |
| Local cost | Inflate orders under proportional rationing to externalize shortage costs to a rival | Preserve as legitimate self-interested behavior; always publish system, other-role, allocation, and service diagnostics; use a separate system objective only in a separately named task |
| Local cost | Keep orders near zero and accept a backlog spiral | Backlog cost accumulates every week and in settlement; service and terminal backlog make the failure visible |
| Normalized score | Tune against or memorize the base-stock reference and public seeds | Freeze policy/version before evaluation, separate development/validation/test manifests, and publish test manifests only after the benchmark version is frozen |
| Normalized score | Exploit clipping or a weak random denominator | Use the monotonic unclipped base-stock ratio; publish raw costs; reject zero-cost reference cells |
| Protocol retries | Submit an invalid call deliberately to obtain extra context or inference time | Any protocol error makes the official episode reward zero, even if repaired |
| Early exit | Trigger termination before expensive backlog weeks | Protocol termination scores zero; voluntary early completion is unavailable; partial costs are never ranked with complete episodes |
| Service level | Claim perfect service in a zero-demand episode | Zero denominators produce `null`; demand totals and undefined counts are reported |
| Horizon fulfillment | Serve old backlog late and present it as immediate service | Immediate fill is computed backlog-first from transition data; settlement never improves operational service |
| Service level | Overorder aggressively to maximize availability | Service is diagnostic, not a reward term; local/system costs and order-cap hits expose the trade-off |
| Bullwhip ratio | Emit a constant zero order stream | Bullwhip is diagnostic only and must be read with service, cost, order/demand ratio, and terminal backlog |
| Bullwhip ratio | Exploit nearly constant demand to create unstable division | Return `null` below the variance threshold and report normalized order volatility instead |
| Bullwhip ratio | Hide startup oscillation by manipulating the warmup | Warmup is a scenario constant shared by all policies; full per-week actions remain published |
| System cost | Let deterministic counterparties dominate the number, obscuring agent quality | Report local and other-role costs separately and stratify by controlled role |
| Reasoning rubric | Keyword stuffing, eloquent post-hoc rationalization, or fabricated facts | Exclude it from reward; require trace citations; score factual grounding; blind the judge; dual-judge an audit subset |
| Trace content | Inject instructions into a downstream judge through model prose | Primary grader never invokes a model; optional judge sees data in a quoted/untrusted trace envelope and has no tools |
| Arithmetic | Cause overflow, NaN, negative quantities, or non-finite scores | Integer order bounds, finite scenario validation, checked invariants, and fail-closed environment errors |
| Failure classification | Trigger an implementation bug with a valid action so a bad episode is excluded as an environment error | Property-test every bounded action path; publish environment-error rates; invalidate and repair an affected benchmark version rather than treating exclusion as model success |
| Seed handling | Influence RNG consumption through invalid actions or shared mutable demand | Invalid actions consume no RNG; streams are namespaced; every episode owns fresh demand state; replay is byte-tested |

## 8. Grader output contract

The framework-neutral grader returns a JSON-serializable record with at least:

```json
{
  "grader_version": "1.0.0",
  "status": "scored",
  "episode_reward": 0.57,
  "protocol_clean": true,
  "primary": {
    "local_total_cost": 412.5,
    "paired_base_stock_local_total_cost": 540.0,
    "cost_score": 0.57
  },
  "costs": {
    "operational_local_cost": 390.5,
    "settlement_local_cost": 16.0,
    "terminal_exposure_cost": 6.0,
    "system_total_cost": 1480.0,
    "other_roles_total_cost": 1067.5
  },
  "service": {
    "immediate_fill_rate": 0.91,
    "cycle_service_level": 0.78,
    "horizon_fulfillment": 0.96,
    "ending_backlog": 11
  },
  "stability": {
    "bullwhip_ratio": 1.42,
    "normalized_order_volatility": 0.31,
    "order_cap_hit_rate": 0.0
  },
  "termination_reason": "horizon_completed"
}
```

Unscored environment failures use `status="environment_error"` and
`episode_reward=null`. Undefined diagnostics are JSON `null`, never NaN or
infinity.

## 9. Required grading tests

Before implementation is accepted:

1. Hand-calculated traces match weekly, operational, settlement, terminal, and system costs.
2. Lower agent cost always implies a higher cost score for fixed reference cost.
3. A base-stock rollout scored against itself is exactly `0.5` when its cost is positive.
4. Any protocol error makes official reward zero without changing the cost trace.
5. Environment errors produce no score and remain visible in aggregate counts.
6. Settlement uses zero new demand/orders, consumes no RNG, and advances committed pipelines.
7. Immediate fill rate distinguishes old-backlog service from new-demand service and is calculated per claimant in Y.
8. Zero-demand service metrics and constant-demand bullwhip are `null`.
9. Bullwhip uses sample variance and the fixed warmup window.
10. All grader outputs serialize with no NaN or infinity.
11. Regrading the same canonical trace is byte-identical.
12. Seed-bundle aggregation is invariant to input row order.
