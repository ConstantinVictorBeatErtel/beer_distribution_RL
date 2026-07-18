# Beer Distribution Agent Environment — Interface Specification

**Status:** Stage 2 approved, 2026-07-18
**Primary target:** Prime Intellect Environments Hub via Verifiers
**Secondary target:** HUD v6 adapter after Verifiers parity
**Scope:** Order-only v1; one model-controlled role against scripted counterparties

## 1. Purpose

The environment evaluates whether an LLM agent can operate one role in a delayed,
partially observed supply chain. The agent must balance inventory and backlog cost,
adapt to demand, and avoid amplifying variability. The primary artifact is an
agent environment with deterministic state transitions and programmatic grading,
not a policy-network training wrapper.

The simulator, scenario definition, transcript, and grader are framework-neutral.
Verifiers and HUD are adapters over the same canonical episode API.

## 2. v1 decisions

| Decision | v1 choice | Reason |
|---|---|---|
| Agent exposure | One controlled role per rollout | Cheap, attributable, and reproducible across model providers |
| Other roles | Deterministic scripted counterparties | Removes opponent sampling noise and supports exact replay |
| Role coverage | Separate tasks for every role | Prevents a retailer-only benchmark from hiding echelon-specific difficulty |
| Topologies | Serial and Y | Serial is the canonical control; Y contains the competing-claimant mechanism |
| Strategic channel | Orders only | The existing cheap-talk channel was not load-bearing |
| Action interface | Strict `place_order` tool call | Native LLM-agent interface with unambiguous validation |
| Action value | Absolute integer quantity | Avoids the current demand-relative +/-8 recovery constraint |
| Default horizon | 36 weeks | Long enough for delay effects while keeping rollouts inexpensive |
| Stress horizon | 52 weeks | Reserved for a harder tier and research comparability |
| Model-visible history | Current state plus last 8 own-role records | Bounded token cost and framework-independent memory contract |
| Primary objective | Controlled role's local operating cost | Preserves the self-interested-agent research question |
| Primary distribution | Verifiers environment module | Required for the Prime Intellect Environments Hub |
| Secondary distribution | HUD v6 capability adapter | Demonstrates portability without making HUD a core dependency |

Cheap talk, free-form inter-agent messages, true multi-model play, self-play, and
GRPO orchestration are out of scope for v1.

## 3. Episode model

One episode is identified by a canonical `ScenarioSpec` and one controlled role.
The scenario fixes all exogenous behavior. The model produces only the controlled
role's replenishment order. Counterparties act simultaneously from their own local
observations using fixed policy versions.

For each scenario seed, the task generator SHOULD create one rollout per role:

- Serial: retailer, wholesaler, distributor, factory.
- Y: retailer A, retailer B, wholesaler, distributor, factory.

These rollouts share a demand-generation specification but are independent
episodes. A model never sees the private state from a rollout controlling another
role.

### Why this is not true multi-agent v1

True multi-agent LLM evaluation couples model sampling, provider failures, context
management, and role-specific policies inside one score. It is expensive and makes
causal attribution difficult: a retailer can fail because a separately sampled
factory failed. Single-role control supports clean comparisons against random,
pass-through, and base-stock policies and permits identical counterparty behavior
for every evaluated model.

True multi-agent play remains a later environment mode only after the single-role
benchmark is stable.

## 4. Weekly decision boundary

The agent must make its order after observing the events a Beer Game player knows
at the ordering decision. Each week is split into two phases.

### Phase A — prepare decision

1. Advance and receive inbound shipments.
2. Reveal this week's customer demand or delayed downstream order.
3. Add prior backlog.
4. Fill demand/orders from available inventory and update backlog.
5. Construct each role's local decision observation.

### Phase B — commit decisions

1. Validate exactly one order for every role.
2. Append orders to the upstream order pipelines.
3. Apply factory production capacity.
4. Accrue end-of-week holding and backlog costs.
5. Record the immutable week transition.
6. Begin the next week or terminate at the horizon.

This decision boundary intentionally differs from the current monolithic `step()`
call, whose caller selects orders before the new week's receipts and demand are
revealed. The environment-facing core must expose `prepare_week()` and
`commit_orders()` (names provisional), with an invariant preventing multiple
prepares or commits for one week.

## 5. Initial task prompt

The initial user message is concise and generated entirely from the scenario. It
contains:

- controlled role and topology;
- objective: minimize the controlled role's local cost;
- holding and backlog cost coefficients;
- order and shipment delays;
- factory capacity, if it is public in the scenario;
- horizon and remaining weeks;
- order bounds;
- a statement that counterparties are scripted but their private state is hidden;
- the `place_order` requirement;
- the first observation.

The prompt must not describe an unimplemented coordination goal, imply that system
cost is rewarded, reveal future demand, or reveal another role's inventory,
backlog, pipeline, policy parameters, or next action.

## 6. Observation schema

The canonical observation is structured data. Adapters may render it as compact
JSON or equivalent text but may not add information.

```json
{
  "episode_id": "sha256:...",
  "scenario_id": "y-stochastic-v1",
  "week": 7,
  "horizon": 36,
  "weeks_remaining": 30,
  "role": "retailer_a",
  "topology": "y",
  "state": {
    "inventory_on_hand": 6,
    "backlog": 2,
    "inventory_position": 17,
    "on_order": 13,
    "shipment_received": 4,
    "incoming_demand_or_order": 8,
    "units_filled": 6,
    "last_order_placed": 9,
    "inbound_shipment_pipeline": [4, 9]
  },
  "costs": {
    "holding_per_unit": 0.5,
    "backlog_per_unit": 1.0,
    "current_inventory_backlog_cost": 2.0,
    "cumulative_local_cost_through_previous_week": 16.5
  },
  "constraints": {
    "minimum_order": 0,
    "maximum_order": 128,
    "factory_capacity": 22
  },
  "recent_history": [
    {
      "week": 6,
      "incoming_demand_or_order": 7,
      "shipment_received": 5,
      "order_placed": 9,
      "ending_inventory": 4,
      "ending_backlog": 0,
      "local_cost": 2.0
    }
  ]
}
```

### Observation rules

- `week` is 1-indexed and names the decision currently being made.
- All `state` values describe the controlled role after Phase A and before its
  replenishment order.
- `incoming_demand_or_order` means customer demand for a retailer and the received
  downstream order for an upstream role.
- `on_order` includes shipments in transit plus upstream unfilled replenishment.
- `inbound_shipment_pipeline` is present only in observation modes with advance
  shipment notices. It contains shipments already dispatched to the controlled
  role, never another role's order pipeline.
- A delayed downstream order that has been placed but has not arrived is future
  private information. It is never exposed through an `order_pipeline` field.
- `recent_history` contains at most the previous 8 accepted decisions and outcomes.
- Other roles' private values are forbidden.
- Future demand samples, hidden regime state, and counterparty actions are forbidden.
- Full transcripts are retained for audit even though the model-visible window is 8.

The partial-observability difficulty tier may replace detailed pipeline arrays with
their aggregate `on_order`; it must do so through an explicit scenario flag rather
than adapter-specific prompt changes.

### Context delivery

The history window is part of the benchmark, not a prompt-format preference. On
each decision, adapters provide the fixed task instructions plus the current
canonical observation. They must trim or reconstruct the provider conversation so
that earlier tool results outside `recent_history` are not also visible in chat
history. The complete untrimmed transcript remains available to the evaluator but
not to the model.

## 7. Action/tool schema

The primary environment exposes one agent-callable tool:

```json
{
  "name": "place_order",
  "description": "Place this week's replenishment order. Call exactly once per week.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "quantity": {
        "type": "integer",
        "minimum": 0,
        "maximum": 128
      }
    },
    "required": ["quantity"],
    "additionalProperties": false
  }
}
```

Rules:

- Exactly one `place_order` call is accepted per decision week.
- The environment does not accept a plain-text number as an action.
- Boolean, float, string, missing, negative, and over-cap quantities are invalid.
- Multiple calls in one assistant turn are invalid; no first-call-wins behavior.
- Invalid actions never advance state or consume randomness.
- The tool has no `rationale` field. Free-form explanations are not inputs to the
  primary programmatic score.
- The environment never clamps, rounds, defaults, or silently repairs an action.

### Tool result

An accepted non-terminal call returns:

```json
{
  "status": "accepted",
  "completed_week": 7,
  "order_placed": 9,
  "done": false,
  "next_observation": {"...": "canonical observation for week 8"}
}
```

On the final week it returns `done: true`, the termination reason, and a factual
episode summary. It does not ask the model to produce a separate final answer.

### Protocol errors

An invalid or out-of-turn action returns a structured error with the unchanged
week number and increments `protocol_error_count`. The model receives one repair
opportunity for that decision. A second invalid attempt in the same week, or a
third protocol error anywhere in the episode, terminates with
`termination_reason="protocol_error"`.

The exact validity reward is a Stage 3 decision. Environment errors and protocol
errors must remain distinguishable.

## 8. Scripted counterparties

Counterparties implement a framework-neutral protocol:

```python
class CounterpartyPolicy(Protocol):
    policy_id: str
    policy_version: str

    def reset(self, seed: int, role: Role, scenario: ScenarioSpec) -> None: ...
    def act(self, observation: DecisionObservation) -> int: ...
```

The v1 default is a deterministic, scenario-calibrated base-stock policy. Target
levels are fixed before frontier-model evaluation and versioned in the scenario;
they may not be tuned on held-out evaluation seeds or after observing model
results.

Pass-through, Sterman, random, and adversarial policies are baseline or difficulty
variants, not silent changes to the default counterparty.

If a counterparty is stochastic, it receives a distinct derived RNG stream. Its
randomness must never consume from the demand stream or the controlled agent's
protocol stream.

## 9. Termination

Normal termination occurs only when the configured horizon is completed.

Early termination reasons are:

- `protocol_error`: the model exhausted its invalid-action allowance;
- `environment_error`: an invariant or infrastructure failure;
- `cancelled`: the evaluator explicitly cancelled the rollout.

The agent cannot voluntarily finish early. High cost, backlog, or inventory does
not terminate an episode because that would make catastrophic policies appear
cheaper by shortening their exposure.

An `environment_error` is excluded from model scoring and reported separately. A
protocol error is attributable to the model and remains in evaluation statistics.

## 10. Scenario and seed contract

`ScenarioSpec` is immutable and JSON-serializable. At minimum it contains:

- schema and environment version;
- topology and role set;
- horizon;
- demand process name and all numeric parameters;
- explicit integer capacity or `null` for unlimited capacity;
- cost coefficients per role;
- lead times and initial state;
- order cap;
- rationing rule;
- observation mode and history window;
- counterparty policy IDs, versions, and fixed parameters;
- master seed and split (`development`, `validation`, or `test`).

Capacity is never estimated from the episode seed. For example, Y demand with two
streams of theoretical mean 7.5 uses explicit calibrated capacity 22. Under the
corrected v2 reference policy it binds in 55.6% of validation base-rival weeks,
inside the predeclared 10%--70% calibration band.

### RNG derivation

The master seed is expanded with a stable cryptographic derivation, not Python's
process-randomized `hash()`:

```text
scenario/master
  -> demand
  -> initialization
  -> counterparty/<role>
  -> optional-mechanism/<name>
```

Each episode owns a fresh demand-process instance. Mutable demand objects are never
stored in a shared config or reused across vector environments.

### Episode identity and replay

`episode_id` is the SHA-256 digest of canonical scenario JSON plus the controlled
role. Every trace records:

- episode ID and complete scenario JSON;
- environment package version and git SHA;
- controlled role;
- raw model messages and tool calls;
- validated actions;
- every canonical transition;
- counterparty policy IDs/versions;
- termination reason;
- grader version and all metric components;
- model/provider identifier and decoding parameters when available.

Environment replay means that the recorded action sequence produces the exact same
transition and metric trace. It does not require a provider to reproduce identical
model tokens. Replay tests compare canonical serialized traces byte-for-byte.

## 11. Framework adapters

### Verifiers

The reference adapter targets native Verifiers v1 as shipped in stable Verifiers
0.2.0. That API discovers one exported typed `Taskset` and, when bundled, one
exported `Harness` through the package's `__all__`. New v1 packages must not add the
legacy `load_environment`, `load_taskset`, or `load_harness` functions.

The taskset owns typed scenario rows, lifecycle state, tools, rewards, and metrics.
The bundled harness owns only model interaction and the rolling-context protocol;
both delegate simulation and grading to framework-neutral objects.

### HUD

The optional HUD v6 adapter exposes the same episode through a custom MCP
capability and task template. It must use the canonical scenario, action validator,
transition records, and grader. A cross-adapter test replays the same actions and
requires identical canonical traces and metric components.

HUD types are optional dependencies and may not be imported by the simulator core.

## 12. Required conformance tests

Before any model baseline is reported:

1. Same scenario plus same actions produces byte-identical traces across repeated runs.
2. Vector environments have distinct demand-object identities and independent streams.
3. Resetting one episode cannot change another episode's latent demand state.
4. Capacity is identical across seeds within a named difficulty cell.
5. Serial and Y role lists match their public action interfaces.
6. Missing Y-retailer actions fail validation rather than defaulting to zero.
7. Invalid actions leave state and RNG state unchanged.
8. Current-week decision observations match the documented event order.
9. No observation contains another role's private state or future demand.
10. Counterparty policies are deterministic under their derived seeds.
11. Environment errors are not scored as model failures.
12. Verifiers and HUD adapters produce identical canonical traces for fixed actions.

## 13. Migration boundary

The existing `BeerGameCore` remains the source of domain logic where its semantics
match this document. The implementation must repair or replace:

- shared mutable demand objects;
- seed-derived capacity values;
- the action-before-observation weekly boundary;
- hard-coded serial role lists in wrappers;
- silent action defaults and clamps;
- exposure of not-yet-arrived downstream orders through `order_pipeline`;
- mixed stochastic/greedy evaluation behavior;
- incomplete episode transcripts.

Legacy Gym/PettingZoo/IPPO code may adapt to the repaired core, but it does not
define the Hub environment contract.
