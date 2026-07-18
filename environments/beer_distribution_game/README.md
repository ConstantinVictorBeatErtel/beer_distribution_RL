# Beer Distribution Game

A native Verifiers v1 environment for delayed supply-chain control. One model
controls one role through a strict `place_order(quantity)` tool while deterministic
scripted policies control the other roles.

The environment provides five tiers: steady demand, persistent stochastic demand,
a hidden regime shift, partial pipeline observability, and strategic scarcity in a
two-retailer Y network. Episodes are seeded, exactly replayable, and graded from
the simulator trace. The primary result is controlled-role cost normalized against
a same-seed adaptive base-stock reference; service, bullwhip, and system externality
remain separate metrics.

## Local validation

From this directory with `uv` installed:

```bash
uv sync
uv run validate beer-distribution-game --runtime.type subprocess
```

Example dry-run configuration:

```bash
uv run eval @ eval.toml --dry-run True
```

The package exports one `BeerTaskset` and one bundled `BeerHarness`, following the
native Verifiers 0.2 contract. It requires no API keys for simulation or grading.
The checked-in evaluation config disables result upload by default.

See the repository-level specifications for the complete interface, reward, and
difficulty contracts.
