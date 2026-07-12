# Beer Distribution RL

Emergent cooperation and deception in supply-chain agents playing an extended Beer Distribution Game.

See [PROJECT_SPEC.md](PROJECT_SPEC.md) and [DECISIONS.md](DECISIONS.md).

## Install

```bash
pip install -e ".[dev]"          # env + tests
pip install -e ".[dev,wrappers]" # + PettingZoo / Gymnasium
```

## Validation gate

```bash
python scripts/validation_gate.py
```

No training code until this gate passes (Sterman bullwhip + base-stock costs vs published baseline).

## Package layout

```
beer_distribution_rl/
  env/       # pure-Python core (no ML deps)
  agents/    # baselines; IPPO/LLM arrive in later milestones
```
