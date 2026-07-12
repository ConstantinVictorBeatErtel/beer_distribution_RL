# Beer Distribution RL

Emergent cooperation and deception in supply-chain agents playing an extended Beer Distribution Game.

See [PROJECT_SPEC.md](PROJECT_SPEC.md) and [DECISIONS.md](DECISIONS.md).

## Install

```bash
pip install -e ".[dev]"               # env + tests
pip install -e ".[dev,wrappers]"      # + PettingZoo / Gymnasium
pip install -e ".[dev,wrappers,marl]" # + PyTorch IPPO
```

## Validation gate (M1)

```bash
python scripts/validation_gate.py
```

## Tier-1 IPPO (M2)

One policy + critic **per role** (no parameter sharing). Regime C shares only the system reward.

```bash
python scripts/train_ippo.py --config experiments/regime_a_classic.yaml --seed 0
python scripts/train_ippo.py --config experiments/regime_c_classic.yaml --seed 0
python scripts/eval_ippo.py --run-dir artifacts/runs/ippo/regimeA_seed0
```

## Package layout

```
beer_distribution_rl/
  env/          # pure-Python core (no ML deps)
  agents/
    baselines.py
    ippo/       # Independent PPO
```
