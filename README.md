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

## Tier-1 IPPO (M2–M3)

```bash
# M2 classic anchors
python scripts/train_ippo.py --config experiments/regime_a_classic.yaml --seed 0
python scripts/train_ippo.py --config experiments/regime_c_classic.yaml --seed 0

# M3 phase-diagram matrix (Regime B × capacity × rationing × seeds) — legacy
python scripts/run_m3_matrix.py --skip-existing
make figures

# Tier-1 v1.1 matrix (A/B/C × serial/y × caps × demands) — parallel + vec envs
python scripts/run_tier1_matrix.py --dry-run          # show pruned cell count
python scripts/run_tier1_matrix.py --workers 8 --n-envs 64 --skip-existing
# Colab: notebooks/colab_tier1.ipynb (Drive-mounted, resumable)
```

One policy + critic **per role** (no parameter sharing). Regime C shares only the system reward. Regime B adds optional unverified signals.

## Package layout

```
beer_distribution_rl/
  env/          # pure-Python core (no ML deps)
  agents/
    baselines.py
    ippo/       # Independent PPO
```
