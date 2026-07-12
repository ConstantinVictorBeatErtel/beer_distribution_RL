# M2 IPPO results (classic step demand)

## Summary

| Agent | Mean system cost / period | Notes |
|---|---|---|
| IPPO Regime A (seed 0, 100k) | **10.87** | Local rewards; beats BS & Sterman |
| IPPO Regime C (seed 0, 200k) | **24.91** | System reward; independent policies; beats BS & Sterman |
| Base-stock (S=20 all) | 53.23 | Heuristic levels for classic step |
| Sterman | 404.03 | Amplifies bullwhip |

Regime A (selfish) currently beats Regime C (shared reward) on this classic cell — interesting for the emergence story; revisit with more seeds in M3.

## Emergence constraints verified

- Four separate `ActorCritic` modules; parameter-id uniqueness asserted at init and in tests.
- Regime A rewards = −local cost; Regime C rewards = −system cost (env-enforced).
- No signaling / no honesty shaping in M2.
- Relative actions Δ∈[-8,8]; does not change the scientific claim (still independent local policies).

## Reproduce

```bash
python scripts/train_ippo.py --config experiments/regime_a_classic.yaml --seed 0 --total-timesteps 100000
python scripts/train_ippo.py --config experiments/regime_c_classic.yaml --seed 0 --total-timesteps 200000
python scripts/eval_ippo.py --run-dir artifacts/runs/ippo/regimeA_seed0
python scripts/eval_ippo.py --run-dir artifacts/runs/ippo/regimeC_seed0
```
