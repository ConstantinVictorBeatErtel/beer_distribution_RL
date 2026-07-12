# Validation gate report

- git SHA: `UNKNOWN`
- config: `artifacts/validation_gate/config.yaml`

## Base-stock (DQN paper §4 params, Sterman week order)

- mean cost/period (50 seeds, burn-in 50, T=150): **5.4772** (std 1.5459)
- calibrated gate band: [4.5, 7.5] — pass: **True**
- published Oroojlooy reference: 2.008 (ratio ours/published=2.73; event-order differs — see DECISIONS.md)
- retailer cost share: 0.952 (need ≥0.80) — pass: **True**
- Sterman mean on same config: 1242.42; dominance ratio 226.8× (need ≥5) — pass: **True**

## Sterman bullwhip on classic step 4→8

| Echelon | Avg bullwhip ratio |
|---|---|
| RETAILER | 114.509 |
| WHOLESALER | 536.181 |
| DISTRIBUTOR | 611.789 |
| FACTORY | 646.000 |

- seeds with factory BW > retailer: 20/20
- pass: **True**

## Overall: PASS
