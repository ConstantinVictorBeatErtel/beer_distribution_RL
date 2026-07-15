# Eval-mode blast radius (matched-deterministic re-eval)

**Baseline SHA:** `061aa59235397b7360c32a01cf4f98add0dd503a`

Eval-only re-run on frozen Tier-1 v11 checkpoints. `n_episodes=20`, 10 seeds/cell, `840` runs, wall ≈ 1.5 min. No training / reward / env changes.

**RETRACTED:** all logged Tier-1 A−B scarcity %-gaps from mismatched `final_eval` (headline band ~28–53%, plus Y / `regime_switch` / rationing siblings). Use the matched-deterministic A−B table below.

## Step 0 — Root cause

**Verdict: (a) per-regime config field** — a real bug. `IPPOTrainer.evaluate` sets action mode from `self.signaling`, so Regime B (signaling=True) is evaluated stochastically while A/C (signaling=False) are evaluated with argmax.

```455:455:beer_distribution_rl/agents/ippo/trainer.py
                        a, _, _ = self._policy_act(r, o, greedy=not self.signaling)
```

Same pattern in the analysis helper `analysis/diag/eval_ablation.py` (`greedy=not signaling`). Not an analysis-time table default: every `final_eval.json` written by the matrix runner inherits this coupling.

| Regime | `signaling` | `greedy=not signaling` | Mode |
|---|---|---|---|
| A | False | True | deterministic (argmax) |
| B | True | False | stochastic (sample) |
| C | False | True | deterministic (argmax) |

## Step 1 — Contaminated outputs (blast radius)

Any **cross-regime cost comparison that includes Regime B** and was sourced from `final_eval.json` / `tier1_v11/index.json` is contaminated. A-vs-C comparisons were already matched-deterministic (both non-signaling) and are listed for completeness; they should not move under this re-eval.

### Contaminated (mismatched mode)

| Output | Why |
|---|---|
| Tier-1 v11 `final_eval.json` × all Regime-B cells (and any A/B or B/C gap derived from `index.json`) | B stochastic vs A/C greedy |
| Headline 28–53% A−B scarcity gaps (serial/Y × AR(1)/regime_switch × prop/uniform/honesty_weighted × {1.0μ,0.8μ}) | Primary published confound |
| All Y-topology A−B / B−C cost comparisons in the matrix | Same `evaluate` path |
| All `regime_switch` A−B / B−C cost comparisons | Same `evaluate` path |
| C-vs-B (and B-vs-C) at every capacity including ∞ | B stoch vs C det |
| `artifacts/diagnostics/v11_ablation.md` “Context” stochastic columns (`B stoch (logged)`, `A−B stoch`) | Echoes mismatched `final_eval` |
| Paper-feeding P1/P2 narratives that cite the logged scarcity A−B % gap (see ablation M4-gate section; DECISIONS B′/M4 notes) | Numbers retracted below |

### Not contaminated by *cross-regime mode mismatch*

| Output | Why |
|---|---|
| A-vs-C at ∞ (and all A-vs-C cells) | Both `signaling=False` → both greedy |
| M3 phase diagram / `M3_REPORT.md` (Regime B only) | No cross-regime compare; all B cells share stochastic eval (absolute levels still stochastic-mode) |
| M2 A vs C classic (`M2_REPORT.md`) | Both non-signaling |
| `v11_ablation.md` matched-det cost tables + shuffle rubric | Already forced greedy |
| `v11_signal_content.md` MI/decoder tables | Within-B content metrics, not A/B cost |
| D2–D6 M3 diagnostics (`analysis/DIAGNOSTICS.md`) | Pre-v11 / B-only or non-cost |

## Step 2 — Corrected table (old mismatched → new matched-det)

Old = mean±CI95 over 10 seeds from logged `final_eval` (`greedy=not signaling`). New = same checkpoints, `greedy=True` for every regime, same seed offset (`seed+10_000`), 20 episodes.

### A vs B (all topologies / demands / rationing / caps in matrix)

| Topo | Cap | Rationing | Demand | A old | B old | A−B old (%) | A new | B new | A−B new (%) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| serial | ∞ | proportional | ar1 | 29.4±1.4 | 32.2±1.3 | -2.8 (-9.4%) | 29.4±1.4 | 28.5±1.7 | 0.9 (3.0%) |
| serial | ∞ | proportional | regime_switch | 49.1±2.1 | 51.4±2.0 | -2.2 (-4.5%) | 49.1±2.1 | 44.1±2.4 | 5.0 (10.2%) |
| serial | 1.2μ | proportional | ar1 | 663.5±155.2 | 313.7±60.6 | 349.8 (52.7%) | 663.5±155.2 | 509.5±135.1 | 154.0 (23.2%) |
| serial | 1.2μ | proportional | regime_switch | 535.5±160.9 | 333.1±73.7 | 202.4 (37.8%) | 535.5±160.9 | 474.6±120.5 | 60.8 (11.4%) |
| serial | 1.0μ | proportional | ar1 | 627.7±52.7 | 351.3±30.6 | 276.4 (44.0%) | 627.7±52.7 | 577.9±126.1 | 49.9 (7.9%) |
| serial | 1.0μ | proportional | regime_switch | 627.9±153.2 | 401.9±48.0 | 225.9 (36.0%) | 627.9±153.2 | 576.2±132.1 | 51.6 (8.2%) |
| serial | 0.8μ | proportional | ar1 | 528.3±116.5 | 382.9±19.5 | 145.4 (27.5%) | 528.3±116.5 | 478.5±114.9 | 49.8 (9.4%) |
| serial | 0.8μ | proportional | regime_switch | 573.6±123.2 | 461.3±40.9 | 112.3 (19.6%) | 573.6±123.2 | 619.0±142.3 | -45.3 (-7.9%) |
| y | ∞ | proportional | ar1 | 95.2±2.7 | 99.9±4.3 | -4.7 (-4.9%) | 95.2±2.7 | 88.6±4.3 | 6.6 (6.9%) |
| y | ∞ | proportional | regime_switch | 104.6±2.8 | 104.4±2.8 | 0.2 (0.1%) | 104.6±2.8 | 108.4±5.7 | -3.8 (-3.6%) |
| y | 1.2μ | honesty_weighted | ar1 | 1337.2±122.4 | 746.2±51.4 | 591.0 (44.2%) | 1337.2±122.4 | 1232.9±93.8 | 104.3 (7.8%) |
| y | 1.2μ | honesty_weighted | regime_switch | 1292.3±101.5 | 645.3±38.8 | 647.0 (50.1%) | 1292.3±101.5 | 1249.0±120.3 | 43.3 (3.4%) |
| y | 1.2μ | proportional | ar1 | 1337.2±122.4 | 868.4±74.6 | 468.7 (35.1%) | 1337.2±122.4 | 1331.1±66.1 | 6.1 (0.5%) |
| y | 1.2μ | proportional | regime_switch | 1292.3±101.5 | 796.1±46.0 | 496.1 (38.4%) | 1292.3±101.5 | 1250.7±99.8 | 41.6 (3.2%) |
| y | 1.2μ | uniform | ar1 | 1000.0±180.8 | 543.1±61.1 | 456.9 (45.7%) | 1000.0±180.8 | 987.6±154.6 | 12.4 (1.2%) |
| y | 1.2μ | uniform | regime_switch | 995.8±143.6 | 460.2±41.0 | 535.6 (53.8%) | 995.8±143.6 | 815.3±146.4 | 180.5 (18.1%) |
| y | 1.0μ | honesty_weighted | ar1 | 1458.2±80.3 | 847.7±48.2 | 610.5 (41.9%) | 1458.2±80.3 | 1350.8±92.7 | 107.3 (7.4%) |
| y | 1.0μ | honesty_weighted | regime_switch | 1404.8±63.5 | 749.6±28.0 | 655.2 (46.6%) | 1404.8±63.5 | 1228.0±106.3 | 176.7 (12.6%) |
| y | 1.0μ | proportional | ar1 | 1458.2±80.3 | 940.9±33.5 | 517.3 (35.5%) | 1458.2±80.3 | 1457.3±95.5 | 0.9 (0.1%) |
| y | 1.0μ | proportional | regime_switch | 1404.8±63.5 | 862.1±16.9 | 542.7 (38.6%) | 1404.8±63.5 | 1323.9±85.8 | 80.9 (5.8%) |
| y | 1.0μ | uniform | ar1 | 995.0±161.3 | 633.8±57.5 | 361.2 (36.3%) | 995.0±161.3 | 938.3±175.6 | 56.7 (5.7%) |
| y | 1.0μ | uniform | regime_switch | 906.9±120.8 | 568.6±25.7 | 338.3 (37.3%) | 906.9±120.8 | 926.8±113.6 | -19.9 (-2.2%) |
| y | 0.8μ | honesty_weighted | ar1 | 1352.3±39.6 | 910.8±36.2 | 441.6 (32.7%) | 1352.3±39.6 | 1300.1±104.7 | 52.3 (3.9%) |
| y | 0.8μ | honesty_weighted | regime_switch | 1303.2±71.3 | 851.0±24.6 | 452.1 (34.7%) | 1303.2±71.3 | 1201.4±99.5 | 101.7 (7.8%) |
| y | 0.8μ | proportional | ar1 | 1352.3±39.6 | 958.0±27.3 | 394.3 (29.2%) | 1352.3±39.6 | 1393.1±95.1 | -40.8 (-3.0%) |
| y | 0.8μ | proportional | regime_switch | 1303.2±71.3 | 892.9±12.7 | 410.3 (31.5%) | 1303.2±71.3 | 1349.1±121.9 | -45.9 (-3.5%) |
| y | 0.8μ | uniform | ar1 | 895.9±151.6 | 737.8±49.6 | 158.1 (17.6%) | 895.9±151.6 | 992.0±127.0 | -96.1 (-10.7%) |
| y | 0.8μ | uniform | regime_switch | 887.8±141.9 | 692.3±25.6 | 195.5 (22.0%) | 887.8±141.9 | 840.6±99.5 | 47.2 (5.3%) |

### C vs A (should be unchanged — already matched-det)

| Topo | Cap | Rationing | Demand | A old | C old | A−C old (%) | A new | C new | A−C new (%) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| serial | ∞ | proportional | ar1 | 29.4±1.4 | 63.8±7.2 | -34.4 (-116.8%) | 29.4±1.4 | 63.8±7.2 | -34.4 (-116.8%) |
| serial | ∞ | proportional | regime_switch | 49.1±2.1 | 92.6±8.1 | -43.5 (-88.5%) | 49.1±2.1 | 92.6±8.1 | -43.5 (-88.5%) |
| serial | 1.2μ | proportional | ar1 | 663.5±155.2 | 99.3±15.4 | 564.2 (85.0%) | 663.5±155.2 | 99.3±15.4 | 564.2 (85.0%) |
| serial | 1.2μ | proportional | regime_switch | 535.5±160.9 | 154.8±40.9 | 380.6 (71.1%) | 535.5±160.9 | 154.8±40.9 | 380.6 (71.1%) |
| serial | 1.0μ | proportional | ar1 | 627.7±52.7 | 111.0±12.7 | 516.7 (82.3%) | 627.7±52.7 | 111.0±12.7 | 516.7 (82.3%) |
| serial | 1.0μ | proportional | regime_switch | 627.9±153.2 | 178.8±48.9 | 449.1 (71.5%) | 627.9±153.2 | 178.8±48.9 | 449.1 (71.5%) |
| serial | 0.8μ | proportional | ar1 | 528.3±116.5 | 131.1±25.6 | 397.2 (75.2%) | 528.3±116.5 | 131.1±25.6 | 397.2 (75.2%) |
| serial | 0.8μ | proportional | regime_switch | 573.6±123.2 | 232.5±77.7 | 341.1 (59.5%) | 573.6±123.2 | 232.5±77.7 | 341.1 (59.5%) |
| y | ∞ | proportional | ar1 | 95.2±2.7 | 190.0±11.2 | -94.9 (-99.7%) | 95.2±2.7 | 190.0±11.2 | -94.9 (-99.7%) |
| y | ∞ | proportional | regime_switch | 104.6±2.8 | 192.8±13.1 | -88.2 (-84.4%) | 104.6±2.8 | 192.8±13.1 | -88.2 (-84.4%) |
| y | 1.2μ | honesty_weighted | ar1 | 1337.2±122.4 | 349.0±55.2 | 988.2 (73.9%) | 1337.2±122.4 | 349.0±55.2 | 988.2 (73.9%) |
| y | 1.2μ | honesty_weighted | regime_switch | 1292.3±101.5 | 338.2±24.4 | 954.0 (73.8%) | 1292.3±101.5 | 338.2±24.4 | 954.0 (73.8%) |
| y | 1.2μ | proportional | ar1 | 1337.2±122.4 | 349.0±55.2 | 988.2 (73.9%) | 1337.2±122.4 | 349.0±55.2 | 988.2 (73.9%) |
| y | 1.2μ | proportional | regime_switch | 1292.3±101.5 | 338.2±24.4 | 954.0 (73.8%) | 1292.3±101.5 | 338.2±24.4 | 954.0 (73.8%) |
| y | 1.2μ | uniform | ar1 | 1000.0±180.8 | 349.9±54.4 | 650.1 (65.0%) | 1000.0±180.8 | 349.9±54.4 | 650.1 (65.0%) |
| y | 1.2μ | uniform | regime_switch | 995.8±143.6 | 339.6±23.2 | 656.2 (65.9%) | 995.8±143.6 | 339.6±23.2 | 656.2 (65.9%) |
| y | 1.0μ | honesty_weighted | ar1 | 1458.2±80.3 | 467.0±57.9 | 991.2 (68.0%) | 1458.2±80.3 | 467.0±57.9 | 991.2 (68.0%) |
| y | 1.0μ | honesty_weighted | regime_switch | 1404.8±63.5 | 431.7±25.9 | 973.1 (69.3%) | 1404.8±63.5 | 431.7±25.9 | 973.1 (69.3%) |
| y | 1.0μ | proportional | ar1 | 1458.2±80.3 | 467.0±57.9 | 991.2 (68.0%) | 1458.2±80.3 | 467.0±57.9 | 991.2 (68.0%) |
| y | 1.0μ | proportional | regime_switch | 1404.8±63.5 | 431.7±25.9 | 973.1 (69.3%) | 1404.8±63.5 | 431.7±25.9 | 973.1 (69.3%) |
| y | 1.0μ | uniform | ar1 | 995.0±161.3 | 468.3±56.8 | 526.7 (52.9%) | 995.0±161.3 | 468.3±56.8 | 526.7 (52.9%) |
| y | 1.0μ | uniform | regime_switch | 906.9±120.8 | 432.6±25.9 | 474.3 (52.3%) | 906.9±120.8 | 432.6±25.9 | 474.3 (52.3%) |
| y | 0.8μ | honesty_weighted | ar1 | 1352.3±39.6 | 591.5±51.6 | 760.9 (56.3%) | 1352.3±39.6 | 591.5±51.6 | 760.9 (56.3%) |
| y | 0.8μ | honesty_weighted | regime_switch | 1303.2±71.3 | 553.8±31.6 | 749.4 (57.5%) | 1303.2±71.3 | 553.8±31.6 | 749.4 (57.5%) |
| y | 0.8μ | proportional | ar1 | 1352.3±39.6 | 591.5±51.6 | 760.9 (56.3%) | 1352.3±39.6 | 591.5±51.6 | 760.9 (56.3%) |
| y | 0.8μ | proportional | regime_switch | 1303.2±71.3 | 553.8±31.6 | 749.4 (57.5%) | 1303.2±71.3 | 553.8±31.6 | 749.4 (57.5%) |
| y | 0.8μ | uniform | ar1 | 895.9±151.6 | 591.5±51.6 | 304.4 (34.0%) | 895.9±151.6 | 591.5±51.6 | 304.4 (34.0%) |
| y | 0.8μ | uniform | regime_switch | 887.8±141.9 | 554.4±31.5 | 333.4 (37.6%) | 887.8±141.9 | 554.4±31.5 | 333.4 (37.6%) |

### B vs C (mismatched old → matched-det new)

| Topo | Cap | Rationing | Demand | B old | C old | C−B old (%) | B new | C new | C−B new (%) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| serial | ∞ | proportional | ar1 | 32.2±1.3 | 63.8±7.2 | 31.6 (49.6%) | 28.5±1.7 | 63.8±7.2 | 35.3 (55.3%) |
| serial | ∞ | proportional | regime_switch | 51.4±2.0 | 92.6±8.1 | 41.2 (44.5%) | 44.1±2.4 | 92.6±8.1 | 48.5 (52.4%) |
| serial | 1.2μ | proportional | ar1 | 313.7±60.6 | 99.3±15.4 | -214.4 (-216.0%) | 509.5±135.1 | 99.3±15.4 | -410.2 (-413.3%) |
| serial | 1.2μ | proportional | regime_switch | 333.1±73.7 | 154.8±40.9 | -178.3 (-115.1%) | 474.6±120.5 | 154.8±40.9 | -319.8 (-206.6%) |
| serial | 1.0μ | proportional | ar1 | 351.3±30.6 | 111.0±12.7 | -240.2 (-216.4%) | 577.9±126.1 | 111.0±12.7 | -466.8 (-420.4%) |
| serial | 1.0μ | proportional | regime_switch | 401.9±48.0 | 178.8±48.9 | -223.1 (-124.8%) | 576.2±132.1 | 178.8±48.9 | -397.4 (-222.3%) |
| serial | 0.8μ | proportional | ar1 | 382.9±19.5 | 131.1±25.6 | -251.8 (-192.2%) | 478.5±114.9 | 131.1±25.6 | -347.4 (-265.1%) |
| serial | 0.8μ | proportional | regime_switch | 461.3±40.9 | 232.5±77.7 | -228.8 (-98.4%) | 619.0±142.3 | 232.5±77.7 | -386.4 (-166.2%) |
| y | ∞ | proportional | ar1 | 99.9±4.3 | 190.0±11.2 | 90.2 (47.4%) | 88.6±4.3 | 190.0±11.2 | 101.5 (53.4%) |
| y | ∞ | proportional | regime_switch | 104.4±2.8 | 192.8±13.1 | 88.4 (45.8%) | 108.4±5.7 | 192.8±13.1 | 84.5 (43.8%) |
| y | 1.2μ | honesty_weighted | ar1 | 746.2±51.4 | 349.0±55.2 | -397.1 (-113.8%) | 1232.9±93.8 | 349.0±55.2 | -883.9 (-253.2%) |
| y | 1.2μ | honesty_weighted | regime_switch | 645.3±38.8 | 338.2±24.4 | -307.0 (-90.8%) | 1249.0±120.3 | 338.2±24.4 | -910.7 (-269.3%) |
| y | 1.2μ | proportional | ar1 | 868.4±74.6 | 349.0±55.2 | -519.4 (-148.8%) | 1331.1±66.1 | 349.0±55.2 | -982.0 (-281.4%) |
| y | 1.2μ | proportional | regime_switch | 796.1±46.0 | 338.2±24.4 | -457.9 (-135.4%) | 1250.7±99.8 | 338.2±24.4 | -912.5 (-269.8%) |
| y | 1.2μ | uniform | ar1 | 543.1±61.1 | 349.9±54.4 | -193.2 (-55.2%) | 987.6±154.6 | 349.9±54.4 | -637.7 (-182.3%) |
| y | 1.2μ | uniform | regime_switch | 460.2±41.0 | 339.6±23.2 | -120.6 (-35.5%) | 815.3±146.4 | 339.6±23.2 | -475.7 (-140.1%) |
| y | 1.0μ | honesty_weighted | ar1 | 847.7±48.2 | 467.0±57.9 | -380.7 (-81.5%) | 1350.8±92.7 | 467.0±57.9 | -883.9 (-189.3%) |
| y | 1.0μ | honesty_weighted | regime_switch | 749.6±28.0 | 431.7±25.9 | -317.9 (-73.6%) | 1228.0±106.3 | 431.7±25.9 | -796.3 (-184.5%) |
| y | 1.0μ | proportional | ar1 | 940.9±33.5 | 467.0±57.9 | -474.0 (-101.5%) | 1457.3±95.5 | 467.0±57.9 | -990.3 (-212.1%) |
| y | 1.0μ | proportional | regime_switch | 862.1±16.9 | 431.7±25.9 | -430.3 (-99.7%) | 1323.9±85.8 | 431.7±25.9 | -892.2 (-206.7%) |
| y | 1.0μ | uniform | ar1 | 633.8±57.5 | 468.3±56.8 | -165.5 (-35.3%) | 938.3±175.6 | 468.3±56.8 | -470.0 (-100.4%) |
| y | 1.0μ | uniform | regime_switch | 568.6±25.7 | 432.6±25.9 | -136.0 (-31.4%) | 926.8±113.6 | 432.6±25.9 | -494.2 (-114.2%) |
| y | 0.8μ | honesty_weighted | ar1 | 910.8±36.2 | 591.5±51.6 | -319.3 (-54.0%) | 1300.1±104.7 | 591.5±51.6 | -708.6 (-119.8%) |
| y | 0.8μ | honesty_weighted | regime_switch | 851.0±24.6 | 553.8±31.6 | -297.3 (-53.7%) | 1201.4±99.5 | 553.8±31.6 | -647.7 (-117.0%) |
| y | 0.8μ | proportional | ar1 | 958.0±27.3 | 591.5±51.6 | -366.6 (-62.0%) | 1393.1±95.1 | 591.5±51.6 | -801.7 (-135.5%) |
| y | 0.8μ | proportional | regime_switch | 892.9±12.7 | 553.8±31.6 | -339.1 (-61.2%) | 1349.1±121.9 | 553.8±31.6 | -795.3 (-143.6%) |
| y | 0.8μ | uniform | ar1 | 737.8±49.6 | 591.5±51.6 | -146.3 (-24.7%) | 992.0±127.0 | 591.5±51.6 | -400.5 (-67.7%) |
| y | 0.8μ | uniform | regime_switch | 692.3±25.6 | 554.4±31.5 | -137.9 (-24.9%) | 840.6±99.5 | 554.4±31.5 | -286.2 (-51.6%) |

## Takeaways

1. **RETRACT** the logged 28–53% (and sibling) A−B scarcity cost gaps from `final_eval` / index aggregates — they mix greedy A with stochastic B.
2. Under matched deterministic eval, A−B scarcity gaps shrink dramatically (typically into seed CIs); see corrected A−B table.
3. A−C gaps are unchanged (already matched). B−C gaps move because only B's logged costs were stochastic.
4. Code fix (out of scope here): evaluate with an explicit mode, never `greedy=not signaling`.

