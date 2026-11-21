# Evaluation summary

Datasets: 5 seeds x redundancy levels [0, 1, 2], n_train=1000, n_test=500, generated from the known structural process. Budget k = 6 (the number of truly-used features) at every level; planted copies compete for the same budget.

## Omission rate by usage type (all levels)

| Explainer | Redundancy | All used | Main | Interaction-only | Subgroup-only | Commission |
|---|---|---|---|---|---|---|
| permutation | 0 | 10.0% | 0.0% | 20.0% | 20.0% | 10.0% |
| loco | 0 | 26.7% | 6.7% | 20.0% | 100.0% | 26.7% |
| shapley | 0 | 13.3% | 13.3% | 10.0% | 20.0% | 13.3% |
| surrogate | 0 | 16.7% | 0.0% | 50.0% | 0.0% | 16.7% |
| random | 0 | 53.3% | 40.0% | 60.0% | 80.0% | 53.3% |
| permutation | 1 | 20.0% | 0.0% | 30.0% | 60.0% | 20.0% |
| loco | 1 | 36.7% | 40.0% | 10.0% | 80.0% | 36.7% |
| shapley | 1 | 23.3% | 0.0% | 50.0% | 40.0% | 23.3% |
| surrogate | 1 | 26.7% | 0.0% | 80.0% | 0.0% | 26.7% |
| random | 1 | 60.0% | 46.7% | 70.0% | 80.0% | 60.0% |
| permutation | 2 | 36.7% | 0.0% | 90.0% | 40.0% | 36.7% |
| loco | 2 | 56.7% | 66.7% | 20.0% | 100.0% | 56.7% |
| shapley | 2 | 40.0% | 6.7% | 90.0% | 40.0% | 40.0% |
| surrogate | 2 | 33.3% | 0.0% | 100.0% | 0.0% | 33.3% |
| random | 2 | 60.0% | 46.7% | 70.0% | 80.0% | 60.0% |

## Interaction-only omission growth (redundancy 0 -> 2)

- permutation: 20.0% -> 90.0% (growth +70.0 points)
- loco: 20.0% -> 20.0% (growth +0.0 points)
- shapley: 10.0% -> 90.0% (growth +80.0 points)
- surrogate: 50.0% -> 100.0% (growth +50.0 points)
- random: 60.0% -> 70.0% (growth +10.0 points)

Caveat: all data is synthetic and seeded, generated from the audit's own structural process; the numbers measure explainer behaviour under known ground truth, not performance on real datasets. See README section 8.
