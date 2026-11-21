# Omission audit report

Dataset: seed 7, redundancy 1, 240 train / 120 test rows, budget k = 6.

| Explainer | Omission (all used) | Main | Interaction-only | Subgroup-only | Commission | Copy slots | Noise slots |
|---|---|---|---|---|---|---|---|
| permutation | 33.3% | 0.0% | 50.0% | 100.0% | 33.3% | 1 | 1 |
| loco | 33.3% | 0.0% | 50.0% | 100.0% | 33.3% | 1 | 1 |
| shapley | 33.3% | 0.0% | 50.0% | 100.0% | 33.3% | 1 | 1 |
| surrogate | 33.3% | 0.0% | 100.0% | 0.0% | 33.3% | 1 | 1 |
| random | 50.0% | 33.3% | 50.0% | 100.0% | 50.0% | 1 | 2 |

## Ranking: permutation

| Rank | Feature | Usage | In top-k |
|---|---|---|---|
| 1 | main_a | main | yes |
| 2 | main_c | main | yes |
| 3 | copy_1 | redundant_copy | yes |
| 4 | int_a | interaction_only | yes |
| 5 | main_b | main | yes |
| 6 | noise_a | unused | yes |
| 7 | noise_b | unused | no |
| 8 | noise_c | unused | no |
| 9 | noise_d | unused | no |
| 10 | noise_e | unused | no |
| 11 | sub_feat | subgroup_only | no |
| 12 | int_b | interaction_only | no |

Omitted used features: int_b, sub_feat

