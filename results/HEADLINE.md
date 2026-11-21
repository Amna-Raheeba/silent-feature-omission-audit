# Headline results

An omission audit of post-hoc explainers against known data-generating ground truth: datasets are generated from a structural process that uses 3 main-effect features, 2 interaction-only features and 1 subgroup-only feature, plus 5 noise features and up to 2 planted redundant copies of a main feature. Each explainer's top-6 (the number of truly-used features) is scored for the used features it *fails to mention*, per usage type, averaged over 5 seeds.

- Interaction-only omission at redundancy 0: permutation 20.0%, LOCO 20.0%, sampled Shapley 10.0%, surrogate coefficients 50.0% -- against a random-ranking floor of 60.0%.
- At redundancy 2 the same rates are: permutation 90.0%, LOCO 20.0%, Shapley 90.0%, surrogate 100.0%, random 70.0%.
- Overall omission at redundancy 2: permutation 36.7%, LOCO 56.7%, Shapley 40.0%, surrogate 33.3%, random 60.0%.

Caveat: all data is synthetic and seeded, with feature usage planted by construction; the audit measures explainer behaviour under known ground truth, not faithfulness on real datasets.
