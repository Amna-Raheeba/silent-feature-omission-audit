# Changelog

## 0.1.0 — 2026-11-21

- Initial release: omission audit of post-hoc explainers against known data-generating ground truth.
- Structural DGP with main / interaction-only / subgroup-only / unused features and planted redundant copies (redundancy levels 0–2).
- Stdlib CART decision tree and ridge surrogate; explainers: permutation, LOCO, sampled Shapley, surrogate coefficients, random baseline.
- Omission profile per usage type with commission split (copy vs noise slots) and source displaced-vs-omitted tracking.
- Headline evaluation over 5 seeds × 3 redundancy levels; interaction-only omission at redundancy 2 reaches 90.0% (permutation, Shapley) and 100.0% (surrogate) vs a 70.0% random floor, while LOCO holds at 20.0% but omits the subgroup-only feature at 100.0%.
- Streamlit dashboard, CLI (demo / audit / evaluate), reproducible results with a headline-drift guard.
