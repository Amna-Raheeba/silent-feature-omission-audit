# Silent Feature Omission Audit

An omission audit of post-hoc explainers against known data-generating ground truth: it scores each explainer on the features it *fails to mention* — features the data-generating process actually used — broken down by how each feature is used, and swept over planted redundancy.

## 1. What it is

Explanation methods are usually scored on the features they rank highly: are the top features faithful, stable, plausible? This audit scores the other side of the ranking. Data is generated from a known structural process, so the set of truly-used features is known exactly. Each explainer gets a top-k budget (k = 6, the number of truly-used features), and the audit reports which used features fall outside it — the **omission profile** — separately from commission errors (top-k slots spent on features the process never used).

Four stdlib explainers are audited — permutation importance, leave-one-covariate-out (LOCO), sampled Shapley values, and linear-surrogate coefficients — against a random-ranking floor. The explained model is a small CART decision tree, chosen because it *can* represent every structure in the process, so omissions are the explainers' fault, not the model's.

## 2. The novel mechanism

The data-generating process uses each feature in a specific way, and omission is reported **per usage type**:

| Usage type | Features | How the process uses them |
|---|---|---|
| Main effect | `main_a`, `main_b`, `main_c` | Own linear term in the latent score |
| Interaction-only | `int_a`, `int_b` | Only inside the product `int_a * int_b`; no main effect |
| Subgroup-only | `sub_feat` | Effect switched on only for a latent subgroup (35% of rows; the subgroup flag is not a column) |
| Redundant copy | `copy_1`, `copy_2` | Not used; noisy copies of `main_a`, planted at redundancy levels 1–2 |
| Unused | `noise_a` … `noise_e` | Pure noise; naming one is a commission error |

Three design choices make the measurement clean:

1. **Ground truth by construction.** Usage is planted, not annotated, so omission per usage type is exact — including for interaction-only and subgroup-only features, where omission is expected to concentrate and where no annotated corpus gives per-type ground truth.
2. **Redundancy is controlled, not incidental.** Planted copies of `main_a` compete for the *same* fixed top-6 budget. The audit measures how each explainer's omission profile degrades as redundancy grows from 0 to 2 copies — the exact condition under which removal-based faithfulness metrics are known to break down. For the copied source it also distinguishes *displaced* (a copy is named instead of the source) from *omitted* (neither is named; the signal is lost).
3. **Omission is separated from commission.** An explainer can fill its budget with copies (signal present, wrong name), with noise (pure error), or with the right features. The profile reports copy slots and noise slots separately.

## 3. Prior art and the differentiator

ROAR and Recursive ROAR (Hooker et al.) measure performance degradation after removing an explainer's top-ranked features and retraining; Hooker et al. themselves note ROAR cannot identify faithfulness under dataset redundancy. Neither reports a per-feature omission profile against data-generating ground truth, and neither isolates interaction-only and subgroup-only usage. This audit inverts the question — not "does performance drop when the top features are removed?" but "which truly-used features never made the explanation at all, and which usage types are they?" — and makes redundancy a swept experimental variable with planted, exactly-known copies instead of an uncontrolled property of a benchmark dataset.

## 4. Measured headline results

Datasets: 5 seeds × redundancy levels 0/1/2, 1,000 train / 500 test rows per dataset, budget k = 6 at every level. Rates are seed-averaged omission rates (fraction of used features of that type outside the explainer's top-6). Full tables in `results/evaluation_summary.md`.

**Interaction-only omission — the headline.** Without redundancy, explainers mostly find the interaction pair (permutation 20.0%, LOCO 20.0%, Shapley 10.0%; surrogate coefficients 50.0%, against a random floor of 60.0%). Plant two redundant copies and the picture inverts:

| Explainer | Interaction omission, redundancy 0 | Redundancy 2 | Growth |
|---|---|---|---|
| Permutation | 20.0% | 90.0% | +70.0 pts |
| Sampled Shapley | 10.0% | 90.0% | +80.0 pts |
| Surrogate coefficients | 50.0% | 100.0% | +50.0 pts |
| LOCO | 20.0% | 20.0% | +0.0 pts |
| Random ranking | 60.0% | 70.0% | +10.0 pts |

At redundancy 2, permutation, Shapley and the surrogate omit the interaction pair **more often than a random ranking does**: the copies split the main feature's credit three ways, the budget fills with the duplicated signal, and the weakest-marginal-signal features — the interaction pair — are pushed out.

**Each explainer has a different blind spot.** LOCO is the only method robust on interaction (removing one of the pair destroys the product term, so the drop is real), but it omits the subgroup-only feature at 100% at redundancy 0 and 2 (its effect is diluted across the full sample, so retraining without it costs little), and under redundancy its *overall* omission is the worst of all explainers at 56.7% — removing `main_a` costs nothing when a copy remains, so LOCO drops the source itself (main-effect omission 66.7% at redundancy 2). The linear surrogate never omits the subgroup feature (0.0% at every level — it sees the average effect) but is structurally blind to the interaction, reaching 100.0% omission at redundancy 2.

## 5. Quick start

```bash
pip install -r requirements.txt
python -m omission_audit demo          # audit one small dataset
python -m omission_audit evaluate --outdir results
python examples/run_audit.py           # audit the bundled demo dataset
streamlit run app.py                   # dashboard
pytest -q
```

## 6. How it works

1. `generate` draws rows from the structural process (Section 2) with a seeded RNG; labels are the sign of the latent score.
2. `tree` fits a CART classifier (Gini, quantile thresholds, deterministic tie-breaking) and a standardised ridge surrogate.
3. `explainers` ranks the features four ways plus a random baseline, all on the same trained tree and data.
4. `audit` compares each ranking's top-6 with the ground-truth usage map and builds the omission profile: per-type omission, commission split into copy/noise slots, and source displaced-vs-omitted for the copied feature.
5. `evaluate` repeats this over seeds × redundancy levels and writes `results/`.

## 7. Repository layout

- `omission_audit/` — the package: `models` (ground-truth spec), `generate`, `tree`, `explainers`, `audit`, `evaluate`, `report`, `cli`
- `data/` — the DGP spec and one stored dataset (`dataset_r1.json`)
- `examples/` — a small demo dataset, a runnable audit script, and its real output (`sample_report.md`)
- `notebooks/` — demo run and a reduced evaluation smoke run, as runnable scripts
- `scripts/` — `build_cases.py` (regenerates `data/` + demo dataset) and `reproduce_results.py` (headline-drift guard)
- `results/` — `evaluation.json`, `evaluation.csv`, `evaluation_summary.md`, `HEADLINE.md`
- `tests/` — the pytest suite
- `app.py` — Streamlit dashboard (explorer / by-usage-type / evaluation)

## 8. Honest limits

- All data is synthetic and seeded, generated from the audit's own structural process. The numbers measure explainer behaviour under *known* ground truth; they are not faithfulness measurements on real datasets, where the used-feature set is exactly what is unknown.
- The explained model is one small decision tree and the outcome is one binary label from one process family (Gaussian features, one product interaction, one latent subgroup at 35%). Other model classes and process shapes may shift the rates; the per-type *pattern* (omission concentrating in interaction/subgroup usage, worsening under redundancy) is the claim, not the exact percentages.
- Sampled Shapley uses 32 permutations with marginal (background) replacement and a 120-row eval subset — an estimator of Shapley values, with the usual Monte-Carlo and masking caveats. LOCO and permutation inherit the known weakness that a feature whose signal is fully duplicated looks unimportant; the audit measures that weakness rather than fixing it.
- Omission is judged at a single budget, k = number of used features. An explainer that ranks a used feature 7th is scored identically to one that ranks it last; rank-sensitive omission curves are future work.

## 9. Reproducing the results

```bash
python -m omission_audit evaluate --outdir results   # canonical run (~10 s)
python scripts/reproduce_results.py                  # drift guard vs stored results
python scripts/build_cases.py                        # regenerate stored datasets
```

## 10. License

MIT — see `LICENSE`. Author: Amna Raheeba.
