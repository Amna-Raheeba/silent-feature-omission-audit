"""Headline evaluation: omission profiles swept over redundancy.

For each redundancy level (0, 1, 2 planted copies of ``main_a``) and
each seed, a dataset is generated from the known structural process,
a decision tree is trained, every explainer ranks the features, and
the audit scores each ranking against the ground-truth usage map.
Rates are averaged over seeds.

The headline numbers are the interaction-only omission rate per
explainer and its growth from redundancy 0 to redundancy 2: planted
copies compete for the same top-k budget, and the features with the
weakest marginal signal -- the interaction pair -- are expected to be
pushed out first.

Everything is deterministic: same config, same numbers.
"""

import csv
import json
import os

from .audit import audit_dataset
from .explainers import EXPLAINER_NAMES, run_explainers
from .generate import generate_dataset

DEFAULT_CONFIG = {
    "seeds": (101, 202, 303, 404, 505),
    "redundancy_levels": (0, 1, 2),
    "n_train": 1000,
    "n_test": 500,
}

_USAGE_ROWS = ("main", "interaction_only", "subgroup_only", "all_used")


def _mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def run_evaluation(config=None):
    """Run the full sweep; returns the evaluation dict."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    collected = {}
    for redundancy in cfg["redundancy_levels"]:
        for seed in cfg["seeds"]:
            dataset = generate_dataset(
                seed=seed,
                redundancy=redundancy,
                n_train=cfg["n_train"],
                n_test=cfg["n_test"],
            )
            profiles = audit_dataset(dataset, run_explainers(dataset))
            for name in EXPLAINER_NAMES:
                key = (redundancy, name)
                collected.setdefault(key, []).append(profiles[name])

    rows = []
    cells = {}
    for redundancy in cfg["redundancy_levels"]:
        for name in EXPLAINER_NAMES:
            profiles = collected[(redundancy, name)]
            cell = {
                "omission_rate": _mean([p["omission_rate"] for p in profiles]),
                "commission_rate": _mean([p["commission_rate"] for p in profiles]),
                "copy_slots": _mean([p["copy_slots"] for p in profiles]),
                "noise_slots": _mean([p["noise_slots"] for p in profiles]),
                "source_omitted_rate": _mean(
                    [1.0 if p["source_omitted"] else 0.0 for p in profiles]
                ),
                "source_displaced_rate": _mean(
                    [1.0 if p["source_displaced"] else 0.0 for p in profiles]
                ),
                "per_type": dict(
                    (
                        usage_type,
                        _mean(
                            [
                                p["per_type"][usage_type]["omission_rate"]
                                for p in profiles
                            ]
                        ),
                    )
                    for usage_type in ("main", "interaction_only", "subgroup_only")
                ),
            }
            cells[(redundancy, name)] = cell
            for usage_type in _USAGE_ROWS:
                rate = (
                    cell["omission_rate"]
                    if usage_type == "all_used"
                    else cell["per_type"][usage_type]
                )
                rows.append({
                    "redundancy": redundancy,
                    "explainer": name,
                    "usage_type": usage_type,
                    "omission_rate": round(rate, 6),
                })

    aggregates = _aggregates(cells, cfg)
    return {
        "config": dict(
            (key, list(value) if isinstance(value, tuple) else value)
            for key, value in cfg.items()
        ),
        "rows": rows,
        "cells": dict(
            ("%d|%s" % (redundancy, name), cells[(redundancy, name)])
            for redundancy in cfg["redundancy_levels"]
            for name in EXPLAINER_NAMES
        ),
        "aggregates": aggregates,
    }


def _aggregates(cells, cfg):
    agg = {}
    levels = cfg["redundancy_levels"]
    for name in EXPLAINER_NAMES:
        for redundancy in levels:
            cell = cells[(redundancy, name)]
            agg["%s_omission_r%d" % (name, redundancy)] = round(
                cell["omission_rate"], 6
            )
            agg["%s_interaction_omission_r%d" % (name, redundancy)] = round(
                cell["per_type"]["interaction_only"], 6
            )
            agg["%s_subgroup_omission_r%d" % (name, redundancy)] = round(
                cell["per_type"]["subgroup_only"], 6
            )
            agg["%s_commission_r%d" % (name, redundancy)] = round(
                cell["commission_rate"], 6
            )
        agg["%s_interaction_growth" % name] = round(
            cells[(levels[-1], name)]["per_type"]["interaction_only"]
            - cells[(levels[0], name)]["per_type"]["interaction_only"],
            6,
        )
    return agg


def write_results(evaluation, outdir):
    """Write evaluation.json/.csv, evaluation_summary.md, HEADLINE.md."""
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "evaluation.json"), "w") as handle:
        json.dump(evaluation, handle, indent=2)
    with open(os.path.join(outdir, "evaluation.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("redundancy", "explainer", "usage_type", "omission_rate"),
        )
        writer.writeheader()
        for row in evaluation["rows"]:
            writer.writerow(row)
    with open(os.path.join(outdir, "evaluation_summary.md"), "w") as handle:
        handle.write(summary_markdown(evaluation))
    with open(os.path.join(outdir, "HEADLINE.md"), "w") as handle:
        handle.write(headline_markdown(evaluation))


def _pct(value):
    return "%.1f%%" % (100.0 * value)


def summary_markdown(evaluation):
    agg = evaluation["aggregates"]
    cfg = evaluation["config"]
    lines = [
        "# Evaluation summary",
        "",
        "Datasets: %d seeds x redundancy levels %s, n_train=%d, "
        "n_test=%d, generated from the known structural process. "
        "Budget k = 6 (the number of truly-used features) at every "
        "level; planted copies compete for the same budget."
        % (
            len(cfg["seeds"]),
            cfg["redundancy_levels"],
            cfg["n_train"],
            cfg["n_test"],
        ),
        "",
        "## Omission rate by usage type (all levels)",
        "",
        "| Explainer | Redundancy | All used | Main | Interaction-only | Subgroup-only | Commission |",
        "|---|---|---|---|---|---|---|",
    ]
    cells = evaluation["cells"]
    for redundancy in cfg["redundancy_levels"]:
        for name in EXPLAINER_NAMES:
            cell = cells["%d|%s" % (redundancy, name)]
            lines.append(
                "| %s | %d | %s | %s | %s | %s | %s |"
                % (
                    name,
                    redundancy,
                    _pct(cell["omission_rate"]),
                    _pct(cell["per_type"]["main"]),
                    _pct(cell["per_type"]["interaction_only"]),
                    _pct(cell["per_type"]["subgroup_only"]),
                    _pct(cell["commission_rate"]),
                )
            )
    lines += [
        "",
        "## Interaction-only omission growth (redundancy 0 -> %d)" % cfg["redundancy_levels"][-1],
        "",
    ]
    for name in EXPLAINER_NAMES:
        lines.append(
            "- %s: %s -> %s (growth %+.1f points)"
            % (
                name,
                _pct(agg["%s_interaction_omission_r%d" % (name, cfg["redundancy_levels"][0])]),
                _pct(agg["%s_interaction_omission_r%d" % (name, cfg["redundancy_levels"][-1])]),
                100.0 * agg["%s_interaction_growth" % name],
            )
        )
    lines += [
        "",
        "Caveat: all data is synthetic and seeded, generated from the "
        "audit's own structural process; the numbers measure explainer "
        "behaviour under known ground truth, not performance on real "
        "datasets. See README section 8.",
        "",
    ]
    return "\n".join(lines)


def headline_markdown(evaluation):
    agg = evaluation["aggregates"]
    cfg = evaluation["config"]
    last = cfg["redundancy_levels"][-1]
    lines = [
        "# Headline results",
        "",
        "An omission audit of post-hoc explainers against known "
        "data-generating ground truth: datasets are generated from a "
        "structural process that uses 3 main-effect features, 2 "
        "interaction-only features and 1 subgroup-only feature, plus 5 "
        "noise features and up to 2 planted redundant copies of a main "
        "feature. Each explainer's top-6 (the number of truly-used "
        "features) is scored for the used features it *fails to "
        "mention*, per usage type, averaged over %d seeds."
        % len(cfg["seeds"]),
        "",
        "- Interaction-only omission at redundancy 0: permutation %s, "
        "LOCO %s, sampled Shapley %s, surrogate coefficients %s -- "
        "against a random-ranking floor of %s."
        % (
            _pct(agg["permutation_interaction_omission_r0"]),
            _pct(agg["loco_interaction_omission_r0"]),
            _pct(agg["shapley_interaction_omission_r0"]),
            _pct(agg["surrogate_interaction_omission_r0"]),
            _pct(agg["random_interaction_omission_r0"]),
        ),
        "- At redundancy %d the same rates are: permutation %s, LOCO "
        "%s, Shapley %s, surrogate %s, random %s."
        % (
            last,
            _pct(agg["permutation_interaction_omission_r%d" % last]),
            _pct(agg["loco_interaction_omission_r%d" % last]),
            _pct(agg["shapley_interaction_omission_r%d" % last]),
            _pct(agg["surrogate_interaction_omission_r%d" % last]),
            _pct(agg["random_interaction_omission_r%d" % last]),
        ),
        "- Overall omission at redundancy %d: permutation %s, LOCO %s, "
        "Shapley %s, surrogate %s, random %s."
        % (
            last,
            _pct(agg["permutation_omission_r%d" % last]),
            _pct(agg["loco_omission_r%d" % last]),
            _pct(agg["shapley_omission_r%d" % last]),
            _pct(agg["surrogate_omission_r%d" % last]),
            _pct(agg["random_omission_r%d" % last]),
        ),
        "",
        "Caveat: all data is synthetic and seeded, with feature usage "
        "planted by construction; the audit measures explainer "
        "behaviour under known ground truth, not faithfulness on real "
        "datasets.",
        "",
    ]
    return "\n".join(lines)


_DRIFT_KEYS = (
    "permutation_interaction_omission_r0",
    "permutation_interaction_omission_r2",
    "loco_interaction_omission_r0",
    "loco_interaction_omission_r2",
    "shapley_interaction_omission_r0",
    "shapley_interaction_omission_r2",
    "surrogate_interaction_omission_r0",
    "surrogate_interaction_omission_r2",
    "random_interaction_omission_r0",
    "permutation_omission_r2",
    "surrogate_omission_r0",
    "permutation_interaction_growth",
)


def check_drift(stored, fresh, tol=1e-6):
    """Compare stored vs fresh headline aggregates; returns (ok, problems)."""
    problems = []
    stored_agg = stored.get("aggregates", {})
    fresh_agg = fresh.get("aggregates", {})
    for key in _DRIFT_KEYS:
        stored_value = stored_agg.get(key)
        fresh_value = fresh_agg.get(key)
        if stored_value is None or fresh_value is None:
            problems.append((key, stored_value, fresh_value, None))
            continue
        diff = abs(stored_value - fresh_value)
        if diff > tol:
            problems.append((key, stored_value, fresh_value, diff))
    return (not problems), problems
