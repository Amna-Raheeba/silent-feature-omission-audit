"""Drift-guard logic tests on fabricated evaluations.

The full reproduction (re-running the canonical evaluation and
comparing against results/evaluation.json) is exercised by
scripts/reproduce_results.py; here only the comparison logic is
unit-tested so the suite stays fast.
"""

from omission_audit.evaluate import check_drift

_BASE = {
    "permutation_interaction_omission_r0": 0.2,
    "permutation_interaction_omission_r2": 0.9,
    "loco_interaction_omission_r0": 0.2,
    "loco_interaction_omission_r2": 0.2,
    "shapley_interaction_omission_r0": 0.1,
    "shapley_interaction_omission_r2": 0.9,
    "surrogate_interaction_omission_r0": 0.5,
    "surrogate_interaction_omission_r2": 1.0,
    "random_interaction_omission_r0": 0.6,
    "permutation_omission_r2": 0.366667,
    "surrogate_omission_r0": 0.166667,
    "permutation_interaction_growth": 0.7,
}


def _wrap(agg):
    return {"aggregates": dict(agg)}


def test_identical_passes():
    ok, problems = check_drift(_wrap(_BASE), _wrap(_BASE))
    assert ok
    assert problems == []


def test_within_tolerance_passes():
    fresh = dict(_BASE)
    fresh["loco_interaction_omission_r0"] += 5e-7
    ok, _problems = check_drift(_wrap(_BASE), _wrap(fresh))
    assert ok


def test_beyond_tolerance_fails():
    fresh = dict(_BASE)
    fresh["surrogate_interaction_omission_r2"] -= 0.1
    ok, problems = check_drift(_wrap(_BASE), _wrap(fresh))
    assert not ok
    assert problems[0][0] == "surrogate_interaction_omission_r2"


def test_missing_key_fails():
    fresh = dict(_BASE)
    del fresh["permutation_omission_r2"]
    ok, problems = check_drift(_wrap(_BASE), _wrap(fresh))
    assert not ok
    assert any(key == "permutation_omission_r2" for key, *_rest in problems)
