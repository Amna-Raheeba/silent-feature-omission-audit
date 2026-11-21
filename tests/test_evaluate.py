import copy

from omission_audit.evaluate import check_drift, run_evaluation, write_results

SMALL = {
    "seeds": (101,),
    "redundancy_levels": (0, 2),
    "n_train": 200,
    "n_test": 100,
}


def test_run_evaluation_structure():
    evaluation = run_evaluation(SMALL)
    assert "aggregates" in evaluation
    assert "rows" in evaluation
    # 2 levels x 5 explainers x 4 usage rows
    assert len(evaluation["rows"]) == 40


def test_run_evaluation_deterministic():
    first = run_evaluation(SMALL)
    second = run_evaluation(SMALL)
    assert first["aggregates"] == second["aggregates"]


def test_aggregates_have_drift_keys():
    evaluation = run_evaluation(SMALL)
    agg = evaluation["aggregates"]
    for key in (
        "permutation_interaction_omission_r0",
        "permutation_interaction_omission_r2",
        "surrogate_interaction_omission_r2",
        "random_interaction_omission_r0",
        "permutation_interaction_growth",
    ):
        assert key in agg


def test_rates_within_unit_interval():
    evaluation = run_evaluation(SMALL)
    for row in evaluation["rows"]:
        assert 0.0 <= row["omission_rate"] <= 1.0


def test_check_drift_identical_passes():
    evaluation = run_evaluation(SMALL)
    ok, problems = check_drift(evaluation, copy.deepcopy(evaluation))
    assert ok
    assert problems == []


def test_check_drift_detects_change():
    evaluation = run_evaluation(SMALL)
    fresh = copy.deepcopy(evaluation)
    fresh["aggregates"]["permutation_interaction_omission_r0"] += 0.5
    ok, problems = check_drift(evaluation, fresh)
    assert not ok
    assert problems[0][0] == "permutation_interaction_omission_r0"


def test_write_results_creates_files(tmp_path):
    evaluation = run_evaluation(SMALL)
    write_results(evaluation, str(tmp_path))
    for name in (
        "evaluation.json",
        "evaluation.csv",
        "evaluation_summary.md",
        "HEADLINE.md",
    ):
        assert (tmp_path / name).exists()
