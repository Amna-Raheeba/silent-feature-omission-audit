import math

from omission_audit.explainers import (
    EXPLAINER_NAMES,
    loco_importance,
    permutation_importance,
    random_importance,
    run_explainers,
    shapley_importance,
    surrogate_importance,
    train_tree,
)
from omission_audit.generate import generate_dataset


def _small(redundancy=0):
    return generate_dataset(seed=7, redundancy=redundancy, n_train=300, n_test=150)


def test_explainer_names():
    assert EXPLAINER_NAMES == (
        "permutation", "loco", "shapley", "surrogate", "random"
    )


def test_run_explainers_covers_all_features():
    dataset = _small(redundancy=1)
    results = run_explainers(dataset)
    assert set(results) == set(EXPLAINER_NAMES)
    for scores in results.values():
        assert set(scores) == set(dataset["feature_names"])


def test_scores_are_finite():
    dataset = _small()
    for scores in run_explainers(dataset).values():
        assert all(math.isfinite(value) for value in scores.values())


def test_permutation_main_beats_noise():
    dataset = _small()
    tree = train_tree(dataset)
    scores = permutation_importance(tree, dataset)
    assert scores["main_a"] > scores["noise_a"]
    assert scores["main_a"] > 0.0


def test_loco_main_beats_noise():
    dataset = _small()
    scores = loco_importance(dataset)
    assert scores["main_a"] > scores["noise_a"]


def test_surrogate_blind_to_interaction():
    # A linear surrogate cannot represent a pure product term: its
    # coefficients for the interaction pair stay near zero relative to
    # the main effects. This is a structural property, not tuning.
    dataset = generate_dataset(seed=7, redundancy=0, n_train=800, n_test=10)
    scores = surrogate_importance(dataset)
    assert scores["main_a"] > scores["int_a"]
    assert scores["main_a"] > scores["int_b"]


def test_shapley_scores_sum_is_bounded():
    dataset = _small()
    tree = train_tree(dataset)
    scores = shapley_importance(tree, dataset, n_permutations=8, eval_rows=60)
    assert abs(sum(scores.values())) <= 1.0


def test_shapley_main_positive():
    dataset = _small()
    tree = train_tree(dataset)
    scores = shapley_importance(tree, dataset, n_permutations=8, eval_rows=60)
    assert scores["main_a"] > scores["noise_a"]


def test_random_scores_in_unit_interval():
    dataset = _small()
    scores = random_importance(dataset)
    assert all(0.0 <= value < 1.0 for value in scores.values())


def test_random_deterministic():
    dataset = _small()
    assert random_importance(dataset) == random_importance(dataset)
