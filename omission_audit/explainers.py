"""Post-hoc explainers, all stdlib, all returning feature -> score.

Four real explainers plus a random-ranking baseline:

- ``permutation`` -- accuracy drop when one test column is shuffled.
- ``loco`` -- leave-one-covariate-out: retrain the tree without the
  feature and measure the accuracy drop (ROAR-style removal, without
  the retraining-cost blindness: the drop *is* the score).
- ``shapley`` -- sampled Shapley values over permutations of features,
  with absent features replaced by background-row values (marginal
  sampling) and the value function = accuracy on a fixed eval subset.
- ``surrogate`` -- absolute standardised ridge coefficients from a
  linear surrogate fit on the training data.
- ``random`` -- seeded uniform scores; the floor every real explainer
  must beat.

Every explainer sees the same trained tree and the same data. Scores
are importances: higher means "the explainer thinks this feature
matters more". Rankings are derived in :mod:`omission_audit.audit`.
"""

import random

from .tree import DecisionTree, accuracy, fit_ridge

EXPLAINER_NAMES = ("permutation", "loco", "shapley", "surrogate", "random")

TREE_PARAMS = {"max_depth": 5, "min_leaf": 20}


def train_tree(dataset):
    tree = DecisionTree(**TREE_PARAMS)
    tree.fit(dataset["X_train"], dataset["y_train"])
    return tree


def permutation_importance(tree, dataset, seed=5, repeats=3):
    X = dataset["X_test"]
    y = dataset["y_test"]
    names = dataset["feature_names"]
    base = accuracy(tree, X, y)
    rng = random.Random(seed)
    scores = {}
    for j, name in enumerate(names):
        drops = []
        for _ in range(repeats):
            shuffled = [list(row) for row in X]
            column = [row[j] for row in X]
            rng.shuffle(column)
            for i in range(len(shuffled)):
                shuffled[i][j] = column[i]
            drops.append(base - accuracy(tree, shuffled, y))
        scores[name] = sum(drops) / len(drops)
    return scores


def loco_importance(dataset):
    names = dataset["feature_names"]
    full = train_tree(dataset)
    base = accuracy(full, dataset["X_test"], dataset["y_test"])
    scores = {}
    for j, name in enumerate(names):
        keep = [k for k in range(len(names)) if k != j]
        reduced = {
            "X_train": [[row[k] for k in keep] for row in dataset["X_train"]],
            "y_train": dataset["y_train"],
            "X_test": [[row[k] for k in keep] for row in dataset["X_test"]],
            "y_test": dataset["y_test"],
        }
        tree = train_tree(reduced)
        scores[name] = base - accuracy(tree, reduced["X_test"], reduced["y_test"])
    return scores


def shapley_importance(tree, dataset, seed=11, n_permutations=32, eval_rows=120,
                       background_rows=40):
    """Sampled Shapley with marginal (background) replacement.

    For each sampled permutation, features are "added" one at a time:
    the eval matrix starts as background values and each added feature
    takes its eval-row value. A feature's marginal contribution is the
    accuracy change when it is added; the score is the mean marginal
    over permutations.
    """
    X = dataset["X_test"][:eval_rows]
    y = dataset["y_test"][:eval_rows]
    background = dataset["X_train"][:background_rows]
    names = dataset["feature_names"]
    p = len(names)
    rng = random.Random(seed)
    base_rows = [background[rng.randrange(len(background))] for _ in X]

    def value(matrix):
        return accuracy(tree, matrix, y)

    totals = dict((name, 0.0) for name in names)
    for _ in range(n_permutations):
        order = list(range(p))
        rng.shuffle(order)
        matrix = [list(row) for row in base_rows]
        previous = value(matrix)
        for j in order:
            for i in range(len(X)):
                matrix[i][j] = X[i][j]
            current = value(matrix)
            totals[names[j]] += current - previous
            previous = current
    return dict(
        (name, totals[name] / n_permutations) for name in names
    )


def surrogate_importance(dataset):
    coefs, _means, _stds = fit_ridge(dataset["X_train"], dataset["y_train"])
    return dict(
        (name, abs(coefs[j]))
        for j, name in enumerate(dataset["feature_names"])
    )


def random_importance(dataset, seed=17):
    rng = random.Random(seed)
    return dict((name, rng.random()) for name in dataset["feature_names"])


def run_explainers(dataset):
    """Run every explainer on one dataset; returns name -> scores."""
    tree = train_tree(dataset)
    return {
        "permutation": permutation_importance(tree, dataset),
        "loco": loco_importance(dataset),
        "shapley": shapley_importance(tree, dataset, seed=11 + dataset["seed"]),
        "surrogate": surrogate_importance(dataset),
        "random": random_importance(dataset, seed=17 + dataset["seed"]),
    }
