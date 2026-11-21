from omission_audit.generate import generate_dataset
from omission_audit.tree import DecisionTree, accuracy, fit_ridge


def _separable():
    X = []
    y = []
    for i in range(60):
        X.append([float(i % 2), float((i // 2) % 2)])
        y.append(i % 2)
    return X, y


def test_tree_fits_separable_data():
    X, y = _separable()
    tree = DecisionTree(max_depth=3, min_leaf=5).fit(X, y)
    assert accuracy(tree, X, y) == 1.0


def test_tree_deterministic():
    X, y = _separable()
    first = DecisionTree(max_depth=3, min_leaf=5).fit(X, y).predict(X)
    second = DecisionTree(max_depth=3, min_leaf=5).fit(X, y).predict(X)
    assert first == second


def test_tree_predict_shape():
    dataset = generate_dataset(seed=2, redundancy=0, n_train=200, n_test=30)
    tree = DecisionTree().fit(dataset["X_train"], dataset["y_train"])
    preds = tree.predict(dataset["X_test"])
    assert len(preds) == 30
    assert set(preds) <= {0, 1}


def test_tree_beats_chance_on_dgp():
    dataset = generate_dataset(seed=2, redundancy=0, n_train=800, n_test=400)
    tree = DecisionTree().fit(dataset["X_train"], dataset["y_train"])
    assert accuracy(tree, dataset["X_test"], dataset["y_test"]) > 0.65


def test_accuracy_helper_perfect_and_zero():
    class Fixed(object):
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return [self.value] * len(X)

    assert accuracy(Fixed(1), [[0.0], [1.0]], [1, 1]) == 1.0
    assert accuracy(Fixed(1), [[0.0], [1.0]], [0, 0]) == 0.0


def test_ridge_recovers_coefficient_signs():
    X = []
    y = []
    for i in range(200):
        a = (i % 20) - 10.0
        b = ((i * 7) % 20) - 10.0
        X.append([a / 10.0, b / 10.0])
        y.append(1 if 2.0 * (a / 10.0) - 1.0 * (b / 10.0) > 0 else 0)
    coefs, _means, _stds = fit_ridge(X, y)
    assert coefs[0] > 0
    assert coefs[1] < 0
    assert abs(coefs[0]) > abs(coefs[1])


def test_ridge_output_shapes():
    X = [[float(i), float(i % 3)] for i in range(50)]
    y = [i % 2 for i in range(50)]
    coefs, means, stds = fit_ridge(X, y)
    assert len(coefs) == len(means) == len(stds) == 2
    assert all(std > 0 for std in stds)
