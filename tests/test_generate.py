from omission_audit.generate import generate_dataset


def _corr(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5


def test_deterministic_same_seed():
    first = generate_dataset(seed=3, redundancy=1, n_train=50, n_test=20)
    second = generate_dataset(seed=3, redundancy=1, n_train=50, n_test=20)
    assert first["X_train"] == second["X_train"]
    assert first["y_train"] == second["y_train"]


def test_different_seeds_differ():
    first = generate_dataset(seed=3, redundancy=0, n_train=50, n_test=20)
    second = generate_dataset(seed=4, redundancy=0, n_train=50, n_test=20)
    assert first["X_train"] != second["X_train"]


def test_shapes():
    dataset = generate_dataset(seed=3, redundancy=2, n_train=40, n_test=15)
    assert len(dataset["X_train"]) == 40
    assert len(dataset["X_test"]) == 15
    assert all(len(row) == 13 for row in dataset["X_train"])
    assert len(dataset["y_train"]) == 40


def test_labels_binary():
    dataset = generate_dataset(seed=3, redundancy=0, n_train=100, n_test=50)
    assert set(dataset["y_train"]) <= {0, 1}
    assert set(dataset["y_test"]) <= {0, 1}


def test_labels_roughly_balanced():
    dataset = generate_dataset(seed=3, redundancy=0, n_train=1000, n_test=10)
    rate = sum(dataset["y_train"]) / float(len(dataset["y_train"]))
    assert 0.35 < rate < 0.65


def test_no_subgroup_column():
    dataset = generate_dataset(seed=3, redundancy=0, n_train=10, n_test=5)
    assert "subgroup" not in dataset["feature_names"]
    assert len(dataset["feature_names"]) == 11


def test_usage_matches_names():
    dataset = generate_dataset(seed=3, redundancy=1, n_train=10, n_test=5)
    assert set(dataset["usage"]) == set(dataset["feature_names"])


def test_copies_track_source():
    dataset = generate_dataset(seed=3, redundancy=1, n_train=400, n_test=10)
    names = dataset["feature_names"]
    source = [row[names.index("main_a")] for row in dataset["X_train"]]
    copy = [row[names.index("copy_1")] for row in dataset["X_train"]]
    assert _corr(source, copy) > 0.9


def test_no_copies_at_level_zero():
    dataset = generate_dataset(seed=3, redundancy=0, n_train=10, n_test=5)
    assert not any(name.startswith("copy_") for name in dataset["feature_names"])
