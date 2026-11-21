"""Small deterministic models, pure stdlib.

Two models are needed:

- :class:`DecisionTree` -- a CART classifier (Gini impurity, quantile
  candidate thresholds). A tree is used as the explained model because
  it can represent the interaction and subgroup structure of the DGP,
  so explainer omissions are not just "the model could not use the
  feature anyway".
- :func:`fit_ridge` -- a standardised ridge regression used as the
  linear surrogate explainer. It structurally cannot represent an
  interaction-only or subgroup-gated effect, which is part of what the
  audit measures.

Both are deterministic: ties are broken by lowest feature index, then
lowest threshold.
"""


def _gini(count0, count1):
    total = count0 + count1
    if total == 0:
        return 0.0
    p0 = count0 / total
    p1 = count1 / total
    return 1.0 - p0 * p0 - p1 * p1


class _Node(object):
    __slots__ = ("feature", "threshold", "left", "right", "prediction")

    def __init__(self, prediction):
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.prediction = prediction


class DecisionTree(object):
    """A small CART classifier with deterministic tie-breaking."""

    def __init__(self, max_depth=5, min_leaf=20, max_thresholds=12):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_thresholds = max_thresholds
        self.root = None
        self.n_features = 0

    def fit(self, X, y):
        self.n_features = len(X[0])
        indices = list(range(len(X)))
        self.root = self._build(X, y, indices, 0)
        return self

    def _build(self, X, y, indices, depth):
        count1 = sum(y[i] for i in indices)
        count0 = len(indices) - count1
        node = _Node(1 if count1 >= count0 else 0)
        if depth >= self.max_depth or count0 == 0 or count1 == 0:
            return node
        if len(indices) < 2 * self.min_leaf:
            return node
        split = self._best_split(X, y, indices, count0, count1)
        if split is None:
            return node
        feature, threshold, left_idx, right_idx = split
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X, y, left_idx, depth + 1)
        node.right = self._build(X, y, right_idx, depth + 1)
        return node

    def _best_split(self, X, y, indices, count0, count1):
        n = len(indices)
        parent = _gini(count0, count1)
        best = None
        best_gain = 1e-12
        for feature in range(self.n_features):
            pairs = sorted((X[i][feature], y[i]) for i in indices)
            thresholds = self._candidate_thresholds(pairs)
            left0 = 0
            left1 = 0
            pos = 0
            for threshold in thresholds:
                while pos < n and pairs[pos][0] <= threshold:
                    if pairs[pos][1]:
                        left1 += 1
                    else:
                        left0 += 1
                    pos += 1
                right0 = count0 - left0
                right1 = count1 - left1
                n_left = left0 + left1
                n_right = n - n_left
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue
                gain = parent - (
                    n_left * _gini(left0, left1) + n_right * _gini(right0, right1)
                ) / n
                if gain > best_gain:
                    best_gain = gain
                    best = (feature, threshold)
        if best is None:
            return None
        feature, threshold = best
        left_idx = [i for i in indices if X[i][feature] <= threshold]
        right_idx = [i for i in indices if X[i][feature] > threshold]
        return feature, threshold, left_idx, right_idx

    def _candidate_thresholds(self, pairs):
        """Up to ``max_thresholds`` midpoints between distinct values."""
        distinct = []
        for value, _label in pairs:
            if not distinct or value != distinct[-1]:
                distinct.append(value)
        if len(distinct) < 2:
            return []
        midpoints = [
            (distinct[i] + distinct[i + 1]) / 2.0
            for i in range(len(distinct) - 1)
        ]
        if len(midpoints) <= self.max_thresholds:
            return midpoints
        step = len(midpoints) / float(self.max_thresholds)
        picked = []
        for k in range(self.max_thresholds):
            picked.append(midpoints[int(k * step)])
        return picked

    def predict_one(self, row):
        node = self.root
        while node.feature >= 0:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X):
        return [self.predict_one(row) for row in X]


def accuracy(model, X, y):
    preds = model.predict(X)
    correct = sum(1 for pred, target in zip(preds, y) if pred == target)
    return correct / float(len(y)) if y else 0.0


def _solve_linear(matrix, vector):
    """Gaussian elimination with partial pivoting (small systems)."""
    n = len(vector)
    aug = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / pivot_value
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] if abs(aug[i][i]) >= 1e-12 else 0.0 for i in range(n)]


def fit_ridge(X, y, lam=1.0):
    """Standardised ridge regression; returns (coefs, means, stds).

    Coefficients are on standardised features, so their magnitudes are
    comparable across features -- that is what the surrogate explainer
    ranks by.
    """
    n = len(X)
    p = len(X[0])
    means = [sum(row[j] for row in X) / n for j in range(p)]
    stds = []
    for j in range(p):
        var = sum((row[j] - means[j]) ** 2 for row in X) / n
        stds.append(var ** 0.5 if var > 0 else 1.0)
    y_mean = sum(y) / float(n)
    gram = [[0.0] * p for _ in range(p)]
    rhs = [0.0] * p
    for row, target in zip(X, y):
        z = [(row[j] - means[j]) / stds[j] for j in range(p)]
        centred = target - y_mean
        for a in range(p):
            rhs[a] += z[a] * centred
            for b in range(a, p):
                gram[a][b] += z[a] * z[b]
    for a in range(p):
        for b in range(a):
            gram[a][b] = gram[b][a]
        gram[a][a] += lam
    coefs = _solve_linear(gram, rhs)
    return coefs, means, stds
