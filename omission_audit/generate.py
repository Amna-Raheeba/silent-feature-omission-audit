"""Seeded generation of datasets from the known structural process.

Every dataset is fully determined by ``(seed, redundancy, n_train,
n_test)``: same arguments, same rows, on any machine. The latent score
follows :data:`omission_audit.models.DGP_PARAMS`; the label is 1 when
the score is positive.

The subgroup indicator is drawn per row but never stored as a column:
``sub_feat`` only moves the score for subgroup rows, which is what
makes its usage *subgroup-only* by construction.
"""

import json
import random

from .models import (
    BASE_FEATURES,
    COPY_SOURCE,
    DGP_PARAMS,
    feature_names,
    usage_map,
)


def _draw_row(rng, redundancy):
    """Draw one feature row (dict) and its latent subgroup flag."""
    values = {}
    for name in BASE_FEATURES:
        values[name] = rng.gauss(0.0, 1.0)
    subgroup = 1 if rng.random() < DGP_PARAMS["subgroup_rate"] else 0
    for index in range(1, redundancy + 1):
        values["copy_%d" % index] = values[COPY_SOURCE] + rng.gauss(
            0.0, DGP_PARAMS["copy_noise_sd"]
        )
    return values, subgroup


def _label(values, subgroup, rng):
    coefs = DGP_PARAMS["main_coefs"]
    score = 0.0
    for name, coef in coefs.items():
        score += coef * values[name]
    score += DGP_PARAMS["interaction_coef"] * values["int_a"] * values["int_b"]
    if subgroup:
        score += DGP_PARAMS["subgroup_coef"] * values["sub_feat"]
    score += rng.gauss(0.0, DGP_PARAMS["noise_sd"])
    return 1 if score > 0.0 else 0


def generate_dataset(seed=101, redundancy=0, n_train=1000, n_test=500):
    """Generate one dataset dict (see :mod:`omission_audit.models`)."""
    rng = random.Random(seed * 100003 + redundancy)
    names = feature_names(redundancy)

    def make_rows(count):
        rows = []
        labels = []
        for _ in range(count):
            values, subgroup = _draw_row(rng, redundancy)
            rows.append([values[name] for name in names])
            labels.append(_label(values, subgroup, rng))
        return rows, labels

    X_train, y_train = make_rows(n_train)
    X_test, y_test = make_rows(n_test)
    return {
        "seed": seed,
        "redundancy": redundancy,
        "feature_names": names,
        "usage": usage_map(redundancy),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def dataset_to_jsonable(dataset, ndigits=4):
    """A copy of the dataset with floats rounded, for compact JSON."""

    def rounded(rows):
        return [[round(value, ndigits) for value in row] for row in rows]

    out = dict(dataset)
    out["X_train"] = rounded(dataset["X_train"])
    out["X_test"] = rounded(dataset["X_test"])
    return out


def save_dataset(dataset, path):
    with open(path, "w") as handle:
        json.dump(dataset_to_jsonable(dataset), handle)


def load_dataset(path):
    with open(path) as handle:
        return json.load(handle)
