"""Ground-truth specification of the data-generating process (DGP).

The whole audit rests on one fact: the data is generated from a known
structural process, so the set of features the process actually uses --
and *how* it uses each one -- is known exactly, before any explainer
runs.

Usage types
-----------
- ``main``: the feature enters the latent score with its own linear
  term.
- ``interaction_only``: the feature enters only inside a product term;
  it has no main effect, so any method that scores features one at a
  time on marginal signal can miss it.
- ``subgroup_only``: the feature's effect is switched on only for a
  latent subgroup of rows (the subgroup indicator is *not* a column),
  so its marginal signal is diluted across the full sample.
- ``redundant_copy``: not used by the DGP at all; a noisy copy of a
  main-effect feature, planted to test how explainers behave under
  redundancy (the condition under which removal-based faithfulness
  metrics are known to break down).
- ``unused``: pure noise; flagging one is a commission error.

A dataset is a plain dict with keys ``feature_names``, ``usage``,
``X_train``, ``y_train``, ``X_test``, ``y_test``, ``seed`` and
``redundancy``.
"""

USAGE_MAIN = "main"
USAGE_INTERACTION = "interaction_only"
USAGE_SUBGROUP = "subgroup_only"
USAGE_COPY = "redundant_copy"
USAGE_UNUSED = "unused"

USAGE_TYPES = (
    USAGE_MAIN,
    USAGE_INTERACTION,
    USAGE_SUBGROUP,
    USAGE_COPY,
    USAGE_UNUSED,
)

# Features the DGP uses directly, in generation order.
MAIN_FEATURES = ("main_a", "main_b", "main_c")
INTERACTION_FEATURES = ("int_a", "int_b")
SUBGROUP_FEATURES = ("sub_feat",)
UNUSED_FEATURES = ("noise_a", "noise_b", "noise_c", "noise_d", "noise_e")

BASE_FEATURES = (
    MAIN_FEATURES + INTERACTION_FEATURES + SUBGROUP_FEATURES + UNUSED_FEATURES
)

USED_FEATURES = MAIN_FEATURES + INTERACTION_FEATURES + SUBGROUP_FEATURES

# The feature that planted redundant copies duplicate.
COPY_SOURCE = "main_a"
COPY_PREFIX = "copy_"

# Structural coefficients of the latent score. Kept in one place so the
# generator, the tests and the README all describe the same process.
DGP_PARAMS = {
    "main_coefs": {"main_a": 1.2, "main_b": 0.9, "main_c": -0.8},
    "interaction_coef": 1.7,
    "subgroup_coef": 1.5,
    "subgroup_rate": 0.35,
    "noise_sd": 0.8,
    "copy_noise_sd": 0.1,
}


def copy_name(index):
    """Name of the ``index``-th planted copy (1-based)."""
    return "%s%d" % (COPY_PREFIX, index)


def feature_names(redundancy):
    """All column names for a dataset with ``redundancy`` planted copies."""
    names = list(BASE_FEATURES)
    for index in range(1, redundancy + 1):
        names.append(copy_name(index))
    return names


def usage_map(redundancy):
    """Map feature name -> usage type for the given redundancy level."""
    usage = {}
    for name in MAIN_FEATURES:
        usage[name] = USAGE_MAIN
    for name in INTERACTION_FEATURES:
        usage[name] = USAGE_INTERACTION
    for name in SUBGROUP_FEATURES:
        usage[name] = USAGE_SUBGROUP
    for name in UNUSED_FEATURES:
        usage[name] = USAGE_UNUSED
    for index in range(1, redundancy + 1):
        usage[copy_name(index)] = USAGE_COPY
    return usage


def used_features(redundancy):
    """Features the DGP uses (copies are *not* used; they duplicate one)."""
    return list(USED_FEATURES)


def budget(redundancy):
    """Top-k budget for the audit: the number of truly-used features."""
    return len(USED_FEATURES)
