from omission_audit.audit import audit_dataset, omission_profile, rank_features
from omission_audit.generate import generate_dataset
from omission_audit.models import usage_map


def _scores(order):
    """Scores that rank features exactly in the given order."""
    return dict((name, float(len(order) - i)) for i, name in enumerate(order))


def test_rank_features_descending():
    scores = {"b": 1.0, "a": 3.0, "c": 2.0}
    assert rank_features(scores) == ["a", "c", "b"]


def test_rank_features_tie_break_by_name():
    scores = {"b": 1.0, "a": 1.0}
    assert rank_features(scores) == ["a", "b"]


def test_perfect_ranking_no_omission():
    usage = usage_map(0)
    order = [
        "main_a", "main_b", "main_c", "int_a", "int_b", "sub_feat",
        "noise_a", "noise_b", "noise_c", "noise_d", "noise_e",
    ]
    profile = omission_profile(_scores(order), usage, 0)
    assert profile["omission_rate"] == 0.0
    assert profile["commission_rate"] == 0.0
    assert profile["omitted"] == []


def test_interaction_omitted_when_ranked_last():
    usage = usage_map(0)
    order = [
        "main_a", "main_b", "main_c", "sub_feat",
        "noise_a", "noise_b", "noise_c", "noise_d", "noise_e",
        "int_a", "int_b",
    ]
    profile = omission_profile(_scores(order), usage, 0)
    cell = profile["per_type"]["interaction_only"]
    assert cell["omission_rate"] == 1.0
    assert set(cell["omitted_names"]) == {"int_a", "int_b"}
    assert profile["per_type"]["main"]["omission_rate"] == 0.0


def test_commission_counts_noise_and_copies():
    usage = usage_map(1)
    order = [
        "main_a", "main_b", "main_c", "int_a", "copy_1", "noise_a",
        "int_b", "sub_feat",
        "noise_b", "noise_c", "noise_d", "noise_e",
    ]
    profile = omission_profile(_scores(order), usage, 1)
    assert profile["copy_slots"] == 1
    assert profile["noise_slots"] == 1
    assert profile["commission_rate"] == 2 / 6.0
    assert set(profile["omitted"]) == {"int_b", "sub_feat"}


def test_source_displaced_by_copy():
    usage = usage_map(1)
    order = [
        "copy_1", "main_b", "main_c", "int_a", "int_b", "sub_feat",
        "main_a", "noise_a", "noise_b", "noise_c", "noise_d", "noise_e",
    ]
    profile = omission_profile(_scores(order), usage, 1)
    assert profile["source_displaced"] is True
    assert profile["source_omitted"] is False


def test_source_omitted_when_neither_in_topk():
    usage = usage_map(1)
    order = [
        "main_b", "main_c", "int_a", "int_b", "sub_feat", "noise_a",
        "main_a", "copy_1", "noise_b", "noise_c", "noise_d", "noise_e",
    ]
    profile = omission_profile(_scores(order), usage, 1)
    assert profile["source_omitted"] is True
    assert profile["source_displaced"] is False


def test_source_present_no_flags():
    usage = usage_map(1)
    order = [
        "main_a", "main_b", "main_c", "int_a", "int_b", "sub_feat",
        "copy_1", "noise_a", "noise_b", "noise_c", "noise_d", "noise_e",
    ]
    profile = omission_profile(_scores(order), usage, 1)
    assert profile["source_omitted"] is False
    assert profile["source_displaced"] is False


def test_budget_and_topk_size():
    usage = usage_map(2)
    scores = dict((name, 1.0) for name in usage)
    profile = omission_profile(scores, usage, 2)
    assert profile["budget"] == 6
    assert len(profile["top_k"]) == 6
    assert len(profile["ranking"]) == 13


def test_audit_dataset_keys():
    dataset = generate_dataset(seed=1, redundancy=0, n_train=30, n_test=10)
    scores = dict((name, 1.0) for name in dataset["feature_names"])
    profiles = audit_dataset(dataset, {"stub": scores})
    assert set(profiles) == {"stub"}
    assert profiles["stub"]["budget"] == 6
