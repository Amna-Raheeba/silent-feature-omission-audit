from omission_audit.models import (
    BASE_FEATURES,
    USED_FEATURES,
    budget,
    copy_name,
    feature_names,
    usage_map,
    used_features,
)


def test_base_feature_count():
    assert len(BASE_FEATURES) == 11


def test_feature_names_no_redundancy():
    assert feature_names(0) == list(BASE_FEATURES)


def test_feature_names_with_copies():
    names = feature_names(2)
    assert len(names) == 13
    assert names[-2:] == ["copy_1", "copy_2"]


def test_copy_name():
    assert copy_name(1) == "copy_1"
    assert copy_name(3) == "copy_3"


def test_usage_map_types():
    usage = usage_map(0)
    assert usage["main_a"] == "main"
    assert usage["int_a"] == "interaction_only"
    assert usage["int_b"] == "interaction_only"
    assert usage["sub_feat"] == "subgroup_only"
    assert usage["noise_a"] == "unused"


def test_usage_map_copies():
    usage = usage_map(2)
    assert usage["copy_1"] == "redundant_copy"
    assert usage["copy_2"] == "redundant_copy"


def test_usage_map_covers_all_names():
    for level in (0, 1, 2):
        assert set(usage_map(level)) == set(feature_names(level))


def test_used_features_exclude_copies_and_noise():
    used = used_features(2)
    assert set(used) == set(USED_FEATURES)
    assert "copy_1" not in used
    assert "noise_a" not in used


def test_budget_is_used_count():
    for level in (0, 1, 2):
        assert budget(level) == 6
