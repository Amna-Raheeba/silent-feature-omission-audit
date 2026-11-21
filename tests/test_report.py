from omission_audit.audit import audit_dataset
from omission_audit.explainers import run_explainers
from omission_audit.generate import generate_dataset
from omission_audit.report import audit_report_markdown, ranking_markdown


def _profiles():
    dataset = generate_dataset(seed=7, redundancy=1, n_train=200, n_test=100)
    return dataset, audit_dataset(dataset, run_explainers(dataset))


def test_report_has_title_and_explainers():
    dataset, profiles = _profiles()
    text = audit_report_markdown(dataset, profiles)
    assert "# Omission audit report" in text
    for name in ("permutation", "loco", "shapley", "surrogate", "random"):
        assert name in text


def test_report_mentions_budget():
    dataset, profiles = _profiles()
    text = audit_report_markdown(dataset, profiles)
    assert "budget k = 6" in text


def test_ranking_markdown_lists_all_features():
    dataset, profiles = _profiles()
    text = ranking_markdown(dataset, profiles, "permutation")
    for name in dataset["feature_names"]:
        assert name in text


def test_ranking_markdown_header():
    dataset, profiles = _profiles()
    text = ranking_markdown(dataset, profiles, "loco")
    assert "## Ranking: loco" in text
