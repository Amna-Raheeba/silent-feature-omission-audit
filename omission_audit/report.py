"""Markdown reports for a single-dataset audit."""

from .explainers import EXPLAINER_NAMES
from .models import USAGE_MAIN


def _pct(value):
    return "%.1f%%" % (100.0 * value)


def audit_report_markdown(dataset, profiles):
    """Per-explainer omission profile for one audited dataset."""
    lines = [
        "# Omission audit report",
        "",
        "Dataset: seed %d, redundancy %d, %d train / %d test rows, "
        "budget k = %d."
        % (
            dataset["seed"],
            dataset["redundancy"],
            len(dataset["X_train"]),
            len(dataset["X_test"]),
            profiles[EXPLAINER_NAMES[0]]["budget"],
        ),
        "",
        "| Explainer | Omission (all used) | Main | Interaction-only | Subgroup-only | Commission | Copy slots | Noise slots |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in EXPLAINER_NAMES:
        profile = profiles[name]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %d | %d |"
            % (
                name,
                _pct(profile["omission_rate"]),
                _pct(profile["per_type"][USAGE_MAIN]["omission_rate"]),
                _pct(profile["per_type"]["interaction_only"]["omission_rate"]),
                _pct(profile["per_type"]["subgroup_only"]["omission_rate"]),
                _pct(profile["commission_rate"]),
                profile["copy_slots"],
                profile["noise_slots"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def ranking_markdown(dataset, profiles, explainer):
    """One explainer's full ranking with usage types and omissions."""
    profile = profiles[explainer]
    usage = dataset["usage"]
    top_set = set(profile["top_k"])
    lines = [
        "## Ranking: %s" % explainer,
        "",
        "| Rank | Feature | Usage | In top-k |",
        "|---|---|---|---|",
    ]
    for rank, name in enumerate(profile["ranking"], start=1):
        lines.append(
            "| %d | %s | %s | %s |"
            % (rank, name, usage[name], "yes" if name in top_set else "no")
        )
    if profile["omitted"]:
        lines.append("")
        lines.append(
            "Omitted used features: %s" % ", ".join(profile["omitted"])
        )
    lines.append("")
    return "\n".join(lines)
