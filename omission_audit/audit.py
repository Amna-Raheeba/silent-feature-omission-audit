"""The omission audit: score a ranking against DGP ground truth.

An explainer *omits* a feature when the feature truly drives the
outcome but does not appear in the explainer's top-k, where k is the
number of truly-used features (the natural budget: an explanation that
names k features should name the k that matter). Omissions are
reported per usage type, because the headline question is *where*
omission concentrates -- interaction-only and subgroup-only usage --
not just how much there is.

Commissions are the mirror error: top-k slots spent on features the
DGP never used. Slots spent on planted redundant copies are reported
separately from slots spent on pure noise, because a copy at least
carries its source's signal. For the copied source feature the audit
also distinguishes:

- ``source_omitted``: neither the source nor any copy is in the top-k
  (the signal is truly lost from the explanation);
- ``source_displaced``: a copy is in the top-k but the source is not
  (the explanation names the duplicate, not the feature).
"""

from .models import (
    COPY_SOURCE,
    USAGE_COPY,
    USAGE_MAIN,
    USAGE_UNUSED,
    budget,
    copy_name,
    used_features,
)


def rank_features(scores):
    """Feature names ordered by descending score, name as tie-break."""
    return sorted(scores, key=lambda name: (-scores[name], name))


def omission_profile(scores, usage, redundancy):
    """Omission/commission profile of one explainer's scores."""
    names = rank_features(scores)
    k = budget(redundancy)
    top = names[:k]
    top_set = set(top)
    used = used_features(redundancy)

    per_type = {}
    for usage_type in (USAGE_MAIN, "interaction_only", "subgroup_only"):
        members = [name for name in used if usage[name] == usage_type]
        omitted = [name for name in members if name not in top_set]
        per_type[usage_type] = {
            "n": len(members),
            "omitted": len(omitted),
            "omitted_names": omitted,
            "omission_rate": len(omitted) / float(len(members)) if members else 0.0,
        }

    omitted_used = [name for name in used if name not in top_set]
    copy_slots = [name for name in top if usage[name] == USAGE_COPY]
    noise_slots = [name for name in top if usage[name] == USAGE_UNUSED]

    copies = [copy_name(i) for i in range(1, redundancy + 1)]
    source_in = COPY_SOURCE in top_set
    any_copy_in = any(name in top_set for name in copies)
    return {
        "budget": k,
        "ranking": names,
        "top_k": top,
        "omitted": omitted_used,
        "omission_rate": len(omitted_used) / float(len(used)),
        "per_type": per_type,
        "commission_rate": (len(copy_slots) + len(noise_slots)) / float(k),
        "copy_slots": len(copy_slots),
        "noise_slots": len(noise_slots),
        "source_omitted": (not source_in) and (not any_copy_in),
        "source_displaced": (not source_in) and any_copy_in,
    }


def audit_dataset(dataset, explainer_scores):
    """Profiles for every explainer on one dataset."""
    usage = dataset["usage"]
    redundancy = dataset["redundancy"]
    return dict(
        (name, omission_profile(scores, usage, redundancy))
        for name, scores in explainer_scores.items()
    )
