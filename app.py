"""Streamlit dashboard for silent-feature-omission-audit.

Three tabs:
1. Omission explorer -- generate a small dataset at a chosen
   redundancy level, run every explainer live, and inspect one
   explainer's ranking against the ground-truth usage map.
2. Omission by usage type -- per-usage-type omission rates from the
   stored headline evaluation (results/evaluation.json).
3. Evaluation -- the stored headline aggregates and caveats.

Run: streamlit run app.py
"""

import json
import os

import streamlit as st

from omission_audit.audit import audit_dataset
from omission_audit.explainers import EXPLAINER_NAMES, run_explainers
from omission_audit.generate import generate_dataset

st.set_page_config(page_title="Silent Feature Omission Audit", layout="wide")

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "evaluation.json"
)


def _load_evaluation():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as handle:
            return json.load(handle)
    return None


def tab_explorer():
    st.header("Explore an omission profile")
    st.write(
        "The data-generating process uses 6 features (3 main, 2 "
        "interaction-only, 1 subgroup-only). Each explainer gets a "
        "top-6 budget. A used feature outside the top-6 is *omitted*."
    )
    redundancy = st.selectbox("Planted redundant copies of main_a", (0, 1, 2), index=1)
    dataset = generate_dataset(
        seed=7, redundancy=int(redundancy), n_train=400, n_test=200
    )
    profiles = audit_dataset(dataset, run_explainers(dataset))
    explainer = st.selectbox("Explainer", EXPLAINER_NAMES, index=0)
    profile = profiles[explainer]
    usage = dataset["usage"]
    top_set = set(profile["top_k"])
    rows = []
    for rank, name in enumerate(profile["ranking"], start=1):
        rows.append({
            "rank": rank,
            "feature": name,
            "usage": usage[name],
            "in_top_k": "yes" if name in top_set else "no",
        })
    st.table(rows)
    st.write(
        "Omission rate (all used): %.1f%% -- omitted: %s"
        % (
            100.0 * profile["omission_rate"],
            ", ".join(profile["omitted"]) if profile["omitted"] else "none",
        )
    )


def tab_by_type():
    st.header("Omission by usage type (stored evaluation)")
    evaluation = _load_evaluation()
    if evaluation is None:
        st.write("No stored evaluation found at results/evaluation.json.")
        return
    redundancy = st.selectbox(
        "Redundancy level", evaluation["config"]["redundancy_levels"], index=0
    )
    rows = []
    for name in EXPLAINER_NAMES:
        cell = evaluation["cells"]["%d|%s" % (redundancy, name)]
        rows.append({
            "explainer": name,
            "all_used": round(cell["omission_rate"], 3),
            "main": round(cell["per_type"]["main"], 3),
            "interaction_only": round(cell["per_type"]["interaction_only"], 3),
            "subgroup_only": round(cell["per_type"]["subgroup_only"], 3),
            "commission": round(cell["commission_rate"], 3),
        })
    st.table(rows)
    st.bar_chart(
        dict(
            (row["explainer"], [row["interaction_only"]])
            for row in rows
        )
    )
    st.caption(
        "Bar chart: interaction-only omission rate per explainer at "
        "the selected redundancy level."
    )


def tab_evaluation():
    st.header("Headline evaluation")
    evaluation = _load_evaluation()
    if evaluation is None:
        st.write("No stored evaluation found at results/evaluation.json.")
        return
    st.json(evaluation["aggregates"])
    st.write(
        "Caveat: all data is synthetic and seeded, generated from the "
        "audit's own structural process. The numbers measure explainer "
        "behaviour under known ground truth -- which features each "
        "explainer silently drops, and how planted redundancy makes "
        "it worse -- not faithfulness on real datasets."
    )


tab1, tab2, tab3 = st.tabs(
    ["Omission explorer", "Omission by usage type", "Evaluation"]
)
with tab1:
    tab_explorer()
with tab2:
    tab_by_type()
with tab3:
    tab_evaluation()
