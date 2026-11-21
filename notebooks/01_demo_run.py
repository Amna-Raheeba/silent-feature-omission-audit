"""Notebook 01 (as a runnable script): demo audit on one dataset.

Generates a small redundancy-1 dataset, runs every explainer, and
prints the omission profile plus the permutation ranking.

Run: python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omission_audit.audit import audit_dataset
from omission_audit.cli import demo_dataset
from omission_audit.explainers import run_explainers
from omission_audit.report import audit_report_markdown, ranking_markdown


def main():
    dataset = demo_dataset()
    profiles = audit_dataset(dataset, run_explainers(dataset))
    print(audit_report_markdown(dataset, profiles))
    print(ranking_markdown(dataset, profiles, "permutation"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
