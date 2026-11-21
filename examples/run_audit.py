"""Audit the bundled demo dataset and print the omission report.

Run: python examples/run_audit.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omission_audit.audit import audit_dataset
from omission_audit.explainers import run_explainers
from omission_audit.generate import load_dataset
from omission_audit.report import audit_report_markdown, ranking_markdown


def main():
    path = os.path.join(os.path.dirname(__file__), "demo_dataset.json")
    dataset = load_dataset(path)
    profiles = audit_dataset(dataset, run_explainers(dataset))
    print(audit_report_markdown(dataset, profiles))
    print(ranking_markdown(dataset, profiles, "permutation"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
