"""Notebook 02 (as a runnable script): reduced evaluation smoke run.

Runs the headline sweep with a single seed and smaller datasets so it
finishes in seconds, and prints the aggregate omission rates. The
canonical numbers in results/ come from the full config via
``python -m omission_audit evaluate``.

Run: python notebooks/02_evaluation_experiment.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omission_audit.evaluate import run_evaluation

SMOKE_CONFIG = {
    "seeds": (101,),
    "redundancy_levels": (0, 1, 2),
    "n_train": 400,
    "n_test": 200,
}


def main():
    evaluation = run_evaluation(SMOKE_CONFIG)
    print(json.dumps(evaluation["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
