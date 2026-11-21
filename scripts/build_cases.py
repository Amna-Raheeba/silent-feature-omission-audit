"""Build the stored datasets under data/ and examples/.

Run: python scripts/build_cases.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omission_audit.generate import generate_dataset, save_dataset
from omission_audit.models import DGP_PARAMS, usage_map

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    full = generate_dataset(seed=101, redundancy=1, n_train=800, n_test=400)
    save_dataset(full, os.path.join(ROOT, "data", "dataset_r1.json"))
    demo = generate_dataset(seed=7, redundancy=1, n_train=240, n_test=120)
    save_dataset(demo, os.path.join(ROOT, "examples", "demo_dataset.json"))
    spec = {
        "dgp_params": DGP_PARAMS,
        "usage_by_redundancy": dict(
            (str(level), usage_map(level)) for level in (0, 1, 2)
        ),
        "stored_datasets": {
            "data/dataset_r1.json": {
                "seed": 101, "redundancy": 1, "n_train": 800, "n_test": 400,
            },
            "examples/demo_dataset.json": {
                "seed": 7, "redundancy": 1, "n_train": 240, "n_test": 120,
            },
        },
    }
    with open(os.path.join(ROOT, "data", "dgp_spec.json"), "w") as handle:
        json.dump(spec, handle, indent=2)
    print("Wrote data/dataset_r1.json, data/dgp_spec.json, "
          "examples/demo_dataset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
