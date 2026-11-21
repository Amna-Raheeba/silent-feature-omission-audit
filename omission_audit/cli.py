"""Command line interface.

Usage:
    python -m omission_audit demo
    python -m omission_audit audit --dataset data/dataset_r1.json
    python -m omission_audit evaluate --outdir results
"""

import argparse
import json
import os

from .audit import audit_dataset
from .evaluate import run_evaluation, write_results
from .explainers import run_explainers
from .generate import generate_dataset, load_dataset
from .report import audit_report_markdown, ranking_markdown

DEMO_SEED = 7


def demo_dataset():
    """One small dataset per redundancy level, for the built-in demo."""
    return generate_dataset(seed=DEMO_SEED, redundancy=1, n_train=400, n_test=200)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="omission_audit",
        description=(
            "Audit post-hoc explainers for silent feature omission: "
            "score each explainer's top-k against the known "
            "data-generating usage of every feature."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Audit one small dataset and report")

    audit_parser = sub.add_parser(
        "audit", help="Audit a stored dataset JSON with ground truth"
    )
    audit_parser.add_argument("--dataset", required=True)

    eval_parser = sub.add_parser(
        "evaluate", help="Run the headline redundancy sweep"
    )
    eval_parser.add_argument("--outdir", default="results")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        dataset = demo_dataset()
        profiles = audit_dataset(dataset, run_explainers(dataset))
        print(audit_report_markdown(dataset, profiles))
        print(ranking_markdown(dataset, profiles, "permutation"))
        return 0

    if args.command == "audit":
        dataset = load_dataset(args.dataset)
        profiles = audit_dataset(dataset, run_explainers(dataset))
        print(audit_report_markdown(dataset, profiles))
        return 0

    if args.command == "evaluate":
        evaluation = run_evaluation()
        write_results(evaluation, args.outdir)
        print(
            "Wrote evaluation results to %s" % os.path.abspath(args.outdir)
        )
        print(json.dumps(evaluation["aggregates"], indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
