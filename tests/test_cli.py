import json
import os

from omission_audit.cli import build_parser, demo_dataset, main


def test_demo_dataset_spec():
    dataset = demo_dataset()
    assert dataset["redundancy"] == 1
    assert len(dataset["feature_names"]) == 12


def test_cli_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Omission audit report" in out


def test_cli_no_command_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "omission" in out.lower()


def test_cli_evaluate_writes_results(tmp_path, capsys):
    outdir = str(tmp_path / "results")
    assert main(["evaluate", "--outdir", outdir]) == 0
    assert os.path.exists(os.path.join(outdir, "evaluation.json"))
    with open(os.path.join(outdir, "evaluation.json")) as handle:
        evaluation = json.load(handle)
    assert "aggregates" in evaluation


def test_cli_audit_on_demo_dataset(tmp_path, capsys):
    from omission_audit.generate import generate_dataset, save_dataset

    path = str(tmp_path / "dataset.json")
    save_dataset(
        generate_dataset(seed=7, redundancy=0, n_train=200, n_test=100), path
    )
    assert main(["audit", "--dataset", path]) == 0
    out = capsys.readouterr().out
    assert "Omission audit report" in out


def test_parser_commands():
    parser = build_parser()
    args = parser.parse_args(["demo"])
    assert args.command == "demo"
    args = parser.parse_args(["evaluate", "--outdir", "x"])
    assert args.command == "evaluate"
    assert args.outdir == "x"
