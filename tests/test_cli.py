from __future__ import annotations

from tcc_benchmark.cli import build_parser


def test_cli_knows_required_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["preflight"]).command == "preflight"
    audit = parser.parse_args(["audit", "--dataset", "mnist"])
    assert audit.dataset == "mnist"
    assert str(audit.registry).replace("\\", "/") == "configs/datasets.yaml"
    run = parser.parse_args(["run", "--all", "--dry-run"])
    assert run.all and run.dry_run
