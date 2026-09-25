from __future__ import annotations

import os
from pathlib import Path

from tcc_benchmark.bootstrap import (
    DEFAULT_DATASET_REGISTRY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SUITE_CONFIG,
    IMPORT_CATALOG,
    configure_training_runtime,
    load_global_configuration,
)


def test_bootstrap_exposes_the_cli_defaults_and_complete_import_inventory() -> None:
    assert DEFAULT_DATASET_REGISTRY == Path("configs/datasets.yaml")
    assert DEFAULT_SUITE_CONFIG == Path("configs/suite.yaml")
    assert DEFAULT_OUTPUT_ROOT == Path("artifacts")
    assert "tensorflow" in IMPORT_CATALOG["third_party"]
    assert "tcc_benchmark.runner" in IMPORT_CATALOG["local_package"]


def test_global_configuration_loads_and_resolves_paths(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    registry_path = tmp_path / "datasets.yaml"
    output_root = tmp_path / "output"
    suite_path.write_text("suite:\n  seeds: [42]\n", encoding="utf-8")
    registry_path.write_text(
        "datasets:\n  mnist:\n    adapter: mnist\n    root: datasets/mnist\n",
        encoding="utf-8",
    )

    configuration = load_global_configuration(
        suite_path=suite_path,
        registry_path=registry_path,
        output_root=output_root,
    )

    assert configuration.suite.output_root == output_root.resolve()
    assert configuration.suite_path == suite_path.resolve()
    assert configuration.registry_path == registry_path.resolve()
    assert configuration.datasets["mnist"].root == (tmp_path / "datasets" / "mnist").resolve()


def test_training_runtime_configuration_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)
    configure_training_runtime()
    assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
