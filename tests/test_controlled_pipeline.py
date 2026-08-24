from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import yaml


_PIPELINE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_controlled_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("run_controlled_pipeline", _PIPELINE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pipeline)


def test_matrix_has_exact_serial_counts() -> None:
    assert len(pipeline.DATASETS) == 9
    assert len(pipeline.DATASETS) * len(pipeline.BATCH_SIZES) == 36
    assert len(pipeline.DATASETS) * 2 == 18
    assert len(pipeline.DATASETS) * len(pipeline.ACTIVATIONS) == 27


def test_dry_run_materializes_all_three_phases_without_training(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((project_root / "configs" / "controlled-augmentation2-mac-m4.yaml").read_text(encoding="utf-8"))
    registry = project_root / "configs" / "datasets.yaml"
    calls: list[tuple[str, bool]] = []

    def fake_execute(dataset, suite_path, registry_path, *, resume, dry_run):
        calls.append((dataset, dry_run))
        return 0

    monkeypatch.setattr(pipeline, "execute_training", fake_execute)
    batch = pipeline.run_stage_batch(copy.deepcopy(base), tmp_path, registry, resume=False, dry_run=True)
    quant = pipeline.run_stage_quant(copy.deepcopy(base), tmp_path, registry, batch["winners"], resume=False, dry_run=True)
    activation = pipeline.run_stage_activation(
        copy.deepcopy(base), tmp_path, registry, batch["winners"], quant["winners"], resume=False, dry_run=True
    )

    assert len(batch["rows"]) == 36
    assert len(quant["rows"]) == 27  # 18 float training plans + 9 PTQ plans
    assert len(activation["rows"]) == 27
    assert len(calls) == 36
    assert all(dry_run for _, dry_run in calls)
    assert all("cinic10" not in row.get("dataset", "") for row in batch["rows"] + quant["rows"] + activation["rows"])


def test_fast_batch_subset_plans_only_batch_256(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((project_root / "configs" / "controlled-augmentation2-mac-m4.yaml").read_text(encoding="utf-8"))
    registry = project_root / "configs" / "datasets.yaml"
    calls: list[tuple[str, bool]] = []

    def fake_execute(dataset, suite_path, registry_path, *, resume, dry_run):
        calls.append((dataset, dry_run))
        return 0

    monkeypatch.setattr(pipeline, "execute_training", fake_execute)
    batch = pipeline.run_stage_batch(
        copy.deepcopy(base), tmp_path, registry, resume=False, dry_run=True, batch_sizes=(256,)
    )

    assert len(batch["rows"]) == 9
    assert len(calls) == 9
    assert {row["batch_size"] for row in batch["rows"]} == {"256"}


def test_activation_only_plan_forces_batch_256_and_fp16(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((project_root / "configs" / "controlled-augmentation2-mac-m4.yaml").read_text(encoding="utf-8"))
    registry = project_root / "configs" / "datasets.yaml"
    batches, quant = pipeline.forced_activation_inputs(batch_size=256, dtype_policy="mixed_float16")

    activation = pipeline.run_stage_activation(
        copy.deepcopy(base),
        tmp_path,
        registry,
        batches,
        quant,
        resume=False,
        dry_run=True,
    )

    assert len(activation["rows"]) == 27
    assert {row["selected_quantization"] for row in activation["rows"]} == {"fp16"}
    suite = yaml.safe_load((tmp_path / "generated-suites" / "activations" / "mnist-relu.yaml").read_text(encoding="utf-8"))
    assert suite["suite"]["training"]["batch_size"] == 256
    assert suite["suite"]["training"]["dtype_policy"] == "mixed_float16"
    assert suite["suite"]["training"]["hidden_activation"] == "relu"


def test_cli_parsers_reject_invalid_or_duplicated_candidates() -> None:
    assert pipeline.parse_batch_sizes("256") == (256,)
    assert pipeline.parse_activations("relu,sigmoid,softmax") == ("relu", "sigmoid", "softmax")
    try:
        pipeline.parse_batch_sizes("128,128")
    except Exception as exc:
        assert "sem repetição" in str(exc)
    else:  # pragma: no cover - documents the required parser contract.
        raise AssertionError("batch duplicado deveria ser rejeitado")
