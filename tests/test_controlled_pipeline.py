from __future__ import annotations

import copy
import importlib.util
import json
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


def test_quantization_only_plan_forces_batch_256_without_batch_sweep(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((project_root / "configs" / "controlled-augmentation2-mac-m4.yaml").read_text(encoding="utf-8"))
    registry = project_root / "configs" / "datasets.yaml"

    batches = pipeline.forced_quantization_inputs(batch_size=256)
    quantization = pipeline.run_stage_quant(
        copy.deepcopy(base),
        tmp_path,
        registry,
        batches,
        resume=False,
        dry_run=True,
    )

    assert len(quantization["rows"]) == 27  # 18 float training plans + 9 PTQ plans
    assert {row.get("batch_size", 256) for row in quantization["rows"]} == {256}
    assert {row["variant"] for row in quantization["rows"]} == {"fp32", "fp16", "int8_ptq"}
    assert {row["variant"] for row in quantization["winners"].values()} == {"fp32"}


def test_quantization_stage_skips_litert_and_ptq_after_completed_training(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((project_root / "configs" / "controlled-augmentation2-mac-m4.yaml").read_text(encoding="utf-8"))
    registry = project_root / "configs" / "datasets.yaml"
    batches = pipeline.forced_quantization_inputs(batch_size=256)
    export_calls: list[tuple[object, ...]] = []

    def fake_execute(dataset, suite_path, registry_path, *, resume, dry_run):
        suite = pipeline.read_suite(suite_path)
        output = Path(suite["suite"]["output_root"])
        current = pipeline.run_root(output, dataset)
        pipeline.atomic_write_json(current / "status.json", {"status": "completed"})
        pipeline.atomic_write_json(current / "artifacts" / "test_metrics.json", {"classification": {"macro_f1": 0.5, "accuracy": 0.6}})
        pipeline.atomic_write_json(current / "logs" / "training_summary.json", {"mean_epoch_seconds": 1.0})
        (current / "checkpoints").mkdir(parents=True, exist_ok=True)
        (current / "checkpoints" / "best.keras").write_bytes(b"checkpoint")
        return 0

    def fail_litert(*args, **kwargs):
        export_calls.append(args)
        raise AssertionError("LiteRT não deve ser chamado quando --skip-litert está ativo.")

    monkeypatch.setattr(pipeline, "execute_training", fake_execute)
    monkeypatch.setattr(pipeline, "export_litert", fail_litert)
    monkeypatch.setattr(pipeline, "ptq_row", fail_litert)
    quantization = pipeline.run_stage_quant(
        copy.deepcopy(base),
        tmp_path,
        registry,
        batches,
        resume=True,
        dry_run=False,
        skip_litert=True,
    )

    assert len(quantization["rows"]) == len(pipeline.DATASETS) * 2
    assert not export_calls
    assert {row["status"] for row in quantization["rows"]} == {"completed"}
    assert {row["variant"] for row in quantization["rows"]} == {"fp32", "fp16"}
    assert {row["litert_status"] for row in quantization["rows"]} == {"skipped"}
    assert quantization["winners"] == {}
    report = pipeline.read_json(tmp_path / "reports" / "quantization_comparison.json")
    assert {row["litert_status"] for row in report["rows"]} == {"skipped"}


def test_activation_stage_skips_litert_after_completed_training(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((project_root / "configs" / "controlled-augmentation2-mac-m4.yaml").read_text(encoding="utf-8"))
    registry = project_root / "configs" / "datasets.yaml"
    batches, quant = pipeline.forced_activation_inputs(batch_size=256, dtype_policy="mixed_float16")
    export_calls: list[tuple[object, ...]] = []

    def fake_execute(dataset, suite_path, registry_path, *, resume, dry_run):
        suite = pipeline.read_suite(suite_path)
        output = Path(suite["suite"]["output_root"])
        current = pipeline.run_root(output, dataset)
        pipeline.atomic_write_json(current / "status.json", {"status": "completed"})
        pipeline.atomic_write_json(current / "artifacts" / "test_metrics.json", {"classification": {"macro_f1": 0.5, "accuracy": 0.6}})
        pipeline.atomic_write_json(current / "logs" / "training_summary.json", {"mean_epoch_seconds": 1.0})
        return 0

    def fail_export(*args, **kwargs):
        export_calls.append(args)
        raise AssertionError("LiteRT não deve ser chamado quando --skip-litert está ativo.")

    monkeypatch.setattr(pipeline, "execute_training", fake_execute)
    monkeypatch.setattr(pipeline, "export_litert", fail_export)
    activation = pipeline.run_stage_activation(
        copy.deepcopy(base),
        tmp_path,
        registry,
        batches,
        quant,
        resume=True,
        dry_run=False,
        activations=("relu",),
        skip_litert=True,
    )

    assert len(activation["rows"]) == len(pipeline.DATASETS)
    assert not export_calls
    assert {row["status"] for row in activation["rows"]} == {"completed"}
    assert {row["litert_status"] for row in activation["rows"]} == {"skipped"}
    report = pipeline.read_json(tmp_path / "reports" / "activation_comparison.json")
    assert {row["litert_status"] for row in report["rows"]} == {"skipped"}


def test_cli_parsers_reject_invalid_or_duplicated_candidates() -> None:
    assert pipeline.parse_batch_sizes("256") == (256,)
    assert pipeline.parse_activations("relu,sigmoid,softmax") == ("relu", "sigmoid", "softmax")
    try:
        pipeline.parse_batch_sizes("128,128")
    except Exception as exc:
        assert "sem repetição" in str(exc)
    else:  # pragma: no cover - documents the required parser contract.
        raise AssertionError("batch duplicado deveria ser rejeitado")


def test_batch_only_cli_skips_quantization_and_activation(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry = project_root / "configs" / "datasets.yaml"
    suite = project_root / "configs" / "controlled-augmentation2-mac-m4.yaml"
    output = tmp_path / "batch-only"

    monkeypatch.setattr(pipeline, "_run_gates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "run_stage_batch",
        lambda *args, **kwargs: {"rows": [{"dataset": "mnist", "batch_size": "256"}], "winners": {}},
    )
    monkeypatch.setattr(
        pipeline,
        "run_stage_quant",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("quantização não deveria rodar")),
    )
    monkeypatch.setattr(
        pipeline,
        "run_stage_activation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ativação não deveria rodar")),
    )
    monkeypatch.setattr(pipeline, "_write_dataset_phase_reports", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "write_phase_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline.sys,
        "argv",
        [
            "run_controlled_pipeline.py",
            "--suite",
            str(suite),
            "--registry",
            str(registry),
            "--output-root",
            str(output),
            "--dry-run",
            "--batch-sizes",
            "256",
            "--skip-quantization",
            "--skip-activation",
        ],
    )

    assert pipeline.main() == 0
    state = json.loads((output / "pipeline-status.json").read_text(encoding="utf-8"))
    assert state["status"] == "planned_batch_only"
    assert state["stages"]["quantization"]["status"] == "skipped"
    assert state["stages"]["activation"]["status"] == "skipped"
    assert state["counts"] == {
        "batch_training_runs": 1,
        "quantization_training_runs": 0,
        "activation_training_runs": 0,
        "total_training_runs": 1,
    }
