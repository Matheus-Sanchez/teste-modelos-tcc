from __future__ import annotations

import csv
import json
from pathlib import Path

from tcc_benchmark.reporting import build_dataset_report, build_global_index, compute_classification_metrics, write_run_report
from tcc_benchmark.state import (
    ManifestCompatibilityError,
    append_csv,
    append_jsonl,
    initialize_run,
    read_json,
    read_jsonl,
    set_run_status,
    update_manifest,
)
from tcc_benchmark.telemetry import TelemetrySampler, make_keras_telemetry_callback


def _config() -> dict[str, object]:
    return {"dataset": "mnist", "normalization": "unit_interval", "balance_mode": "all_raw", "seed": 42}


def test_state_is_atomic_and_resume_validated(tmp_path: Path) -> None:
    run_dir = tmp_path / "mnist" / "runs" / "mnist__unit_interval__all_raw__seed-42"
    manifest = initialize_run(run_dir, _config(), split_fingerprint="split-a")
    assert manifest["status"] == "pending"
    assert set_run_status(run_dir, "running")["attempt"] == 1
    assert set_run_status(run_dir, "interrupted")["status"] == "interrupted"
    assert initialize_run(run_dir, _config(), split_fingerprint="split-a")["run_id"].endswith("seed-42")

    changed = dict(_config(), seed=43)
    try:
        initialize_run(run_dir, changed, split_fingerprint="split-a")
    except ManifestCompatibilityError:
        pass
    else:  # pragma: no cover - makes the expected compatibility policy explicit
        raise AssertionError("resume with changed configuration should fail")

    append_jsonl(run_dir / "logs" / "records.jsonl", {"epoch": 1})
    append_csv(run_dir / "logs" / "epochs.csv", [{"epoch": 1, "loss": 0.2}], fieldnames=("epoch", "loss"))
    assert read_jsonl(run_dir / "logs" / "records.jsonl") == [{"epoch": 1}]


def test_telemetry_works_without_requiring_gpu(tmp_path: Path) -> None:
    sampler = TelemetrySampler(tmp_path / "telemetry", interval_seconds=0.01, data_path=tmp_path)
    sampler.start()
    sampler.capture("epoch_end", epoch=1, epoch_seconds=0.01)
    summary = sampler.stop()
    assert summary["sample_count"] >= 2
    assert (tmp_path / "telemetry" / "samples.csv").exists()
    assert json.loads((tmp_path / "telemetry" / "summary.json").read_text(encoding="utf-8"))["sample_count"] >= 2


def test_unmanaged_keras_callback_keeps_telemetry_running_for_evaluation(tmp_path: Path) -> None:
    sampler = TelemetrySampler(tmp_path / "telemetry", interval_seconds=0.01)
    sampler.start()
    callback = make_keras_telemetry_callback(sampler, manage_lifecycle=False)
    callback.on_train_begin()
    callback.on_train_end({"loss": 0.2})
    assert sampler.running
    summary = sampler.stop()
    with (tmp_path / "telemetry" / "samples.csv").open(encoding="utf-8", newline="") as stream:
        events = [row["event"] for row in csv.DictReader(stream)]
    assert "fit_end" in events
    assert summary["sample_count"] >= 3


def test_reports_include_class_metrics_and_dataset_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "mnist" / "runs" / "mnist__unit_interval__all_raw__seed-42"
    initialize_run(run_dir, _config())
    update_manifest(
        run_dir,
        {
            "data_metadata": {
                "balancing": {
                    "is_noop": True,
                    "raw_class_counts": {0: 2, 1: 2, 2: 2},
                    "output_class_counts": {0: 2, 1: 2, 2: 2},
                }
            }
        },
    )
    set_run_status(run_dir, "running")
    metrics = compute_classification_metrics([0, 1, 1, 2], [0, 1, 0, 2], class_names=["zero", "one", "two"])
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"][1]["support"] == 2

    artifacts = write_run_report(
        run_dir,
        y_true=[0, 1, 1, 2],
        y_pred=[0, 1, 0, 2],
        class_names=["zero", "one", "two"],
        test_metrics={"loss": 0.2},
        history={"loss": [1.0, 0.5], "accuracy": [0.5, 0.75]},
        training_summary={"total_seconds": 4.0, "mean_epoch_seconds": 2.0},
    )
    set_run_status(run_dir, "completed")
    assert artifacts["confusion_png"].exists()
    assert artifacts["html"].exists()

    dataset = build_dataset_report(tmp_path / "mnist")
    index = build_global_index(tmp_path)
    assert dataset["summary_csv"].exists()
    assert read_json(dataset["summary_json"])["runs"][0]["balance_no_op"] is True
    assert index["html"].exists()
