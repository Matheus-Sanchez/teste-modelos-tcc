#!/usr/bin/env python3
"""Export portable, path-scrubbed evidence for the Aug-0.5 and quantization campaigns.

The large run directories remain under ignored ``outputs/`` on the Mac. This
script reads only completed runs and exports metrics, 100-epoch histories,
classwise test scores, and aggregate hardware telemetry; it never copies model
weights, predictions, raw telemetry samples, or machine-specific file paths.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis_reports/activation_quantization_2026-09-30/data"
ACT_ROOT = REPO / "outputs/controlled-augmentation05-activations-mac2/activations"
QUANT_ROOT = REPO / "outputs/controlled-quantization-fast-mac-m4/quantization"
GIB = 1024**3

RUN_FIELDS = [
    "campaign", "stage", "dataset", "condition", "run_id", "status", "status_updated_at",
    "seed", "normalization", "balance_mode", "epochs_completed", "max_epochs", "best_epoch",
    "batch_size", "learning_rate", "extra_fraction", "dtype_policy", "optimizer",
    "test_samples", "num_classes", "test_accuracy", "test_balanced_accuracy",
    "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_macro_ovr_auc", "test_loss",
    "training_seconds", "evaluation_seconds", "wall_seconds", "mean_epoch_seconds",
    "mean_train_examples_per_second", "ptq_model_bytes", "ptq_batch_size",
    "ptq_median_batch_latency_ms", "ptq_p95_pass_seconds", "ptq_throughput_examples_per_second",
    "gpu_backend", "gpu_name", "gpu_core_count", "gpu_memory_kind", "telemetry_samples",
    "telemetry_interval_seconds", "cpu_mean_percent", "cpu_p95_percent", "process_cpu_mean_percent",
    "process_cpu_p95_percent", "process_cpu_max_percent", "gpu_mean_percent", "gpu_p95_percent",
    "gpu_max_percent", "gpu_memory_mean_gib", "gpu_memory_p95_gib", "gpu_memory_max_gib",
    "process_rss_mean_gib", "process_rss_p95_gib", "process_rss_max_gib",
    "ram_mean_percent", "ram_p95_percent", "ram_max_percent", "thermal_pressure",
    "system_ram_gib", "macos_version", "metal_version", "tensorflow_version", "tensorflow_metal_version",
    "python_version", "source_ref",
]

EPOCH_FIELDS = [
    "campaign", "dataset", "condition", "run_id", "epoch", "loss", "accuracy", "val_loss",
    "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "learning_rate", "epoch_seconds",
    "train_examples", "train_examples_per_second",
]


def read_json(path: Path, optional: bool = False) -> dict[str, Any]:
    if not path.exists():
        if optional:
            return {}
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def stat(summary: dict[str, Any], metric: str, statistic: str) -> Any:
    return value(summary, "metrics", metric, statistic)


def gib(value_bytes: Any) -> float | None:
    return round(value_bytes / GIB, 5) if isinstance(value_bytes, (int, float)) else None


def as_int(value_: Any, default: int = 0) -> int:
    try:
        return int(value_)
    except (TypeError, ValueError):
        return default


def export_training_run(
    *, campaign: str, stage: str, condition: str, dataset: str,
    status_file: Path, relative_ref: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run = status_file.parent
    status = read_json(status_file)
    if status.get("status") != "completed":
        raise RuntimeError(f"Refusing to export non-completed run: {relative_ref} ({status.get('status')})")

    manifest = read_json(run / "manifest.json")
    config = manifest.get("config", {})
    train_cfg = config.get("training", {})
    protocol = config.get("protocol", {})
    metrics = read_json(run / "artifacts/test_metrics.json")
    classification = metrics.get("classification", {})
    keras_metrics = metrics.get("keras_metrics", {})
    training = read_json(run / "logs/training_summary.json")
    telemetry = read_json(run / "telemetry/summary.json")
    environment = read_json(run / "telemetry/environment.json")
    hardware = environment.get("hardware", {})
    gpu = (hardware.get("gpus") or [{}])[0]
    packages = environment.get("packages", {})
    telem_metrics = telemetry.get("metrics", {})
    epoch_rows: list[dict[str, Any]] = []
    epoch_path = run / "logs/epoch_metrics.csv"
    with epoch_path.open(newline="", encoding="utf-8") as f:
        for source_row in csv.DictReader(f):
            row = {
                "campaign": campaign,
                "dataset": dataset,
                "condition": condition,
                "run_id": manifest.get("run_id", status.get("run_id", "")),
            }
            for key in EPOCH_FIELDS[4:]:
                raw = source_row.get(key)
                row[key] = raw if raw not in (None, "") else None
            epoch_rows.append(row)

    val_f1 = [float(row["val_macro_f1"]) for row in epoch_rows if row.get("val_macro_f1") not in (None, "")]
    supports = [item.get("support", 0) for item in classification.get("per_class", {}).values()]
    run_id = manifest.get("run_id", status.get("run_id", ""))
    row: dict[str, Any] = {
        "campaign": campaign,
        "stage": stage,
        "dataset": dataset,
        "condition": condition,
        "run_id": run_id,
        "status": status.get("status"),
        "status_updated_at": status.get("status_updated_at") or status.get("updated_at"),
        "seed": config.get("seed"),
        "normalization": config.get("normalization"),
        "balance_mode": config.get("balance_mode"),
        "epochs_completed": training.get("epochs_completed"),
        "max_epochs": training.get("max_epochs"),
        "best_epoch": (int(max(epoch_rows, key=lambda r: float(r["val_macro_f1"]))["epoch"])
                       if val_f1 else None),
        "batch_size": train_cfg.get("batch_size"),
        "learning_rate": train_cfg.get("learning_rate"),
        "extra_fraction": train_cfg.get("extra_fraction"),
        "dtype_policy": train_cfg.get("dtype_policy") or protocol.get("dtype_policy"),
        "optimizer": protocol.get("optimizer"),
        "test_samples": sum(as_int(x) for x in supports),
        "num_classes": config.get("num_classes"),
        "test_accuracy": classification.get("accuracy"),
        "test_balanced_accuracy": classification.get("balanced_accuracy"),
        "test_macro_precision": classification.get("macro_precision"),
        "test_macro_recall": classification.get("macro_recall"),
        "test_macro_f1": classification.get("macro_f1"),
        "test_macro_ovr_auc": classification.get("macro_ovr_auc"),
        "test_loss": keras_metrics.get("loss"),
        "training_seconds": training.get("training_seconds"),
        "evaluation_seconds": training.get("evaluation_seconds"),
        "wall_seconds": training.get("wall_seconds_current_attempt"),
        "mean_epoch_seconds": training.get("mean_epoch_seconds"),
        "mean_train_examples_per_second": training.get("mean_train_examples_per_second"),
        "gpu_backend": telemetry.get("gpu_backend"),
        "gpu_name": gpu.get("name"),
        "gpu_core_count": gpu.get("gpu_core_count"),
        "gpu_memory_kind": gpu.get("memory_kind"),
        "telemetry_samples": telemetry.get("sample_count"),
        "telemetry_interval_seconds": telemetry.get("interval_seconds"),
        "cpu_mean_percent": stat(telemetry, "cpu_percent", "mean"),
        "cpu_p95_percent": stat(telemetry, "cpu_percent", "p95"),
        "process_cpu_mean_percent": stat(telemetry, "process_cpu_percent", "mean"),
        "process_cpu_p95_percent": stat(telemetry, "process_cpu_percent", "p95"),
        "process_cpu_max_percent": stat(telemetry, "process_cpu_percent", "max"),
        "gpu_mean_percent": stat(telemetry, "gpu_utilization_percent", "mean"),
        "gpu_p95_percent": stat(telemetry, "gpu_utilization_percent", "p95"),
        "gpu_max_percent": stat(telemetry, "gpu_utilization_percent", "max"),
        "gpu_memory_mean_gib": gib(stat(telemetry, "gpu_memory_used_bytes", "mean")),
        "gpu_memory_p95_gib": gib(stat(telemetry, "gpu_memory_used_bytes", "p95")),
        "gpu_memory_max_gib": gib(stat(telemetry, "gpu_memory_used_bytes", "max")),
        "process_rss_mean_gib": gib(stat(telemetry, "process_rss_bytes", "mean")),
        "process_rss_p95_gib": gib(stat(telemetry, "process_rss_bytes", "p95")),
        "process_rss_max_gib": gib(stat(telemetry, "process_rss_bytes", "max")),
        "ram_mean_percent": stat(telemetry, "ram_percent", "mean"),
        "ram_p95_percent": stat(telemetry, "ram_percent", "p95"),
        "ram_max_percent": stat(telemetry, "ram_percent", "max"),
        "thermal_pressure": value(telemetry, "categorical", "thermal_pressure", "current"),
        "system_ram_gib": gib(hardware.get("ram_total_bytes")),
        "macos_version": environment.get("macos_version"),
        "metal_version": environment.get("metal_version"),
        "tensorflow_version": value(environment, "packages", "tensorflow"),
        "tensorflow_metal_version": value(environment, "packages", "tensorflow_metal"),
        "python_version": value(environment, "python", "version"),
        "source_ref": relative_ref,
    }

    per_class_rows = []
    class_names = config.get("class_names", [])
    for class_key, class_metrics in classification.get("per_class", {}).items():
        label = as_int(class_metrics.get("label", class_key))
        per_class_rows.append({
            "campaign": campaign, "stage": stage, "dataset": dataset, "condition": condition,
            "run_id": run_id, "class_label": label,
            "class_name": class_names[label] if 0 <= label < len(class_names) else str(class_key),
            "precision": class_metrics.get("precision"), "recall": class_metrics.get("recall"),
            "f1": class_metrics.get("f1"), "support": class_metrics.get("support"),
        })
    return row, epoch_rows, per_class_rows


def export_ptq_run(status_file: Path, dataset: str, relative_ref: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = status_file.parent
    status = read_json(status_file)
    if status.get("status") != "completed":
        raise RuntimeError(f"Refusing to export non-completed PTQ run: {relative_ref}")
    benchmark = read_json(run / "artifacts/litert_benchmark.json")
    metadata = read_json(run / "artifacts/int8_ptq.tflite.metadata.json")
    classification = benchmark.get("classification", {})
    supports = [item.get("support", 0) for item in classification.get("per_class", {}).values()]
    row = {
        "campaign": "quantization", "stage": "post_training_quantization_evaluation",
        "dataset": dataset, "condition": "int8_ptq", "run_id": status_file.parent.name,
        "status": status.get("status"), "status_updated_at": status.get("updated_at"),
        "seed": 42, "normalization": "unit_interval", "balance_mode": "all_raw",
        "epochs_completed": None, "max_epochs": None, "best_epoch": None,
        "test_samples": sum(as_int(x) for x in supports),
        "test_accuracy": classification.get("accuracy"),
        "test_balanced_accuracy": classification.get("balanced_accuracy"),
        "test_macro_precision": classification.get("macro_precision"),
        "test_macro_recall": classification.get("macro_recall"),
        "test_macro_f1": classification.get("macro_f1"),
        "test_macro_ovr_auc": classification.get("macro_ovr_auc"),
        "ptq_model_bytes": metadata.get("bytes"),
        "ptq_batch_size": benchmark.get("batch_size"),
        "ptq_median_batch_latency_ms": benchmark.get("median_batch_latency_ms"),
        "ptq_p95_pass_seconds": benchmark.get("p95_pass_seconds"),
        "ptq_throughput_examples_per_second": benchmark.get("throughput_examples_per_second"),
        "source_ref": relative_ref,
    }
    # PTQ post-processing has no run manifest; the benchmark labels are numeric.
    per_class_rows = []
    for class_key, class_metrics in classification.get("per_class", {}).items():
        label = as_int(class_metrics.get("label", class_key))
        per_class_rows.append({
            "campaign": "quantization", "stage": "post_training_quantization_evaluation",
            "dataset": dataset, "condition": "int8_ptq", "run_id": row["run_id"],
            "class_label": label, "class_name": str(label),
            "precision": class_metrics.get("precision"), "recall": class_metrics.get("recall"),
            "f1": class_metrics.get("f1"), "support": class_metrics.get("support"),
        })
    return row, per_class_rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    activation_statuses = sorted(ACT_ROOT.rglob("status.json"))
    for status_file in activation_statuses:
        parts = status_file.relative_to(ACT_ROOT).parts
        if len(parts) < 6 or "smoke" in parts:
            continue
        dataset, condition = parts[0], parts[1]
        run, history, classes = export_training_run(
            campaign="activation", stage="training", condition=condition, dataset=dataset,
            status_file=status_file, relative_ref=status_file.relative_to(REPO).as_posix(),
        )
        runs.append(run); epochs.extend(history); per_class.extend(classes)

    quant_statuses = sorted(QUANT_ROOT.rglob("status.json"))
    for status_file in quant_statuses:
        parts = status_file.relative_to(QUANT_ROOT).parts
        if len(parts) < 6 or "smoke" in parts:
            continue
        dataset, condition = parts[0], parts[1]
        relative_ref = status_file.relative_to(REPO).as_posix()
        if condition in {"fp32", "fp16"}:
            run, history, classes = export_training_run(
                campaign="quantization", stage="training", condition=condition, dataset=dataset,
                status_file=status_file, relative_ref=relative_ref,
            )
            runs.append(run); epochs.extend(history); per_class.extend(classes)
        elif condition == "int8_ptq":
            run, classes = export_ptq_run(status_file, dataset, relative_ref)
            runs.append(run); per_class.extend(classes)

    expected = {"activation": 27, "quantization": 27}
    counts = {key: sum(row["campaign"] == key for row in runs) for key in expected}
    training_rows = [row for row in runs if row["stage"] == "training"]
    if counts != expected or len(training_rows) != 45 or len(epochs) != 4500:
        raise RuntimeError(
            f"Unexpected campaign coverage: counts={counts}, training={len(training_rows)}, epochs={len(epochs)}"
        )

    write_csv(OUT / "runs.csv", RUN_FIELDS, runs)
    write_csv(OUT / "epoch_history.csv", EPOCH_FIELDS, epochs)
    write_csv(OUT / "per_class_metrics.csv", [
        "campaign", "stage", "dataset", "condition", "run_id", "class_label", "class_name",
        "precision", "recall", "f1", "support",
    ], per_class)
    summary = {
        "schema_version": 1,
        "snapshot_date": "2026-09-30",
        "source_scope": {
            "activation": "outputs/controlled-augmentation05-activations-mac2/activations",
            "quantization": "outputs/controlled-quantization-fast-mac-m4/quantization",
        },
        "campaign_counts": counts,
        "full_training_runs": len(training_rows),
        "full_training_epochs": len(epochs),
        "ptq_evaluation_runs": sum(row["stage"] == "post_training_quantization_evaluation" for row in runs),
        "classwise_test_rows": len(per_class),
        "exported_files": ["runs.csv", "epoch_history.csv", "per_class_metrics.csv"],
        "excluded_artifacts": ["model checkpoints", "predictions/logits", "raw telemetry samples", "local absolute paths"],
    }
    (OUT / "snapshot_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
