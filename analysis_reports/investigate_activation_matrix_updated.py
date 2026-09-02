"""Investigate the updated activation matrix and compare it with the prior run.

The updated matrix changes ``extra_fraction`` from 2.0 to 0.5 while keeping
the activation protocol, split, optimizer, model and evaluation procedure
otherwise comparable. This script reads completed artifacts from both roots,
recomputes concentration/logit diagnostics, and probes hidden activations for
the completed MNIST and Fashion-MNIST runs in the updated root.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis_reports" / "investigacao_ativacoes_atualizada.json"
ACTIVATIONS = ("relu", "sigmoid", "softmax")
DATASETS = ("mnist", "fashion_mnist", "kmnist")
PROBE_DATASETS = ("mnist", "fashion_mnist")
PROBE_LAYERS = (
    "block1_activation1",
    "block3_activation1",
    "block5_activation2",
    "dense1",
    "dense2",
    "logits",
)
ROOT_SPECS = (
    {
        "root_id": "augmentation_2_0",
        "label": "Lote anterior — extra_fraction=2,0",
        "path": ROOT / "outputs" / "controlled-augmentation2-activations-mac2",
        "extra_fraction": 2.0,
    },
    {
        "root_id": "augmentation_0_5",
        "label": "Lote novo — extra_fraction=0,5",
        "path": ROOT / "outputs" / "controlled-augmentation05-activations-mac2",
        "extra_fraction": 0.5,
    },
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: Any, digits: int = 12) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return round(value, digits) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def training_config(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["config"]["training"]


def summary_for_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir.parents[1] / "summary.json"
    summary = read_json(summary_path)
    if not summary.get("runs"):
        raise RuntimeError(f"Resumo sem runs: {summary_path}")
    return summary["runs"][0]


def completed_run_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in ROOT_SPECS:
        pattern = "activations/*/*/*/runs/*/manifest.json"
        for manifest_path in sorted(spec["path"].glob(pattern)):
            manifest = read_json(manifest_path)
            if manifest.get("status") != "completed":
                continue
            run_dir = manifest_path.parent
            config = manifest["config"]
            dataset = str(config["dataset"])
            activation = str(config["protocol"]["hidden_activation"])
            test_path = run_dir / "artifacts" / "test_metrics.json"
            history_path = run_dir / "artifacts" / "history.csv"
            predictions_path = run_dir / "artifacts" / "predictions.csv"
            logits_path = run_dir / "artifacts" / "logits.npy"
            if not all(path.exists() for path in (test_path, history_path, predictions_path, logits_path)):
                continue
            test_metrics = read_json(test_path)
            history = pd.read_csv(history_path)
            predictions = pd.read_csv(predictions_path)
            logits = np.load(logits_path).astype(np.float64)
            classification = test_metrics["classification"]
            keras_metrics = test_metrics.get("keras_metrics", {})
            summary = summary_for_run(run_dir)
            predicted_counts = predictions["predicted_label"].value_counts().sort_index()
            true_counts = predictions["true_label"].value_counts().sort_index()
            split = manifest["data_metadata"]["split"]
            train_counts = manifest["data_metadata"]["raw_split_class_counts"]["train"]
            record = {
                "root_id": spec["root_id"],
                "root_label": spec["label"],
                "root_path": str(spec["path"]),
                "dataset": dataset,
                "activation": activation,
                "run_id": manifest["run_id"],
                "run_path": str(run_dir),
                "status": "completed",
                "generated_at": manifest.get("generated_at"),
                "seed": config["seed"],
                "split_fingerprint": split["fingerprint"],
                "train_samples": int(split["train_size"]),
                "validation_samples": int(split["validation_size"]),
                "test_samples": int(split["test_size"]),
                "train_fraction": clean(config["train_fraction"]),
                "validation_fraction": clean(config["validation_fraction"]),
                "test_fraction": clean(config["test_fraction"]),
                "batch_size": int(training_config(manifest)["batch_size"]),
                "learning_rate": clean(training_config(manifest)["learning_rate"]),
                "extra_fraction": clean(training_config(manifest)["extra_fraction"]),
                "max_epochs": int(training_config(manifest)["max_epochs"]),
                "dtype_policy": training_config(manifest)["dtype_policy"],
                "normalization": config["normalization"],
                "balance_mode": config["balance_mode"],
                "target_size": int(config["target_size"]),
                "test_accuracy": clean(classification["accuracy"]),
                "test_balanced_accuracy": clean(classification["balanced_accuracy"]),
                "test_macro_f1": clean(classification["macro_f1"]),
                "test_macro_auc": clean(classification.get("macro_ovr_auc")),
                "test_loss": clean(keras_metrics.get("loss")),
                "keras_accuracy": clean(keras_metrics.get("accuracy")),
                "epochs": int(len(history)),
                "initial_val_macro_f1": clean(history.iloc[0]["val_macro_f1"]),
                "best_val_macro_f1": clean(history["val_macro_f1"].max()),
                "best_val_epoch": int(history.loc[history["val_macro_f1"].idxmax(), "epoch"]),
                "final_val_macro_f1": clean(history.iloc[-1]["val_macro_f1"]),
                "final_train_accuracy": clean(history.iloc[-1]["accuracy"]),
                "mean_epoch_seconds": clean(summary["mean_epoch_seconds"]),
                "training_hours": clean(summary["training_total_seconds"] / 3600.0),
                "mean_train_examples_per_second": clean(summary.get("mean_train_examples_per_second")),
                "predicted_class_count": int(len(predicted_counts)),
                "predicted_class_mode": int(predicted_counts.idxmax()),
                "predicted_mode_share": clean(predicted_counts.max() / len(predictions)),
                "predicted_entropy_nats": clean(
                    -sum(
                        (count / len(predictions)) * np.log(count / len(predictions))
                        for count in predicted_counts
                    )
                ),
                "predicted_counts": {str(int(k)): int(v) for k, v in predicted_counts.items()},
                "true_counts": {str(int(k)): int(v) for k, v in true_counts.items()},
                "train_class_counts": {str(k): int(v) for k, v in train_counts.items()},
                "logit_component_std_across_rows": clean(logits.std(axis=0).mean()),
                "logit_row_l2_from_mean": clean(
                    np.linalg.norm(logits - logits.mean(axis=0, keepdims=True), axis=1).mean()
                ),
                "logit_max_abs_difference_from_first": clean(np.max(np.abs(logits - logits[0]))),
                "logit_min": clean(logits.min()),
                "logit_max": clean(logits.max()),
            }
            record["history"] = [
                {
                    "epoch": int(row.epoch),
                    "train_accuracy": clean(row.accuracy),
                    "train_loss": clean(row.loss),
                    "val_accuracy": clean(row.val_accuracy),
                    "val_loss": clean(row.val_loss),
                    "val_macro_f1": clean(row.val_macro_f1),
                }
                for row in history.itertuples(index=False)
            ]
            records.append(record)
    return records


def expected_status(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["root_id"], r["dataset"], r["activation"]): r for r in records}
    result: list[dict[str, Any]] = []
    for spec in ROOT_SPECS:
        for dataset in DATASETS:
            for activation in ACTIVATIONS:
                record = lookup.get((spec["root_id"], dataset, activation))
                result.append(
                    {
                        "root_id": spec["root_id"],
                        "root_label": spec["label"],
                        "dataset": dataset,
                        "activation": activation,
                        "status": "completed" if record else "not_completed",
                        "run_id": record["run_id"] if record else None,
                    }
                )
    return result


def paired_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["root_id"], r["dataset"], r["activation"]): r for r in records}
    pairs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for activation in ACTIVATIONS:
            old = lookup.get(("augmentation_2_0", dataset, activation))
            new = lookup.get(("augmentation_0_5", dataset, activation))
            if not old or not new:
                continue
            pairs.append(
                {
                    "dataset": dataset,
                    "activation": activation,
                    "old_extra_fraction": old["extra_fraction"],
                    "new_extra_fraction": new["extra_fraction"],
                    "old_test_accuracy": old["test_accuracy"],
                    "new_test_accuracy": new["test_accuracy"],
                    "delta_test_accuracy": clean(new["test_accuracy"] - old["test_accuracy"]),
                    "old_test_macro_f1": old["test_macro_f1"],
                    "new_test_macro_f1": new["test_macro_f1"],
                    "delta_test_macro_f1": clean(new["test_macro_f1"] - old["test_macro_f1"]),
                    "old_training_hours": old["training_hours"],
                    "new_training_hours": new["training_hours"],
                    "delta_training_hours": clean(new["training_hours"] - old["training_hours"]),
                    "old_logit_component_std": old["logit_component_std_across_rows"],
                    "new_logit_component_std": new["logit_component_std_across_rows"],
                    "old_predicted_mode_share": old["predicted_mode_share"],
                    "new_predicted_mode_share": new["predicted_mode_share"],
                }
            )
    return pairs


def quality_checks(records: list[dict[str, Any]], statuses: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in statuses if row["status"] == "completed"]
    failures: list[str] = []
    for record in records:
        if sum(record["predicted_counts"].values()) != record["test_samples"]:
            failures.append(f"predictions não fecha: {record['run_id']}")
        if sum(record["true_counts"].values()) != record["test_samples"]:
            failures.append(f"rótulos não fecham: {record['run_id']}")
        if record["epochs"] != record["max_epochs"]:
            failures.append(f"épocas incompletas: {record['run_id']}")
        if record["test_accuracy"] is None or not 0 <= record["test_accuracy"] <= 1:
            failures.append(f"acurácia inválida: {record['run_id']}")
    grouped: dict[tuple[str, str], set[str]] = {}
    for record in records:
        grouped.setdefault((record["root_id"], record["dataset"]), set()).add(record["activation"])
    comparable_groups = [
        {"root_id": root_id, "dataset": dataset, "activations": sorted(activations)}
        for (root_id, dataset), activations in sorted(grouped.items())
    ]
    missing = [row for row in statuses if row["status"] != "completed"]
    return {
        "completed_runs": len(completed),
        "expected_runs": len(statuses),
        "missing_or_incomplete_runs": missing,
        "comparable_groups": comparable_groups,
        "failures": failures,
        "passed": not failures,
    }


def intermediate_evidence(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Probe hidden-layer variation for new MNIST/Fashion runs."""

    try:
        import tensorflow as tf
        from tcc_benchmark.adapters import load_local_dataset
        from tcc_benchmark.config import load_dataset_registry
        from tcc_benchmark.data import (
            build_tf_dataset,
            select_samples,
            stratified_split_indices,
            unit_interval_stats,
        )
        from tcc_benchmark.runner import _join_source_samples
    except Exception as exc:  # pragma: no cover
        return [{"status": "unavailable", "reason": repr(exc)}]

    registry = load_dataset_registry(ROOT / "configs" / "datasets.yaml")
    evidence: list[dict[str, Any]] = []
    new_records = [
        row for row in records if row["root_id"] == "augmentation_0_5" and row["dataset"] in PROBE_DATASETS
    ]
    for dataset in PROBE_DATASETS:
        dataset_records = [row for row in new_records if row["dataset"] == dataset]
        if not dataset_records:
            continue
        first = dataset_records[0]
        entry = registry[dataset]
        materials = _join_source_samples(
            load_local_dataset(entry.adapter, entry.root, **entry.options), image_size=first["target_size"]
        )
        split = stratified_split_indices(
            materials.labels,
            seed=first["seed"],
            train_fraction=first["train_fraction"],
            validation_fraction=first["validation_fraction"],
            test_fraction=first["test_fraction"],
        )
        test_samples = select_samples(materials.samples, split.test)
        test_labels = materials.labels[split.test]
        channels = int(materials.channels)
        test_ds, _ = build_tf_dataset(
            test_samples,
            test_labels,
            image_size=first["target_size"],
            channels=channels,
            batch_size=first["batch_size"],
            normalization=unit_interval_stats(channels=channels, image_size=first["target_size"]),
            training=False,
            seed=first["seed"],
            preprocess_cache_max_mib=1024,
        )
        images, _labels = next(iter(test_ds))
        for record in dataset_records:
            model = tf.keras.models.load_model(Path(record["run_path"]) / "checkpoints" / "best.keras", compile=False)
            probe = tf.keras.Model(model.input, [model.get_layer(name).output for name in PROBE_LAYERS])
            outputs = probe(images, training=False)
            for layer, output in zip(PROBE_LAYERS, outputs, strict=True):
                values = np.asarray(output.numpy(), dtype=np.float64)
                flat = values.reshape(values.shape[0], -1)
                centered = flat - flat.mean(axis=0, keepdims=True)
                row: dict[str, Any] = {
                    "root_id": record["root_id"],
                    "root_label": record["root_label"],
                    "dataset": dataset,
                    "activation": record["activation"],
                    "layer": layer,
                    "sample_count": int(flat.shape[0]),
                    "output_shape": list(values.shape),
                    "mean_feature_std_across_samples": clean(flat.std(axis=0).mean()),
                    "mean_l2_distance_from_sample_mean": clean(np.linalg.norm(centered, axis=1).mean()),
                    "max_abs_difference_from_first": clean(np.max(np.abs(flat - flat[0]))),
                }
                if record["activation"] == "softmax" and layer != "logits":
                    sums = values.sum(axis=-1)
                    row["softmax_last_axis_sum_mean"] = clean(sums.mean())
                    row["softmax_last_axis_sum_std"] = clean(sums.std())
                    row["softmax_last_axis_entropy_mean_nats"] = clean(
                        np.mean(-np.sum(values * np.log(np.clip(values, 1e-30, None)), axis=-1))
                    )
                evidence.append(row)
            del probe, model, outputs
            tf.keras.backend.clear_session()
    return evidence


def main() -> None:
    records = completed_run_records()
    statuses = expected_status(records)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "roots": [
                {
                    "root_id": spec["root_id"],
                    "label": spec["label"],
                    "path": str(spec["path"]),
                    "extra_fraction": spec["extra_fraction"],
                }
                for spec in ROOT_SPECS
            ],
            "datasets": list(DATASETS),
            "activations": list(ACTIVATIONS),
            "protocol": "seed 42, split estratificado 70/15/15, batch 256, LR 0.0003, mixed_float16, mesma arquitetura e avaliação sem augmentation",
        },
        "records": records,
        "status_matrix": statuses,
        "paired_comparisons": paired_comparisons(records),
        "quality": quality_checks(records, statuses),
        "intermediate": intermediate_evidence(records),
        "source_paths": {
            "model_code": "src/tcc_benchmark/model.py",
            "data_code": "src/tcc_benchmark/data.py",
            "metrics_code": "src/tcc_benchmark/metrics.py",
            "old_activation_runs": "outputs/controlled-augmentation2-activations-mac2/activations/",
            "new_activation_runs": "outputs/controlled-augmentation05-activations-mac2/activations/",
            "old_activation_suites": "outputs/controlled-augmentation2-activations-mac2/generated-suites/activations/",
            "new_activation_suites": "outputs/controlled-augmentation05-activations-mac2/generated-suites/activations/",
            "analysis_script": "analysis_reports/investigate_activation_matrix_updated.py",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
