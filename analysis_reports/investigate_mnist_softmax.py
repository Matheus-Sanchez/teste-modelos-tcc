"""Reproduce the MNIST activation comparison and diagnose hidden-softmax collapse.

The script reads only completed benchmark artifacts, then optionally loads the
three saved checkpoints to measure how much sample-dependent variation survives
through the hidden layers. It writes a compact JSON evidence package used by
the explanatory report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_ROOT = ROOT / "outputs" / "controlled-augmentation2-activations-mac2" / "activations" / "mnist"
OUTPUT = ROOT / "analysis_reports" / "investigacao_mnist_softmax.json"
ACTIVATIONS = ("relu", "sigmoid", "softmax")
PROBE_LAYERS = (
    "block1_activation1",
    "block2_activation1",
    "block3_activation1",
    "block4_activation1",
    "block5_activation1",
    "block5_activation2",
    "dense1",
    "dense2",
    "logits",
)


def run_dir(activation: str) -> Path:
    candidates = sorted((ACTIVATION_ROOT / activation / "mnist" / "runs").glob("*"))
    if len(candidates) != 1:
        raise RuntimeError(f"Esperava exatamente um run para {activation}, encontrei {len(candidates)}.")
    return candidates[0]


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


def basic_evidence() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    prediction_distribution: dict[str, Any] = {}
    for activation in ACTIVATIONS:
        root = run_dir(activation)
        test = read_json(root / "artifacts" / "test_metrics.json")
        summary = read_json(root.parent.parent / "summary.json")
        manifest = read_json(root / "manifest.json")
        frame = pd.read_csv(root / "artifacts" / "history.csv")
        predictions = pd.read_csv(root / "artifacts" / "predictions.csv")
        logits = np.load(root / "artifacts" / "logits.npy").astype(np.float64)
        classification = test["classification"]
        keras_metrics = test["keras_metrics"]
        counts = predictions["predicted_label"].value_counts().sort_index()
        true_counts = predictions["true_label"].value_counts().sort_index()
        train_counts = {
            str(label): int(count)
            for label, count in manifest["data_metadata"]["raw_split_class_counts"]["train"].items()
        }
        comparison.append(
            {
                "activation": activation,
                "test_accuracy": clean(classification["accuracy"]),
                "test_balanced_accuracy": clean(classification["balanced_accuracy"]),
                "test_macro_f1": clean(classification["macro_f1"]),
                "test_macro_auc": clean(classification["macro_ovr_auc"]),
                "test_loss": clean(keras_metrics["loss"]),
                "epochs": int(len(frame)),
                "initial_val_macro_f1": clean(frame.iloc[0]["val_macro_f1"]),
                "best_val_macro_f1": clean(frame["val_macro_f1"].max()),
                "best_val_epoch": int(frame.loc[frame["val_macro_f1"].idxmax(), "epoch"]),
                "final_val_macro_f1": clean(frame.iloc[-1]["val_macro_f1"]),
                "final_train_accuracy": clean(frame.iloc[-1]["accuracy"]),
                "mean_epoch_seconds": clean(summary["runs"][0]["mean_epoch_seconds"]),
                "training_hours": clean(summary["runs"][0]["training_total_seconds"] / 3600),
                "test_samples": int(len(predictions)),
                "predicted_class_count": int(len(counts)),
                "predicted_class_mode": int(counts.idxmax()),
                "predicted_mode_share": clean(counts.max() / len(predictions)),
                "predicted_entropy_nats": clean(
                    -sum((value / len(predictions)) * np.log(value / len(predictions)) for value in counts)
                ),
                "logit_component_std_across_rows": clean(logits.std(axis=0).mean()),
                "logit_row_l2_from_mean": clean(
                    np.linalg.norm(logits - logits.mean(axis=0, keepdims=True), axis=1).mean()
                ),
                "logit_max_abs_difference_from_first": clean(np.max(np.abs(logits - logits[0]))),
                "logit_min": clean(logits.min()),
                "logit_max": clean(logits.max()),
            }
        )
        for row in frame.itertuples(index=False):
            history.append(
                {
                    "activation": activation,
                    "epoch": int(row.epoch),
                    "train_accuracy": clean(row.accuracy),
                    "train_loss": clean(row.loss),
                    "val_accuracy": clean(row.val_accuracy),
                    "val_loss": clean(row.val_loss),
                    "val_macro_f1": clean(row.val_macro_f1),
                }
            )
        prediction_distribution[activation] = [
            {
                "label": int(label),
                "true_count": int(true_counts.get(label, 0)),
                "predicted_count": int(counts.get(label, 0)),
            }
            for label in range(10)
        ]
    return comparison, history, prediction_distribution


def intermediate_evidence() -> list[dict[str, Any]]:
    """Measure sample-dependent variation on the first test batch."""

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
    except Exception as exc:  # pragma: no cover - the artifact-only path remains useful
        return [{"status": "unavailable", "reason": repr(exc)}]

    registry = load_dataset_registry(ROOT / "configs" / "datasets.yaml")
    entry = registry["mnist"]
    materials = _join_source_samples(load_local_dataset(entry.adapter, entry.root, **entry.options), image_size=64)
    split = stratified_split_indices(materials.labels, seed=42, train_fraction=0.70, validation_fraction=0.15, test_fraction=0.15)
    test_samples = select_samples(materials.samples, split.test)
    test_labels = materials.labels[split.test]
    test_ds, _ = build_tf_dataset(
        test_samples,
        test_labels,
        image_size=64,
        channels=1,
        batch_size=256,
        normalization=unit_interval_stats(channels=1, image_size=64),
        training=False,
        seed=42,
        preprocess_cache_max_mib=1024,
    )
    images, _labels = next(iter(test_ds))
    evidence: list[dict[str, Any]] = []
    for activation in ACTIVATIONS:
        model = tf.keras.models.load_model(run_dir(activation) / "checkpoints" / "best.keras", compile=False)
        probe = tf.keras.Model(model.input, [model.get_layer(name).output for name in PROBE_LAYERS])
        outputs = probe(images, training=False)
        for layer, output in zip(PROBE_LAYERS, outputs, strict=True):
            values = np.asarray(output.numpy(), dtype=np.float64)
            flat = values.reshape(values.shape[0], -1)
            centered = flat - flat.mean(axis=0, keepdims=True)
            row: dict[str, Any] = {
                "activation": activation,
                "layer": layer,
                "sample_count": int(flat.shape[0]),
                "output_shape": list(values.shape),
                "mean_feature_std_across_samples": clean(flat.std(axis=0).mean()),
                "mean_l2_distance_from_sample_mean": clean(np.linalg.norm(centered, axis=1).mean()),
                "max_abs_difference_from_first": clean(np.max(np.abs(flat - flat[0]))),
            }
            if activation == "softmax" and layer != "logits":
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
    comparison, history, prediction_distribution = basic_evidence()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "dataset": "mnist",
            "activations": list(ACTIVATIONS),
            "seed": 42,
            "split_fingerprint": "e318c7442df3e0a1d1f67bd6fa1eac1602b6440bf135240aca0008293af1e6c8",
            "test_samples": 10498,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "batch_size": 256,
            "max_epochs": 100,
            "learning_rate": 0.0003,
            "dtype_policy": "mixed_float16",
            "normalization": "unit_interval",
            "balance_mode": "all_raw",
            "extra_fraction": 2.0,
            "augmentation": "same fixed augmentation policy for all variants; evaluation without augmentation",
        },
        "comparison": comparison,
        "history": history,
        "prediction_distribution": prediction_distribution,
        "intermediate": intermediate_evidence(),
        "source_paths": {
            "model_code": "src/tcc_benchmark/model.py",
            "metrics_code": "src/tcc_benchmark/metrics.py",
            "activation_suite": "outputs/controlled-augmentation2-activations-mac2/generated-suites/activations/",
            "run_artifacts": "outputs/controlled-augmentation2-activations-mac2/activations/mnist/",
            "analysis_script": "analysis_reports/investigate_mnist_softmax.py",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
