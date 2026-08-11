"""Consolidate the canonical 36 benchmark runs into auditable analysis tables.

The first six datasets live in the original optimized output root.  The final
three datasets were intentionally restarted in the RAM-capped output root on
drive E.  Old interrupted attempts are excluded by this explicit canonical map.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASETS = (
    "mnist",
    "fashion_mnist",
    "kmnist",
    "emnist_balanced",
    "cifar10",
    "cifar100_coarse",
    "svhn",
    "gtsrb",
    "fer2013",
)
BALANCE_MODES = ("all_raw", "undersample", "oversample", "class_weight")
LATE_DATASETS = {"svhn", "gtsrb", "fer2013"}
REQUIRED_FILES = (
    "manifest.json",
    "status.json",
    "artifacts/test_metrics.json",
    "artifacts/metrics.json",
    "artifacts/predictions.csv",
    "checkpoints/best.keras",
    "checkpoints/last.keras",
    "checkpoints/epoch_metrics.csv",
    "logs/training_summary.json",
    "telemetry/environment.json",
    "telemetry/samples.csv",
    "telemetry/summary.json",
)


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return value


def _telemetry_stat(payload: dict[str, Any], metric: str, stat: str) -> float:
    value = payload.get("metrics", {}).get(metric, {}).get(stat)
    return float(value) if isinstance(value, (int, float)) else math.nan


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b"")) - 1)


def _add_check(
    checks: list[dict[str, Any]],
    run_id: str,
    check: str,
    passed: bool,
    severity: str,
    detail: str,
) -> None:
    checks.append(
        {
            "run_id": run_id,
            "check": check,
            "passed": bool(passed),
            "severity_if_failed": severity,
            "detail": detail,
        }
    )


def build_analysis(primary_root: Path, remaining_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    epoch_frames: list[pd.DataFrame] = []
    class_rows: list[dict[str, Any]] = []
    quality_checks: list[dict[str, Any]] = []

    for dataset in DATASETS:
        canonical_root = remaining_root if dataset in LATE_DATASETS else primary_root
        for balance_mode in BALANCE_MODES:
            run_id = f"{dataset}__zscore__{balance_mode}__seed-42"
            run_dir = canonical_root / dataset / "runs" / run_id
            missing = [relative for relative in REQUIRED_FILES if not (run_dir / relative).is_file()]
            _add_check(
                quality_checks,
                run_id,
                "required_files",
                not missing,
                "critical",
                "all required files present" if not missing else f"missing: {', '.join(missing)}",
            )
            if missing:
                continue

            manifest = _json(run_dir / "manifest.json")
            status = _json(run_dir / "status.json")
            training = _json(run_dir / "logs" / "training_summary.json")
            telemetry = _json(run_dir / "telemetry" / "summary.json")
            environment = _json(run_dir / "telemetry" / "environment.json")
            test_metrics = _json(run_dir / "artifacts" / "test_metrics.json")
            metrics_report = _json(run_dir / "artifacts" / "metrics.json")

            epochs = pd.read_csv(run_dir / "checkpoints" / "epoch_metrics.csv")
            numeric_columns = [column for column in epochs.columns if column != "epoch"]
            epochs[numeric_columns] = epochs[numeric_columns].apply(pd.to_numeric, errors="coerce")
            epochs["epoch"] = pd.to_numeric(epochs["epoch"], errors="coerce").astype("Int64")
            run_identity = pd.DataFrame(
                {
                    "run_id": [run_id] * len(epochs),
                    "dataset": [dataset] * len(epochs),
                    "balance_mode": [balance_mode] * len(epochs),
                },
                index=epochs.index,
            )
            epochs = pd.concat([run_identity, epochs], axis=1)
            leading_columns = ["run_id", "dataset", "balance_mode"]
            epochs = epochs[leading_columns + [column for column in epochs.columns if column not in leading_columns]].copy()
            epoch_frames.append(epochs)

            metric_epochs = epochs.dropna(subset=["val_macro_f1", "epoch"])
            best_index = metric_epochs["val_macro_f1"].astype(float).idxmax()
            best_row = epochs.loc[best_index]
            final_row = epochs.iloc[-1]
            epochs_completed = int(training["epochs_completed"])
            best_epoch = int(best_row["epoch"])
            patience = int(manifest["config"]["training"]["early_stopping_patience"])
            max_epochs = int(training["max_epochs"])
            early_stop_gap = epochs_completed - best_epoch
            early_stop_consistent = epochs_completed == max_epochs or early_stop_gap == patience

            classification = test_metrics["classification"]
            keras_metrics = test_metrics["keras_metrics"]
            per_class = classification["per_class"]
            support_sum = int(sum(int(values["support"]) for values in per_class.values()))
            predictions_rows = _line_count(run_dir / "artifacts" / "predictions.csv")

            data_metadata = manifest.get("data_metadata", {})
            dataset_metadata = data_metadata.get("datasets", {})
            train_metadata = dataset_metadata.get("train", {})
            validation_metadata = dataset_metadata.get("validation", {})
            test_metadata = dataset_metadata.get("test", {})
            training_config = manifest["config"]["training"]
            telemetry_samples_frame = pd.read_csv(run_dir / "telemetry" / "samples.csv")
            telemetry_samples = int(len(telemetry_samples_frame))
            combined_event_epoch_count = int((telemetry_samples_frame["event"] == "epoch_end").sum())

            def sample_stat(metric: str, stat: str) -> float:
                values = pd.to_numeric(telemetry_samples_frame.get(metric), errors="coerce").dropna()
                if values.empty:
                    return math.nan
                if stat == "mean":
                    return float(values.mean())
                if stat == "p95":
                    return float(values.quantile(0.95))
                if stat == "max":
                    return float(values.max())
                if stat == "min":
                    return float(values.min())
                raise ValueError(f"Unsupported sample statistic: {stat}")

            train_examples_per_epoch = float(final_row.get("train_examples", math.nan))
            total_examples_processed = float(epochs["train_examples"].sum())
            training_seconds_reported = float(training["training_seconds"])
            epoch_seconds_sum = float(epochs["epoch_seconds"].sum())
            training_seconds = epoch_seconds_sum
            weighted_throughput = total_examples_processed / training_seconds if training_seconds else math.nan

            test_summary = metrics_report.get("test", {})
            n_samples = int(test_summary.get("n_samples", support_sum))
            selected_checkpoint = str(training.get("selected_checkpoint", ""))
            summary_event_epoch_count = int(telemetry.get("event_counts", {}).get("epoch_end", -1))
            epoch_numbers = epochs["epoch"].astype(int).tolist()
            sequential_epochs = epoch_numbers == list(range(1, epochs_completed + 1))

            _add_check(quality_checks, run_id, "status_completed", status.get("status") == "completed", "critical", str(status.get("status")))
            _add_check(quality_checks, run_id, "epoch_count_matches_summary", len(epochs) == epochs_completed, "critical", f"csv={len(epochs)}, summary={epochs_completed}")
            _add_check(quality_checks, run_id, "epoch_sequence", sequential_epochs, "high", f"first={epoch_numbers[:1]}, last={epoch_numbers[-1:]}")
            _add_check(quality_checks, run_id, "telemetry_epoch_count", combined_event_epoch_count == epochs_completed, "high", f"samples={combined_event_epoch_count}, summary_current_attempt={summary_event_epoch_count}, epochs={epochs_completed}")
            _add_check(quality_checks, run_id, "prediction_count", predictions_rows == n_samples, "critical", f"predictions={predictions_rows}, expected={n_samples}")
            _add_check(quality_checks, run_id, "classification_support", support_sum == n_samples, "critical", f"support={support_sum}, expected={n_samples}")
            _add_check(quality_checks, run_id, "early_stopping_policy", early_stop_consistent, "high", f"best_epoch={best_epoch}, completed={epochs_completed}, patience={patience}")
            _add_check(quality_checks, run_id, "training_seconds_reconciled", math.isclose(epoch_seconds_sum, training_seconds_reported, rel_tol=1e-8, abs_tol=1e-4), "medium", f"epochs_sum={epoch_seconds_sum:.6f}, summary={training_seconds_reported:.6f}")
            _add_check(quality_checks, run_id, "best_checkpoint_selected", selected_checkpoint.endswith("/best.keras") or selected_checkpoint.endswith("\\best.keras"), "high", selected_checkpoint)
            _add_check(quality_checks, run_id, "finite_headline_metrics", all(math.isfinite(float(value)) for value in (classification["accuracy"], classification["balanced_accuracy"], classification["macro_f1"], keras_metrics["loss"])), "critical", "test headline metrics are finite")

            hardware = environment.get("hardware", {})
            gpus = hardware.get("gpus", [])
            gpu_name = str(gpus[0].get("name", "")) if gpus else ""
            run_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "balance_mode": balance_mode,
                    "normalization": manifest["config"]["normalization"],
                    "seed": int(manifest["config"]["seed"]),
                    "status": status["status"],
                    "attempt": int(status.get("attempt", 0)),
                    "output_root": str(canonical_root),
                    "target_size": int(manifest["config"]["target_size"]),
                    "channels": int(data_metadata.get("channels", manifest["config"].get("native_channels", 0))),
                    "num_classes": int(manifest["config"]["num_classes"]),
                    "source_samples": int(manifest["config"]["source_manifest"]["joined_samples"]),
                    "train_source_examples": int(train_metadata.get("source_examples", 0)),
                    "train_effective_examples": int(train_metadata.get("total_examples", 0)),
                    "validation_examples": int(validation_metadata.get("total_examples", 0)),
                    "test_examples_pipeline": int(test_metadata.get("total_examples", 0)),
                    "test_samples": n_samples,
                    "batch_size": int(training_config["batch_size"]),
                    "shuffle_buffer_max_mib": training_config.get("shuffle_buffer_max_mib"),
                    "preprocess_cache_max_mib": training_config.get("preprocess_cache_max_mib"),
                    "split_fingerprint": manifest.get("split_fingerprint"),
                    "config_fingerprint": manifest.get("config_fingerprint"),
                    "tensorflow_version": environment.get("packages", {}).get("tensorflow"),
                    "gpu_name": gpu_name,
                    "epochs_completed": epochs_completed,
                    "max_epochs": max_epochs,
                    "best_epoch": best_epoch,
                    "epochs_after_best": early_stop_gap,
                    "early_stop_consistent": bool(early_stop_consistent),
                    "best_val_macro_f1": float(best_row["val_macro_f1"]),
                    "best_val_balanced_accuracy": float(best_row["val_balanced_accuracy"]),
                    "best_val_accuracy": float(best_row["val_accuracy"]),
                    "final_train_accuracy": float(final_row["accuracy"]),
                    "final_train_loss": float(final_row["loss"]),
                    "final_val_accuracy": float(final_row["val_accuracy"]),
                    "final_val_loss": float(final_row["val_loss"]),
                    "final_val_macro_f1": float(final_row["val_macro_f1"]),
                    "final_learning_rate": float(final_row["learning_rate"]),
                    "test_accuracy": float(classification["accuracy"]),
                    "test_balanced_accuracy": float(classification["balanced_accuracy"]),
                    "test_macro_f1": float(classification["macro_f1"]),
                    "test_loss": float(keras_metrics["loss"]),
                    "val_test_macro_f1_gap": float(best_row["val_macro_f1"]) - float(classification["macro_f1"]),
                    "train_test_accuracy_gap": float(final_row["accuracy"]) - float(classification["accuracy"]),
                    "training_seconds": training_seconds,
                    "training_seconds_reported": training_seconds_reported,
                    "training_hours": training_seconds / 3600.0,
                    "evaluation_seconds": float(training["evaluation_seconds"]),
                    "wall_seconds_current_attempt": float(training["wall_seconds_current_attempt"]),
                    "mean_epoch_seconds": float(training["mean_epoch_seconds"]),
                    "time_to_best_seconds": float(epochs.loc[epochs["epoch"].astype(int) <= best_epoch, "epoch_seconds"].sum()),
                    "train_examples_per_epoch": train_examples_per_epoch,
                    "total_examples_processed": total_examples_processed,
                    "mean_train_examples_per_second": float(training["mean_train_examples_per_second"]),
                    "median_train_examples_per_second": float(training["median_train_examples_per_second"]),
                    "weighted_train_examples_per_second": weighted_throughput,
                    "telemetry_samples": telemetry_samples,
                    "gpu_util_mean_percent": sample_stat("gpu_utilization_percent", "mean"),
                    "gpu_util_p95_percent": sample_stat("gpu_utilization_percent", "p95"),
                    "gpu_util_max_percent": sample_stat("gpu_utilization_percent", "max"),
                    "gpu_memory_mean_gib": sample_stat("gpu_memory_used_bytes", "mean") / (1024**3),
                    "gpu_memory_p95_gib": sample_stat("gpu_memory_used_bytes", "p95") / (1024**3),
                    "gpu_memory_max_gib": sample_stat("gpu_memory_used_bytes", "max") / (1024**3),
                    "gpu_temperature_mean_c": sample_stat("gpu_temperature_c", "mean"),
                    "gpu_temperature_max_c": sample_stat("gpu_temperature_c", "max"),
                    "ram_mean_percent": sample_stat("ram_percent", "mean"),
                    "ram_p95_percent": sample_stat("ram_percent", "p95"),
                    "ram_max_percent": sample_stat("ram_percent", "max"),
                    "ram_used_mean_gib": sample_stat("ram_used_bytes", "mean") / (1024**3),
                    "ram_used_max_gib": sample_stat("ram_used_bytes", "max") / (1024**3),
                    "process_rss_mean_gib": sample_stat("process_rss_bytes", "mean") / (1024**3),
                    "process_rss_max_gib": sample_stat("process_rss_bytes", "max") / (1024**3),
                    "cpu_mean_percent": sample_stat("cpu_percent", "mean"),
                    "process_cpu_mean_percent": sample_stat("process_cpu_percent", "mean"),
                    "output_disk_free_min_gib": sample_stat("disk_output_free_bytes", "min") / (1024**3),
                    "output_disk_used_mean_percent": sample_stat("disk_output_percent", "mean"),
                    "selected_checkpoint": selected_checkpoint,
                    "run_dir": str(run_dir),
                }
            )

            for class_id, values in per_class.items():
                class_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "balance_mode": balance_mode,
                        "class_id": int(class_id),
                        "class_name": manifest["config"]["class_names"][int(class_id)],
                        "precision": float(values["precision"]),
                        "recall": float(values["recall"]),
                        "f1": float(values["f1"]),
                        "support": int(values["support"]),
                    }
                )

    runs = pd.DataFrame(run_rows)
    all_epochs = pd.concat(epoch_frames, ignore_index=True)
    per_class_metrics = pd.DataFrame(class_rows)
    checks = pd.DataFrame(quality_checks)

    if len(runs) != len(DATASETS) * len(BALANCE_MODES):
        raise RuntimeError(f"Expected 36 canonical runs, found {len(runs)}")

    baseline = runs.loc[runs["balance_mode"] == "all_raw", ["dataset", "test_macro_f1", "test_balanced_accuracy", "test_accuracy"]].rename(
        columns={
            "test_macro_f1": "all_raw_test_macro_f1",
            "test_balanced_accuracy": "all_raw_test_balanced_accuracy",
            "test_accuracy": "all_raw_test_accuracy",
        }
    )
    runs = runs.merge(baseline, on="dataset", how="left", validate="many_to_one")
    runs["delta_macro_f1_vs_all_raw"] = runs["test_macro_f1"] - runs["all_raw_test_macro_f1"]
    runs["delta_balanced_accuracy_vs_all_raw"] = runs["test_balanced_accuracy"] - runs["all_raw_test_balanced_accuracy"]
    runs["delta_accuracy_vs_all_raw"] = runs["test_accuracy"] - runs["all_raw_test_accuracy"]
    runs["macro_f1_rank_within_dataset"] = runs.groupby("dataset")["test_macro_f1"].rank(method="min", ascending=False).astype(int)

    dataset_summary = (
        runs.sort_values(["dataset", "test_macro_f1"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()[
            [
                "dataset",
                "run_id",
                "balance_mode",
                "test_macro_f1",
                "test_balanced_accuracy",
                "test_accuracy",
                "training_hours",
                "epochs_completed",
            ]
        ]
        .rename(
            columns={
                "run_id": "best_run_id",
                "balance_mode": "best_balance_mode",
                "test_macro_f1": "best_test_macro_f1",
                "test_balanced_accuracy": "best_test_balanced_accuracy",
                "test_accuracy": "best_test_accuracy",
                "training_hours": "best_run_training_hours",
                "epochs_completed": "best_run_epochs",
            }
        )
    )
    balance_summary = (
        runs.groupby("balance_mode", as_index=False)
        .agg(
            mean_test_macro_f1=("test_macro_f1", "mean"),
            median_test_macro_f1=("test_macro_f1", "median"),
            mean_delta_macro_f1_vs_all_raw=("delta_macro_f1_vs_all_raw", "mean"),
            median_delta_macro_f1_vs_all_raw=("delta_macro_f1_vs_all_raw", "median"),
            dataset_wins=("macro_f1_rank_within_dataset", lambda values: int((values == 1).sum())),
            mean_training_hours=("training_hours", "mean"),
            total_training_hours=("training_hours", "sum"),
            mean_gpu_util_percent=("gpu_util_mean_percent", "mean"),
            mean_ram_percent=("ram_mean_percent", "mean"),
        )
        .sort_values("mean_delta_macro_f1_vs_all_raw", ascending=False)
    )

    config_by_dataset = (
        runs.groupby("dataset", as_index=False)
        .agg(
            distinct_split_fingerprints=("split_fingerprint", "nunique"),
            distinct_source_samples=("source_samples", "nunique"),
            distinct_batch_sizes=("batch_size", "nunique"),
            distinct_tensorflow_versions=("tensorflow_version", "nunique"),
            distinct_shuffle_caps=("shuffle_buffer_max_mib", lambda values: len({"none" if pd.isna(value) else float(value) for value in values})),
        )
    )
    config_by_dataset["within_dataset_comparable"] = (
        (config_by_dataset["distinct_split_fingerprints"] == 1)
        & (config_by_dataset["distinct_source_samples"] == 1)
        & (config_by_dataset["distinct_batch_sizes"] == 1)
        & (config_by_dataset["distinct_tensorflow_versions"] == 1)
        & (config_by_dataset["distinct_shuffle_caps"] == 1)
    )

    failed_checks = checks.loc[~checks["passed"]].copy()
    total_training_seconds = float(runs["training_seconds"].sum())
    telemetry_weights = runs["telemetry_samples"].astype(float)
    overview = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "canonical_runs": int(len(runs)),
        "completed_runs": int((runs["status"] == "completed").sum()),
        "datasets": int(runs["dataset"].nunique()),
        "balance_modes": int(runs["balance_mode"].nunique()),
        "epochs_recorded": int(runs["epochs_completed"].sum()),
        "training_seconds": total_training_seconds,
        "training_hours": total_training_seconds / 3600.0,
        "evaluation_seconds": float(runs["evaluation_seconds"].sum()),
        "examples_processed": float(runs["total_examples_processed"].sum()),
        "quality_checks": int(len(checks)),
        "failed_quality_checks": int(len(failed_checks)),
        "critical_or_high_failed_checks": int((failed_checks["severity_if_failed"].isin(["critical", "high"])).sum()),
        "all_datasets_within_dataset_comparable": bool(config_by_dataset["within_dataset_comparable"].all()),
        "sample_weighted_gpu_util_mean_percent": float(np.average(runs["gpu_util_mean_percent"], weights=telemetry_weights)),
        "sample_weighted_ram_mean_percent": float(np.average(runs["ram_mean_percent"], weights=telemetry_weights)),
        "peak_gpu_memory_gib": float(runs["gpu_memory_max_gib"].max()),
        "peak_ram_percent": float(runs["ram_max_percent"].max()),
        "peak_process_rss_gib": float(runs["process_rss_max_gib"].max()),
        "max_gpu_temperature_c": float(runs["gpu_temperature_max_c"].max()),
        "best_overall_run_id": str(runs.loc[runs["test_macro_f1"].idxmax(), "run_id"]),
        "best_overall_test_macro_f1": float(runs["test_macro_f1"].max()),
        "single_seed_limitation": True,
        "canonical_primary_root": str(primary_root),
        "canonical_remaining_root": str(remaining_root),
    }

    runs.sort_values(["dataset", "balance_mode"]).to_csv(output_dir / "run_metrics.csv", index=False)
    all_epochs.to_csv(output_dir / "epoch_metrics_all.csv", index=False)
    per_class_metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)
    checks.to_csv(output_dir / "quality_checks.csv", index=False)
    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    balance_summary.to_csv(output_dir / "balance_mode_summary.csv", index=False)
    config_by_dataset.to_csv(output_dir / "comparability_by_dataset.csv", index=False)
    with (output_dir / "analysis_overview.json").open("w", encoding="utf-8") as handle:
        json.dump(overview, handle, ensure_ascii=False, indent=2)

    return {
        "runs": runs,
        "epochs": all_epochs,
        "per_class": per_class_metrics,
        "checks": checks,
        "dataset_summary": dataset_summary,
        "balance_summary": balance_summary,
        "comparability": config_by_dataset,
        "overview": overview,
        "output_dir": output_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--remaining-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_analysis(args.primary_root, args.remaining_root, args.output_dir)
    print(json.dumps(result["overview"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
