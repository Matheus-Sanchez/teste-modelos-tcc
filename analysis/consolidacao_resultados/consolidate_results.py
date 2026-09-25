from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
BASE = REPO / "analysis" / "consolidacao_resultados"
DATA_DIR = BASE / "data"
APP_DIR = BASE / "report_app"
MAC_REF = "origin/codex/mac-training-split"
WINDOWS_CAMPAIGN = "controlled-augmentation05-batch-activation"
MAC_RAW_CAMPAIGN = "controlled-augmentation2-mac-m4-aug05"
MAC_ACTIVATION_CAMPAIGN = "controlled-augmentation05-activations-mac2"
TELEMETRY_PROFILE_BINS = 20


DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "kmnist": "KMNIST",
    "emnist_balanced": "EMNIST Balanced",
    "cifar10": "CIFAR-10",
    "cifar100_coarse": "CIFAR-100 coarse",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "fer2013": "FER2013",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SNAPSHOT_AT = now_utc()


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def nested(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int | None:
    numeric = as_float(value)
    return int(numeric) if numeric is not None else None


def slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unknown"


def git(args: list[str], *, check: bool = True, binary: bool = False) -> bytes | str:
    cmd = [
        "git",
        "-c",
        f"safe.directory={REPO}",
        "-C",
        str(REPO),
        *args,
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8", errors="replace")


_TREE_BLOB_CACHE: dict[str, dict[str, str]] = {}


def git_blob_map(ref: str) -> dict[str, str]:
    cached = _TREE_BLOB_CACHE.get(ref)
    if cached is not None:
        return cached
    payload = git(["ls-tree", "-r", "-z", ref], binary=True)
    mapping: dict[str, str] = {}
    for record in bytes(payload).split(b"\0"):
        if not record or b"\t" not in record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        fields = metadata.split()
        if len(fields) < 3 or fields[1] != b"blob":
            continue
        mapping[raw_path.decode("utf-8", errors="replace")] = fields[2].decode("ascii")
    _TREE_BLOB_CACHE[ref] = mapping
    return mapping


def git_show_text(ref: str, path: str) -> str | None:
    blob = git_blob_map(ref).get(path)
    if not blob:
        return None
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPO}",
            "-C",
            str(REPO),
            "cat-file",
            "blob",
            blob,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8-sig", errors="replace")


def git_list(ref: str, prefix: str) -> list[str]:
    text = git(["ls-tree", "-r", "--name-only", ref, prefix])
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def json_loads(text: str | None) -> dict[str, Any] | list[Any] | None:
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def read_csv_text(text: str | None) -> pd.DataFrame:
    if not text:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()


def relative_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


@dataclass
class RunSource:
    platform: str
    campaign: str
    run_root: str
    source_branch: str
    source_commit: str
    reader: Callable[[str], str | None]
    exists: Callable[[str], bool]
    local_root: Path | None = None

    def text(self, rel: str) -> str | None:
        return self.reader(rel.replace("\\", "/"))

    def json(self, rel: str) -> dict[str, Any] | list[Any] | None:
        return json_loads(self.text(rel))


def local_run_source(run_root: Path, campaign: str, commit: str) -> RunSource:
    def reader(rel: str) -> str | None:
        path = run_root / Path(rel)
        try:
            return path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return None

    def exists(rel: str) -> bool:
        return (run_root / Path(rel)).is_file()

    return RunSource(
        platform="Windows",
        campaign=campaign,
        run_root=relative_repo(run_root),
        source_branch="worktree:consolidacao-de-resultados",
        source_commit=commit,
        reader=reader,
        exists=exists,
        local_root=run_root,
    )


def git_run_source(run_root: str, campaign: str, commit: str) -> RunSource:
    file_set = set(git_list(MAC_REF, run_root))

    def reader(rel: str) -> str | None:
        return git_show_text(MAC_REF, f"{run_root}/{rel}")

    def exists(rel: str) -> bool:
        return f"{run_root}/{rel}" in file_set

    return RunSource(
        platform="Mac",
        campaign=campaign,
        run_root=run_root,
        source_branch=MAC_REF,
        source_commit=commit,
        reader=reader,
        exists=exists,
    )


def infer_context(campaign: str, run_root: str, manifest: dict[str, Any]) -> dict[str, Any]:
    parts = list(PurePosixPath(run_root).parts)
    cfg = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    identity = manifest.get("identity", {}) if isinstance(manifest, dict) else {}
    training = cfg.get("training", {}) if isinstance(cfg, dict) else {}

    dataset = first(identity.get("dataset"), cfg.get("dataset"))
    phase = "other"
    variant = "default"
    batch_size = as_int(training.get("batch_size"))
    activation = first(training.get("hidden_activation"), nested(cfg, "protocol", "hidden_activation"))
    quant_variant = None

    if "activations" in parts:
        idx = parts.index("activations")
        phase = "activation"
        if idx + 2 < len(parts):
            dataset = dataset or parts[idx + 1]
            activation = parts[idx + 2]
            variant = activation
    elif "batch" in parts:
        idx = parts.index("batch")
        phase = "batch"
        if idx + 1 < len(parts):
            variant = parts[idx + 1]
            match = re.search(r"(\d+)", variant)
            batch_size = int(match.group(1)) if match else batch_size
    elif "smoke" in parts:
        phase = "smoke"
        variant = "smoke"
    elif campaign.startswith("kmnist-alldata-noaug-batch-sweep"):
        phase = "batch"
        candidate = next((part for part in parts if part.startswith("batch-")), "batch")
        variant = candidate
        match = re.search(r"(\d+)", candidate)
        batch_size = int(match.group(1)) if match else batch_size
    elif campaign.startswith("kmnist-batch32-noaugmentation-control"):
        phase = "control"
        variant = "batch-032-noaugmentation"
        batch_size = 32
    elif campaign.startswith("kmnist-quantization"):
        phase = "quantization"
        campaign_index = parts.index(campaign) if campaign in parts else 1
        quant_variant = parts[campaign_index + 1] if campaign_index + 1 < len(parts) else "unknown"
        variant = quant_variant
    elif campaign == "quantization_all":
        phase = "quantization"
        campaign_index = parts.index(campaign) if campaign in parts else 1
        if campaign_index + 2 < len(parts):
            dataset = dataset or parts[campaign_index + 1]
            quant_variant = parts[campaign_index + 2]
        else:
            quant_variant = "unknown"
        variant = quant_variant
    elif campaign == "remaining-ram-capped":
        phase = "balance"
        variant = first(identity.get("balance_mode"), cfg.get("balance_mode"), "unknown")

    if not dataset:
        if "runs" in parts:
            runs_idx = parts.index("runs")
            if runs_idx > 0:
                dataset = parts[runs_idx - 1]
        dataset = dataset or "unknown"

    return {
        "phase": phase,
        "variant": str(variant),
        "batch_size": batch_size,
        "activation": activation,
        "quantization_variant": quant_variant,
        "dataset_key": str(dataset),
        "dataset": DATASET_LABELS.get(str(dataset), str(dataset)),
    }


def environment_fields(environment: dict[str, Any] | None, platform_hint: str) -> dict[str, Any]:
    environment = environment if isinstance(environment, dict) else {}
    hardware = environment.get("hardware", {}) if isinstance(environment, dict) else {}
    platform = environment.get("platform", {}) if isinstance(environment, dict) else {}
    packages = environment.get("packages", {}) if isinstance(environment, dict) else {}
    gpus = hardware.get("gpus") or []
    first_gpu = gpus[0] if gpus and isinstance(gpus[0], dict) else {}
    system = platform.get("system")
    release = platform.get("release")
    if platform_hint == "Windows":
        host_runtime = "Windows via WSL2" if "microsoft" in str(release).lower() else "Windows"
    else:
        host_runtime = "macOS"
    return {
        "host_runtime": host_runtime,
        "runtime_system": system,
        "runtime_release": release,
        "runtime_machine": platform.get("machine"),
        "cpu_logical": as_int(hardware.get("cpu_count_logical")),
        "ram_total_bytes": as_float(hardware.get("ram_total_bytes")),
        "gpu_name": first(first_gpu.get("name"), hardware.get("gpu_name")),
        "gpu_backend": first(hardware.get("gpu_backend"), environment.get("gpu_backend")),
        "gpu_memory_total_bytes": as_float(first_gpu.get("memory_total_bytes")),
        "python_version": nested(environment, "python", "version"),
        "tensorflow_version": packages.get("tensorflow"),
        "tensorflow_metal_version": packages.get("tensorflow_metal"),
    }


def telemetry_flat(summary: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = summary if isinstance(summary, dict) else {}
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    wide: dict[str, Any] = {
        "telemetry_sample_count": as_int(summary.get("sample_count")),
        "telemetry_interval_seconds": as_float(summary.get("interval_seconds")),
        "telemetry_elapsed_seconds": as_float(summary.get("elapsed_seconds")),
        "telemetry_gpu_backend": summary.get("gpu_backend"),
    }
    long_rows: list[dict[str, Any]] = []
    selected = {
        "cpu_percent": "cpu_pct",
        "process_cpu_percent": "process_cpu_pct",
        "ram_percent": "ram_pct",
        "ram_used_bytes": "ram_used_bytes",
        "process_rss_bytes": "process_rss_bytes",
        "gpu_utilization_percent": "gpu_util_pct",
        "gpu_memory_percent": "gpu_memory_pct",
        "gpu_memory_used_bytes": "gpu_memory_used_bytes",
        "gpu_power_w": "gpu_power_w",
        "gpu_temperature_c": "gpu_temperature_c",
    }
    for metric_name, values in metrics.items():
        if not isinstance(values, dict):
            continue
        for stat, value in values.items():
            numeric = as_float(value)
            if numeric is not None:
                long_rows.append({"metric": metric_name, "stat": stat, "value": numeric})
        if metric_name in selected:
            prefix = selected[metric_name]
            for stat in ("mean", "p50", "p95", "max", "min", "count"):
                wide[f"{prefix}_{stat}"] = as_float(values.get(stat))
    return wide, long_rows


def extract_per_class(
    class_report: dict[str, Any] | None,
    test_metrics: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    class_report = class_report if isinstance(class_report, dict) else {}
    test_metrics = test_metrics if isinstance(test_metrics, dict) else {}
    class_names = first(class_report.get("class_names"), nested(manifest, "config", "class_names"), []) or []
    rows: list[dict[str, Any]] = []
    report_rows = class_report.get("per_class")
    if isinstance(report_rows, list):
        for idx, item in enumerate(report_rows):
            if not isinstance(item, dict):
                continue
            class_index = as_int(first(item.get("class_index"), item.get("label"), idx))
            class_name = first(
                item.get("class_name"),
                class_names[class_index] if class_index is not None and 0 <= class_index < len(class_names) else None,
                str(item.get("label", class_index)),
            )
            rows.append(
                {
                    "class_index": class_index,
                    "class_label": item.get("label", class_index),
                    "class_name": class_name,
                    "support": as_int(item.get("support")),
                    "precision": as_float(item.get("precision")),
                    "recall": as_float(item.get("recall")),
                    "f1": as_float(first(item.get("f1"), item.get("f1-score"))),
                }
            )
        return rows

    per_class = nested(test_metrics, "classification", "per_class", default={})
    if isinstance(per_class, dict):
        for key, item in sorted(per_class.items(), key=lambda pair: int(pair[0]) if str(pair[0]).isdigit() else str(pair[0])):
            if not isinstance(item, dict):
                continue
            class_index = as_int(first(item.get("label"), key))
            class_name = first(
                class_names[class_index] if class_index is not None and 0 <= class_index < len(class_names) else None,
                str(key),
            )
            rows.append(
                {
                    "class_index": class_index,
                    "class_label": item.get("label", class_index),
                    "class_name": class_name,
                    "support": as_int(item.get("support")),
                    "precision": as_float(item.get("precision")),
                    "recall": as_float(item.get("recall")),
                    "f1": as_float(first(item.get("f1"), item.get("f1-score"))),
                }
            )
    return rows


def extract_confusion(
    test_metrics: dict[str, Any] | None,
    per_class: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matrix = nested(test_metrics or {}, "classification", "confusion_matrix")
    if not isinstance(matrix, list) or not matrix:
        return []
    names = {row.get("class_index"): row.get("class_name") for row in per_class}
    rows: list[dict[str, Any]] = []
    for true_index, line in enumerate(matrix):
        if not isinstance(line, list):
            continue
        row_total = sum(as_float(value) or 0.0 for value in line)
        for pred_index, value in enumerate(line):
            count = as_int(value)
            rows.append(
                {
                    "true_index": true_index,
                    "true_class": names.get(true_index, str(true_index)),
                    "pred_index": pred_index,
                    "pred_class": names.get(pred_index, str(pred_index)),
                    "count": count,
                    "row_total": as_int(row_total),
                    "row_share": (count / row_total) if count is not None and row_total else None,
                }
            )
    return rows


def extract_epochs(source: RunSource) -> tuple[list[dict[str, Any]], int]:
    text = first(source.text("logs/epoch_metrics.csv"), source.text("artifacts/history.csv"))
    if not text:
        return [], 0
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return [], 0
    if df.empty:
        return [], 0
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    duplicates = int(df.duplicated(subset=["epoch"], keep="last").sum())
    df = df.dropna(subset=["epoch"]).drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
    keep = [
        "epoch",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_macro_f1",
        "learning_rate",
        "epoch_seconds",
        "train_examples",
        "train_examples_per_second",
    ]
    rows: list[dict[str, Any]] = []
    for _, item in df.iterrows():
        row = {column: clean_scalar(item.get(column)) for column in keep if column in df.columns}
        row["epoch"] = as_int(row.get("epoch"))
        rows.append(row)
    return rows, duplicates


def extract_telemetry_profile(source: RunSource) -> list[dict[str, Any]]:
    text = source.text("telemetry/samples.csv")
    if not text:
        return []
    try:
        header = pd.read_csv(io.StringIO(text), nrows=0).columns.tolist()
    except Exception:
        return []
    wanted = [
        "elapsed_seconds",
        "cpu_percent",
        "process_cpu_percent",
        "ram_percent",
        "process_rss_bytes",
        "gpu_utilization_percent",
        "gpu_memory_used_bytes",
        "gpu_power_w",
        "gpu_temperature_c",
    ]
    usecols = [column for column in wanted if column in header]
    if "elapsed_seconds" not in usecols:
        return []
    try:
        df = pd.read_csv(io.StringIO(text), usecols=usecols, on_bad_lines="skip")
    except Exception:
        return []
    for column in usecols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["elapsed_seconds"])
    if df.empty:
        return []
    max_elapsed = float(df["elapsed_seconds"].max())
    if max_elapsed <= 0:
        return []
    df["time_bin"] = np.minimum(
        TELEMETRY_PROFILE_BINS - 1,
        np.floor(df["elapsed_seconds"] / max_elapsed * TELEMETRY_PROFILE_BINS),
    ).astype(int)
    numeric_cols = [column for column in usecols if column != "elapsed_seconds"]
    grouped = df.groupby("time_bin", as_index=False)[numeric_cols].mean(numeric_only=True)
    counts = df.groupby("time_bin").size().to_dict()
    rows: list[dict[str, Any]] = []
    for _, item in grouped.iterrows():
        bin_index = int(item["time_bin"])
        row: dict[str, Any] = {
            "time_bin": bin_index,
            "elapsed_fraction_mid": (bin_index + 0.5) / TELEMETRY_PROFILE_BINS,
            "sample_count_bin": int(counts.get(bin_index, 0)),
        }
        for column in numeric_cols:
            row[column] = clean_scalar(item.get(column))
        rows.append(row)
    return rows


def extract_raw_run(source: RunSource) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest = source.json("manifest.json")
    manifest = manifest if isinstance(manifest, dict) else {}
    status = source.json("status.json")
    status = status if isinstance(status, dict) else {}
    test_metrics = source.json("artifacts/test_metrics.json")
    if not isinstance(test_metrics, dict):
        test_metrics = manifest.get("test_metrics", {}) if isinstance(manifest.get("test_metrics"), dict) else {}
    class_report = source.json("artifacts/classification_report.json")
    class_report = class_report if isinstance(class_report, dict) else {}
    training_summary = source.json("logs/training_summary.json")
    if not isinstance(training_summary, dict):
        training_summary = manifest.get("training", {}) if isinstance(manifest.get("training"), dict) else {}
    telemetry_summary = source.json("telemetry/summary.json")
    if not isinstance(telemetry_summary, dict):
        telemetry_summary = manifest.get("telemetry", {}) if isinstance(manifest.get("telemetry"), dict) else {}
    environment = source.json("telemetry/environment.json")
    environment = environment if isinstance(environment, dict) else {}

    context = infer_context(source.campaign, source.run_root, manifest)
    cfg = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    identity = manifest.get("identity", {}) if isinstance(manifest, dict) else {}
    cfg_training = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    classification = test_metrics.get("classification", {}) if isinstance(test_metrics, dict) else {}
    keras_metrics = test_metrics.get("keras_metrics", {}) if isinstance(test_metrics, dict) else {}
    test_section = class_report.get("test", {}) if isinstance(class_report, dict) else {}
    telemetry_wide, telemetry_long = telemetry_flat(telemetry_summary)
    env_fields = environment_fields(environment, source.platform)
    epochs, epoch_duplicates = extract_epochs(source)
    per_class = extract_per_class(class_report, test_metrics, manifest)
    confusion = extract_confusion(test_metrics, per_class)
    telemetry_profile = extract_telemetry_profile(source)

    run_id = first(manifest.get("run_id"), status.get("run_id"), PurePosixPath(source.run_root).name)
    attempt = as_int(first(manifest.get("attempt"), status.get("attempt"), 1)) or 1
    seed = as_int(first(identity.get("seed"), cfg.get("seed"), 42))
    uid_parts = [
        source.platform,
        source.campaign,
        context["phase"],
        context["dataset_key"],
        context["variant"],
        f"seed-{seed}",
        f"attempt-{attempt}",
    ]
    run_uid = "|".join(slug(part) for part in uid_parts)
    status_value = first(status.get("status"), manifest.get("status"), "unknown")
    evidence_level = "raw_complete" if status_value == "completed" else "raw_partial"
    run: dict[str, Any] = {
        "snapshot_at": SNAPSHOT_AT,
        "platform": source.platform,
        "host_runtime": env_fields.get("host_runtime"),
        "machine_label": first(env_fields.get("gpu_name"), env_fields.get("runtime_machine"), source.platform),
        "source_branch": source.source_branch,
        "source_commit": source.source_commit,
        "campaign": source.campaign,
        "phase": context["phase"],
        "dataset_key": context["dataset_key"],
        "dataset": context["dataset"],
        "variant": context["variant"],
        "batch_size": context["batch_size"],
        "activation": context["activation"],
        "quantization_variant": context["quantization_variant"],
        "normalization": first(identity.get("normalization"), cfg.get("normalization")),
        "balance_mode": first(identity.get("balance_mode"), cfg.get("balance_mode")),
        "seed": seed,
        "attempt": attempt,
        "run_id": run_id,
        "run_uid": run_uid,
        "status": status_value,
        "status_updated_at": first(status.get("status_updated_at"), manifest.get("status_updated_at")),
        "evidence_level": evidence_level,
        "source_path": source.run_root,
        "has_manifest": bool(manifest),
        "has_test_metrics": bool(test_metrics),
        "has_class_metrics": bool(per_class),
        "has_confusion_matrix": bool(confusion),
        "has_epoch_metrics": bool(epochs),
        "has_training_summary": bool(training_summary),
        "has_telemetry_summary": bool(telemetry_summary),
        "has_telemetry_samples": source.exists("telemetry/samples.csv"),
        "has_environment": bool(environment),
        "epoch_duplicate_rows": epoch_duplicates,
        "epochs_completed": as_int(first(training_summary.get("epochs_completed"), len(epochs) if epochs else None)),
        "max_epochs": as_int(first(training_summary.get("max_epochs"), cfg_training.get("max_epochs"))),
        "test_samples": as_int(first(test_section.get("n_samples"), sum(row.get("support") or 0 for row in per_class) or None)),
        "accuracy": as_float(first(classification.get("accuracy"), keras_metrics.get("accuracy"), test_section.get("accuracy"))),
        "balanced_accuracy": as_float(first(classification.get("balanced_accuracy"), test_section.get("balanced_accuracy"))),
        "macro_f1": as_float(first(classification.get("macro_f1"), test_section.get("macro_f1"))),
        "macro_precision": as_float(classification.get("macro_precision")),
        "macro_recall": as_float(classification.get("macro_recall")),
        "loss": as_float(first(keras_metrics.get("loss"), test_section.get("loss"))),
        "training_seconds": as_float(first(training_summary.get("training_seconds"), training_summary.get("total_seconds"))),
        "evaluation_seconds": as_float(training_summary.get("evaluation_seconds")),
        "wall_seconds": as_float(first(training_summary.get("wall_seconds_current_attempt"), telemetry_summary.get("elapsed_seconds"))),
        "mean_epoch_seconds": as_float(training_summary.get("mean_epoch_seconds")),
        "mean_train_examples_per_second": as_float(training_summary.get("mean_train_examples_per_second")),
        "extra_fraction": as_float(cfg_training.get("extra_fraction")),
        "dtype_policy": first(cfg_training.get("dtype_policy"), nested(cfg, "protocol", "dtype_policy")),
        "learning_rate": as_float(cfg_training.get("learning_rate")),
        "target_size": as_int(cfg.get("target_size")),
        "num_classes": as_int(cfg.get("num_classes")),
        "split_fingerprint": first(manifest.get("split_fingerprint"), nested(manifest, "data_metadata", "split", "fingerprint")),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "source_fingerprint": cfg.get("source_fingerprint"),
        **env_fields,
        **telemetry_wide,
    }
    run["training_hours"] = run["training_seconds"] / 3600 if run["training_seconds"] is not None else None

    for collection in (per_class, confusion, epochs, telemetry_long, telemetry_profile):
        for item in collection:
            item.update(
                {
                    "run_uid": run_uid,
                    "platform": source.platform,
                    "campaign": source.campaign,
                    "phase": context["phase"],
                    "dataset_key": context["dataset_key"],
                    "dataset": context["dataset"],
                    "variant": context["variant"],
                }
            )
    return run, per_class, confusion, epochs, telemetry_long, telemetry_profile


def initial_windows_inventory(commit: str) -> tuple[list[RunSource], list[dict[str, Any]]]:
    sources: list[RunSource] = []
    inventory: list[dict[str, Any]] = []
    for status_path in sorted(OUTPUTS.rglob("status.json")):
        run_root = status_path.parent
        rel_parts = status_path.relative_to(OUTPUTS).parts
        if not rel_parts:
            continue
        campaign = rel_parts[0]
        status = read_json(status_path)
        status = status if isinstance(status, dict) else {}
        row = {
            "snapshot_at": SNAPSHOT_AT,
            "platform": "Windows",
            "campaign": campaign,
            "source_path": relative_repo(run_root),
            "status": status.get("status", "unreadable"),
            "run_id": status.get("run_id"),
            "status_updated_at": status.get("status_updated_at"),
            "evidence_level": "raw_status",
        }
        inventory.append(row)
        if row["status"] == "completed":
            sources.append(local_run_source(run_root, campaign, commit))
    return sources, inventory


def mac_raw_sources(commit: str) -> tuple[list[RunSource], list[dict[str, Any]]]:
    sources: list[RunSource] = []
    inventory: list[dict[str, Any]] = []
    paths = [
        path
        for path in git_list(MAC_REF, f"outputs/{MAC_RAW_CAMPAIGN}")
        if path.endswith("/status.json")
    ]
    for path in sorted(paths):
        run_root = path[: -len("/status.json")]
        status = json_loads(git_show_text(MAC_REF, path))
        status = status if isinstance(status, dict) else {}
        inventory.append(
            {
                "snapshot_at": SNAPSHOT_AT,
                "platform": "Mac",
                "campaign": MAC_RAW_CAMPAIGN,
                "source_path": run_root,
                "status": status.get("status", "unreadable"),
                "run_id": status.get("run_id"),
                "status_updated_at": status.get("status_updated_at"),
                "evidence_level": "raw_status",
            }
        )
        if status.get("status") == "completed":
            sources.append(git_run_source(run_root, MAC_RAW_CAMPAIGN, commit))
    return sources, inventory


def extract_mac_activation_final(
    commit: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    csv_path = "docs/augmentation05_final/data/resultados_augmentation05.csv"
    results = read_csv_text(git_show_text(MAC_REF, csv_path))
    summary = json_loads(git_show_text(MAC_REF, "docs/augmentation05_final/data/resumo_augmentation05.json"))
    summary = summary if isinstance(summary, dict) else {}
    hardware = summary.get("hardware", {}) if isinstance(summary.get("hardware"), dict) else {}
    runs: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    telemetry_long: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    for _, item in results.iterrows():
        dataset_key = str(item.get("dataset_key"))
        activation = str(item.get("activation_key"))
        metric_path = str(item.get("metric_path"))
        run_root = metric_path.rsplit("/artifacts/test_metrics.json", 1)[0]
        class_path = f"{run_root}/artifacts/classification_report.json"
        telemetry_path = f"{run_root}/telemetry/summary.json"
        status_path = f"{run_root}/status.json"
        class_report = json_loads(git_show_text(MAC_REF, class_path))
        class_report = class_report if isinstance(class_report, dict) else {}
        telemetry = json_loads(git_show_text(MAC_REF, telemetry_path))
        telemetry = telemetry if isinstance(telemetry, dict) else {}
        status = json_loads(git_show_text(MAC_REF, status_path))
        status = status if isinstance(status, dict) else {}
        compact_available = bool(class_report or telemetry or status)
        evidence_level = "raw_compact_plus_consolidated" if compact_available else "consolidated_report_only"
        run_uid = "|".join(
            slug(part)
            for part in ["Mac", MAC_ACTIVATION_CAMPAIGN, "activation", dataset_key, activation, "seed-42", "attempt-1"]
        )
        telemetry_wide, telemetry_rows = telemetry_flat(telemetry)
        fallback_telemetry = {
            "telemetry_sample_count": as_int(item.get("telemetry_samples")),
            "gpu_util_pct_mean": as_float(item.get("gpu_util_mean_pct")),
            "gpu_util_pct_max": as_float(item.get("gpu_util_max_pct")),
            "gpu_memory_used_bytes_mean": as_float(item.get("gpu_memory_mean_bytes")),
            "gpu_memory_used_bytes_max": as_float(item.get("peak_gpu_memory_bytes")),
            "process_cpu_pct_mean": as_float(item.get("process_cpu_mean_pct")),
            "process_cpu_pct_max": as_float(item.get("peak_process_cpu_pct")),
            "process_rss_bytes_mean": as_float(item.get("process_rss_mean_bytes")),
            "process_rss_bytes_max": as_float(item.get("peak_process_rss_bytes")),
            "ram_pct_mean": as_float(item.get("ram_mean_pct")),
            "ram_pct_max": as_float(item.get("peak_ram_percent")),
        }
        for key, value in fallback_telemetry.items():
            telemetry_wide[key] = first(telemetry_wide.get(key), value)

        run = {
            "snapshot_at": SNAPSHOT_AT,
            "platform": "Mac",
            "host_runtime": "macOS",
            "machine_label": hardware.get("gpu_name", "Apple M4"),
            "source_branch": MAC_REF,
            "source_commit": commit,
            "campaign": MAC_ACTIVATION_CAMPAIGN,
            "phase": "activation",
            "dataset_key": dataset_key,
            "dataset": first(item.get("dataset"), DATASET_LABELS.get(dataset_key, dataset_key)),
            "variant": activation,
            "batch_size": 256,
            "activation": activation,
            "quantization_variant": None,
            "normalization": "unit_interval",
            "balance_mode": "all_raw",
            "seed": 42,
            "attempt": 1,
            "run_id": f"{dataset_key}__unit_interval__all_raw__seed-42",
            "run_uid": run_uid,
            "status": item.get("status", "completed"),
            "status_updated_at": status.get("status_updated_at"),
            "evidence_level": evidence_level,
            "source_path": metric_path,
            "has_manifest": False,
            "has_test_metrics": not pd.isna(item.get("test_macro_f1")),
            "has_class_metrics": bool(class_report),
            "has_confusion_matrix": False,
            "has_epoch_metrics": False,
            "has_training_summary": True,
            "has_telemetry_summary": True,
            "has_telemetry_samples": False,
            "has_environment": False,
            "epoch_duplicate_rows": 0,
            "epochs_completed": as_int(item.get("epochs_completed")),
            "max_epochs": 100,
            "test_samples": as_int(item.get("test_samples")),
            "accuracy": as_float(item.get("test_accuracy")),
            "balanced_accuracy": as_float(item.get("test_balanced_accuracy")),
            "macro_f1": as_float(item.get("test_macro_f1")),
            "macro_precision": as_float(item.get("test_macro_precision")),
            "macro_recall": as_float(item.get("test_macro_recall")),
            "loss": as_float(item.get("test_loss")),
            "training_seconds": as_float(item.get("training_seconds")),
            "evaluation_seconds": as_float(item.get("evaluation_seconds")),
            "wall_seconds": None,
            "mean_epoch_seconds": as_float(item.get("mean_epoch_seconds")),
            "mean_train_examples_per_second": as_float(item.get("mean_train_examples_per_second")),
            "extra_fraction": 0.5,
            "dtype_policy": "mixed_float16",
            "learning_rate": None,
            "target_size": None,
            "num_classes": None,
            "split_fingerprint": None,
            "config_fingerprint": None,
            "source_fingerprint": None,
            "runtime_system": "Darwin",
            "runtime_release": hardware.get("macos_version"),
            "runtime_machine": "arm64",
            "cpu_logical": as_int(hardware.get("cpu_logical_cores")),
            "ram_total_bytes": (as_float(hardware.get("ram_gib")) or 0) * (1024**3),
            "gpu_name": hardware.get("gpu_name", "Apple M4"),
            "gpu_backend": hardware.get("gpu_backend"),
            "gpu_memory_total_bytes": None,
            "python_version": hardware.get("python_version"),
            "tensorflow_version": hardware.get("tensorflow"),
            "tensorflow_metal_version": hardware.get("tensorflow_metal"),
            "thermal_pressure": item.get("thermal_pressure"),
            **telemetry_wide,
        }
        run["training_hours"] = run["training_seconds"] / 3600 if run["training_seconds"] is not None else None
        runs.append(run)
        inventory.append(
            {
                "snapshot_at": SNAPSHOT_AT,
                "platform": "Mac",
                "campaign": MAC_ACTIVATION_CAMPAIGN,
                "source_path": metric_path,
                "status": run["status"],
                "run_id": run["run_id"],
                "status_updated_at": run["status_updated_at"],
                "evidence_level": evidence_level,
            }
        )
        evidence_rows.append(
            {
                "run_uid": run_uid,
                "platform": "Mac",
                "campaign": MAC_ACTIVATION_CAMPAIGN,
                "dataset": run["dataset"],
                "variant": activation,
                "evidence_level": evidence_level,
                "overall_metrics": True,
                "class_metrics": bool(class_report),
                "confusion_matrix": False,
                "epoch_metrics": False,
                "telemetry_summary": True,
                "telemetry_samples": False,
                "note": (
                    "Métricas finais exatas e telemetria agregada; relatório por classe compacto."
                    if compact_available
                    else "Métricas finais exatas apenas no CSV consolidado; artefatos por classe, matriz e épocas não versionados."
                ),
            }
        )

        per_class = extract_per_class(class_report, {}, {})
        for class_row in per_class:
            class_row.update(
                {
                    "run_uid": run_uid,
                    "platform": "Mac",
                    "campaign": MAC_ACTIVATION_CAMPAIGN,
                    "phase": "activation",
                    "dataset_key": dataset_key,
                    "dataset": run["dataset"],
                    "variant": activation,
                }
            )
            classes.append(class_row)
        for telemetry_row in telemetry_rows:
            telemetry_row.update(
                {
                    "run_uid": run_uid,
                    "platform": "Mac",
                    "campaign": MAC_ACTIVATION_CAMPAIGN,
                    "phase": "activation",
                    "dataset_key": dataset_key,
                    "dataset": run["dataset"],
                    "variant": activation,
                }
            )
            telemetry_long.append(telemetry_row)
    return runs, classes, telemetry_long, inventory, evidence_rows


def reported_mac_batch_cells(raw_run_keys: set[tuple[str, int]]) -> pd.DataFrame:
    completed = {
        "mnist": [32, 64, 256],
        "fashion_mnist": [128, 256],
        "kmnist": [32, 64, 128, 256],
        "emnist_balanced": [64, 128, 256],
        "cifar10": [32, 64, 128, 256],
    }
    best = {
        "mnist": (64, 0.9908),
        "fashion_mnist": (128, 0.9112),
        "kmnist": (32, 0.9863),
        "emnist_balanced": (64, 0.8780),
        "cifar10": (32, 0.7235),
    }
    rows: list[dict[str, Any]] = []
    for dataset_key, batches in completed.items():
        best_batch, best_macro_f1 = best[dataset_key]
        for batch in batches:
            raw_available = (dataset_key, batch) in raw_run_keys
            rows.append(
                {
                    "platform": "Mac",
                    "campaign": MAC_RAW_CAMPAIGN,
                    "snapshot_date": "2026-09-14",
                    "dataset_key": dataset_key,
                    "dataset": DATASET_LABELS[dataset_key],
                    "batch_size": batch,
                    "status": "completed",
                    "raw_artifact_versioned": raw_available,
                    "evidence_level": "raw_complete" if raw_available else "progress_report_only",
                    "is_reported_best": batch == best_batch,
                    "reported_macro_f1": best_macro_f1 if batch == best_batch else None,
                    "reported_precision": "rounded_4_decimals" if batch == best_batch else None,
                    "source_path": "analysis_reports/relatorio_treinamento_2026-09-14.md",
                }
            )
    return pd.DataFrame(rows)


def reported_mac_quantization_status() -> pd.DataFrame:
    datasets = list(DATASET_LABELS)
    completed = {
        ("mnist", "fp32"),
        ("mnist", "fp16"),
        ("fashion_mnist", "fp32"),
        ("fashion_mnist", "fp16"),
        ("kmnist", "fp32"),
        ("kmnist", "fp16"),
        ("emnist_balanced", "fp32"),
    }
    stale = {("emnist_balanced", "fp16")}
    rows = []
    for dataset_key in datasets:
        for variant in ("fp32", "fp16"):
            key = (dataset_key, variant)
            status = "completed" if key in completed else "partial_stale" if key in stale else "not_started"
            rows.append(
                {
                    "platform": "Mac",
                    "campaign": "controlled-quantization-fast-mac-m4",
                    "snapshot_date": "2026-09-14",
                    "dataset_key": dataset_key,
                    "dataset": DATASET_LABELS[dataset_key],
                    "variant": variant,
                    "status": status,
                    "epochs_observed": 66 if key in stale else 100 if key in completed else 0,
                    "has_final_metrics_reported": key in completed,
                    "exact_metric_values_available": False,
                    "evidence_level": "progress_report_only",
                    "source_path": "analysis_reports/relatorio_progresso_quantizacao_2026-09-10.md",
                }
            )
    return pd.DataFrame(rows)


def evidence_row_from_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_uid": run["run_uid"],
        "platform": run["platform"],
        "campaign": run["campaign"],
        "dataset": run["dataset"],
        "variant": run["variant"],
        "evidence_level": run["evidence_level"],
        "overall_metrics": bool(run["has_test_metrics"]),
        "class_metrics": bool(run["has_class_metrics"]),
        "confusion_matrix": bool(run["has_confusion_matrix"]),
        "epoch_metrics": bool(run["has_epoch_metrics"]),
        "telemetry_summary": bool(run["has_telemetry_summary"]),
        "telemetry_samples": bool(run["has_telemetry_samples"]),
        "note": "Artefatos brutos por execução disponíveis.",
    }


def pair_rows(windows: pd.DataFrame, mac: pd.DataFrame, pair_type: str) -> pd.DataFrame:
    if pair_type == "batch":
        left = windows[(windows["campaign"] == WINDOWS_CAMPAIGN) & (windows["phase"] == "batch")]
        right = mac[(mac["campaign"] == MAC_RAW_CAMPAIGN) & (mac["phase"] == "batch")]
        keys = ["dataset_key", "batch_size"]
    else:
        left = windows[(windows["campaign"] == WINDOWS_CAMPAIGN) & (windows["phase"] == "activation")]
        right = mac[(mac["campaign"] == MAC_ACTIVATION_CAMPAIGN) & (mac["phase"] == "activation")]
        keys = ["dataset_key", "activation"]

    merged = left.merge(right, on=keys, how="inner", suffixes=("_windows", "_mac"), validate="one_to_one")
    rows: list[dict[str, Any]] = []
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "loss",
        "training_seconds",
        "mean_epoch_seconds",
        "mean_train_examples_per_second",
        "gpu_util_pct_mean",
        "gpu_util_pct_max",
        "gpu_memory_used_bytes_mean",
        "gpu_memory_used_bytes_max",
        "process_cpu_pct_mean",
        "process_rss_bytes_mean",
        "ram_pct_mean",
        "ram_pct_max",
    ]
    for _, item in merged.iterrows():
        batch_windows = item.get("batch_size") if pair_type == "batch" else item.get("batch_size_windows")
        batch_mac = item.get("batch_size") if pair_type == "batch" else item.get("batch_size_mac")
        activation_value = item.get("activation") if pair_type == "activation" else first(item.get("activation_windows"), item.get("activation_mac"))
        same_batch = batch_windows == batch_mac
        same_extra = item.get("extra_fraction_windows") == item.get("extra_fraction_mac")
        same_seed = item.get("seed_windows") == item.get("seed_mac")
        same_split = (
            item.get("split_fingerprint_windows") == item.get("split_fingerprint_mac")
            if pd.notna(item.get("split_fingerprint_windows")) and pd.notna(item.get("split_fingerprint_mac"))
            else None
        )
        if pair_type == "batch":
            comparability = "A_protocol_aligned_runtime_differs" if same_batch and same_extra and same_seed else "C_protocol_mismatch"
            limitation = "Mesmo dataset, batch, augmentation, seed e divisão quando verificável; hardware, SO e versão do TensorFlow diferem."
        else:
            comparability = "B_descriptive_batch_mismatch"
            limitation = f"Mesmo dataset, ativação, augmentation e seed, mas Windows usa batch {as_int(batch_windows)} e Mac batch {as_int(batch_mac)}; fingerprint Mac não está versionado."
        row: dict[str, Any] = {
            "pair_type": pair_type,
            "pair_id": f"{pair_type}|{item['dataset_key']}|{batch_windows if pair_type == 'batch' else activation_value}",
            "dataset_key": item["dataset_key"],
            "dataset": item.get("dataset_windows"),
            "variant": (
                f"batch {as_int(batch_windows)}"
                if pair_type == "batch"
                else str(activation_value)
            ),
            "batch_size_windows": as_int(batch_windows),
            "batch_size_mac": as_int(batch_mac),
            "activation": activation_value,
            "windows_run_uid": item.get("run_uid_windows"),
            "mac_run_uid": item.get("run_uid_mac"),
            "comparability_grade": comparability,
            "same_batch_size": same_batch,
            "same_extra_fraction": same_extra,
            "same_seed": same_seed,
            "same_split_fingerprint": same_split,
            "tensorflow_windows": item.get("tensorflow_version_windows"),
            "tensorflow_mac": item.get("tensorflow_version_mac"),
            "limitation": limitation,
        }
        for metric in metrics:
            win = as_float(item.get(f"{metric}_windows"))
            mac_value = as_float(item.get(f"{metric}_mac"))
            row[f"{metric}_windows"] = win
            row[f"{metric}_mac"] = mac_value
            row[f"delta_{metric}_windows_minus_mac"] = (
                win - mac_value if win is not None and mac_value is not None else None
            )
        row["delta_accuracy_pp_windows_minus_mac"] = (
            row["delta_accuracy_windows_minus_mac"] * 100
            if row.get("delta_accuracy_windows_minus_mac") is not None
            else None
        )
        row["delta_macro_f1_pp_windows_minus_mac"] = (
            row["delta_macro_f1_windows_minus_mac"] * 100
            if row.get("delta_macro_f1_windows_minus_mac") is not None
            else None
        )
        mac_time = row.get("training_seconds_mac")
        win_time = row.get("training_seconds_windows")
        row["training_time_ratio_windows_over_mac"] = (
            win_time / mac_time if win_time is not None and mac_time not in (None, 0) else None
        )
        mac_epoch = row.get("mean_epoch_seconds_mac")
        win_epoch = row.get("mean_epoch_seconds_windows")
        row["epoch_time_ratio_windows_over_mac"] = (
            win_epoch / mac_epoch if win_epoch is not None and mac_epoch not in (None, 0) else None
        )
        if row.get("macro_f1_windows") is not None and row.get("macro_f1_mac") is not None:
            row["macro_f1_winner"] = "Windows" if row["macro_f1_windows"] > row["macro_f1_mac"] else "Mac" if row["macro_f1_mac"] > row["macro_f1_windows"] else "Empate"
        if win_time is not None and mac_time is not None:
            row["training_time_winner"] = "Windows" if win_time < mac_time else "Mac" if mac_time < win_time else "Empate"
        rows.append(row)
    return pd.DataFrame(rows)


def campaign_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (platform, campaign, phase), group in runs.groupby(["platform", "campaign", "phase"], dropna=False):
        weights = pd.to_numeric(group["telemetry_sample_count"], errors="coerce").fillna(0)
        gpu = pd.to_numeric(group["gpu_util_pct_mean"], errors="coerce")
        ram = pd.to_numeric(group["ram_pct_mean"], errors="coerce")
        def weighted(series: pd.Series) -> float | None:
            valid = series.notna() & weights.gt(0)
            if valid.any():
                return float(np.average(series[valid], weights=weights[valid]))
            return float(series.mean()) if series.notna().any() else None
        rows.append(
            {
                "platform": platform,
                "campaign": campaign,
                "phase": phase,
                "completed_runs": int(len(group)),
                "datasets": int(group["dataset_key"].nunique()),
                "epochs": int(pd.to_numeric(group["epochs_completed"], errors="coerce").fillna(0).sum()),
                "training_hours": float(pd.to_numeric(group["training_seconds"], errors="coerce").fillna(0).sum() / 3600),
                "mean_macro_f1": clean_scalar(pd.to_numeric(group["macro_f1"], errors="coerce").mean()),
                "median_macro_f1": clean_scalar(pd.to_numeric(group["macro_f1"], errors="coerce").median()),
                "mean_accuracy": clean_scalar(pd.to_numeric(group["accuracy"], errors="coerce").mean()),
                "median_epoch_seconds": clean_scalar(pd.to_numeric(group["mean_epoch_seconds"], errors="coerce").median()),
                "weighted_gpu_util_mean_pct": weighted(gpu),
                "weighted_ram_mean_pct": weighted(ram),
                "telemetry_samples": int(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def validate_data(
    runs: pd.DataFrame,
    classes: pd.DataFrame,
    confusion: pd.DataFrame,
    epochs: pd.DataFrame,
    activation_pairs: pd.DataFrame,
    batch_pairs: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(check: str, status: str, observed: Any, expected: Any, detail: str) -> None:
        checks.append(
            {
                "check": check,
                "status": status,
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    duplicate_count = int(runs["run_uid"].duplicated().sum())
    add("run_uid_unique", "pass" if duplicate_count == 0 else "fail", duplicate_count, 0, "Chave composta por plataforma, campanha, fase, dataset, variante, seed e tentativa.")

    invalid_metric_count = 0
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall"):
        values = pd.to_numeric(runs[metric], errors="coerce")
        invalid_metric_count += int(((values < 0) | (values > 1)).sum())
    add("metrics_in_unit_interval", "pass" if invalid_metric_count == 0 else "fail", invalid_metric_count, 0, "Métricas de classificação devem permanecer entre 0 e 1.")

    class_diff_max = 0.0
    class_runs_checked = 0
    if not classes.empty:
        class_means = classes.groupby("run_uid")["f1"].mean()
        declared = runs.set_index("run_uid")["macro_f1"]
        aligned = pd.concat([class_means.rename("recomputed"), declared.rename("declared")], axis=1).dropna()
        if not aligned.empty:
            class_diff_max = float((aligned["recomputed"] - aligned["declared"]).abs().max())
            class_runs_checked = int(len(aligned))
    add("macro_f1_recomputed_from_classes", "pass" if class_diff_max <= 1e-9 else "warn", class_diff_max, "<=1e-9", f"{class_runs_checked} runs com relatório por classe comparável.")

    confusion_mismatches = 0
    confusion_runs_checked = 0
    if not confusion.empty:
        matrix_totals = confusion.groupby("run_uid")["count"].sum()
        test_samples = runs.set_index("run_uid")["test_samples"]
        aligned = pd.concat([matrix_totals.rename("matrix"), test_samples.rename("samples")], axis=1).dropna()
        confusion_runs_checked = int(len(aligned))
        confusion_mismatches = int((aligned["matrix"] != aligned["samples"]).sum())
    add("confusion_matrix_totals", "pass" if confusion_mismatches == 0 else "warn", confusion_mismatches, 0, f"{confusion_runs_checked} matrizes confrontadas com o total de teste.")

    epoch_mismatches = 0
    epoch_runs_checked = 0
    if not epochs.empty:
        epoch_counts = epochs.groupby("run_uid")["epoch"].nunique()
        expected = runs.set_index("run_uid")["epochs_completed"]
        aligned = pd.concat([epoch_counts.rename("rows"), expected.rename("expected")], axis=1).dropna()
        epoch_runs_checked = int(len(aligned))
        epoch_mismatches = int((aligned["rows"] != aligned["expected"]).sum())
    add("epoch_counts_match_summary", "pass" if epoch_mismatches == 0 else "warn", epoch_mismatches, 0, f"{epoch_runs_checked} runs com série por época.")

    add("activation_pair_count", "pass" if len(activation_pairs) == 21 else "warn", len(activation_pairs), 21, "Pares disponíveis para sete datasets e três ativações; GTSRB e FER2013 ainda não têm resultado Windows final no snapshot.")
    add("batch_pair_count", "pass" if len(batch_pairs) == 6 else "warn", len(batch_pairs), 6, "Pares com artefatos brutos versionados nos dois ambientes.")

    failed = [row for row in checks if row["status"] == "fail"]
    if failed:
        raise RuntimeError(f"Critical data checks failed: {failed}")
    return pd.DataFrame(checks)


def reviewed_query(rows: list[dict[str, Any]], label: str, files: list[str], definitions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": [{key: clean_scalar(value) for key, value in row.items()} for row in rows],
        "source": {
            "label": label,
            "files": files,
            "filters": [f"Snapshot UTC: {SNAPSHOT_AT}", "Runs concluídos com métricas finais; estados parciais permanecem em tabelas de cobertura."],
            "executedAt": SNAPSHOT_AT,
            "metricDefinitions": definitions,
            "evidenceFlow": [
                {"title": "Windows", "detail": "Artefatos locais em outputs/, capturados sem modificar os treinamentos."},
                {"title": "Mac", "detail": f"Artefatos e relatórios lidos de {MAC_REF}."},
                {"title": "Consolidação", "detail": "Parsing determinístico, reconciliação por run_uid e comparação por dataset/variante."},
            ],
        },
        "methods": [
            {"language": "python", "code": "analysis/consolidacao_resultados/consolidate_results.py"},
        ],
    }


def build_app_snapshot(
    runs: pd.DataFrame,
    campaigns: pd.DataFrame,
    activation_pairs: pd.DataFrame,
    batch_pairs: pd.DataFrame,
    evidence: pd.DataFrame,
    classes: pd.DataFrame,
    confusion: pd.DataFrame,
    epochs: pd.DataFrame,
    reported_batch: pd.DataFrame,
    quant_status: pd.DataFrame,
) -> dict[str, Any]:
    comparable_uids = set(batch_pairs.get("windows_run_uid", [])) | set(batch_pairs.get("mac_run_uid", []))
    class_pair_rows = classes[classes["run_uid"].isin(comparable_uids)].to_dict("records")
    confusion_pair_rows = confusion[confusion["run_uid"].isin(comparable_uids)].to_dict("records")
    epoch_pair_rows = epochs[epochs["run_uid"].isin(comparable_uids)].to_dict("records")
    metric_defs = [
        {
            "label": "Macro F1",
            "definition": "Média não ponderada do F1 das classes no conjunto de teste.",
            "componentIds": ["run-results", "activation-f1", "batch-f1", "executive-summary"],
            "sourceLineage": [{"files": ["analysis/consolidacao_resultados/data/run_results.csv"]}],
        },
        {
            "label": "Acurácia",
            "definition": "Proporção de amostras do conjunto de teste classificadas corretamente.",
            "componentIds": ["run-results", "activation-f1", "batch-f1"],
            "sourceLineage": [{"files": ["analysis/consolidacao_resultados/data/run_results.csv"]}],
        },
        {
            "label": "Delta Windows menos Mac",
            "definition": "Valor Windows menos valor Mac; para accuracy e macro F1, expresso em pontos percentuais.",
            "componentIds": ["activation-delta", "batch-f1"],
            "sourceLineage": [{"files": ["analysis/consolidacao_resultados/data/comparisons_windows_mac.csv"]}],
        },
        {
            "label": "Tempo médio por época",
            "definition": "Média registrada de segundos por época dentro de cada run.",
            "componentIds": ["epoch-time", "batch-time"],
            "sourceLineage": [{"files": ["analysis/consolidacao_resultados/data/run_results.csv"]}],
        },
    ]
    return {
        "surface": "report",
        "title": "Resultados de treinamento no Windows e no Mac",
        "generatedAt": SNAPSHOT_AT,
        "status": "reviewed",
        "buildStatus": "creating",
        "queries": {
            "campaign_summary": reviewed_query(
                campaigns.to_dict("records"),
                "Resumo reconciliado por plataforma, campanha e fase",
                ["analysis/consolidacao_resultados/data/campaign_summary.csv"],
                metric_defs,
            ),
            "run_results": reviewed_query(
                runs.to_dict("records"),
                "Resultados finais por execução",
                ["analysis/consolidacao_resultados/data/run_results.csv"],
                metric_defs,
            ),
            "activation_comparison": reviewed_query(
                activation_pairs.to_dict("records"),
                "Pares descritivos de ativação Windows e Mac",
                ["analysis/consolidacao_resultados/data/comparisons_activation_windows_mac.csv"],
                metric_defs,
            ),
            "batch_comparison": reviewed_query(
                batch_pairs.to_dict("records"),
                "Pares de batch com artefatos brutos nos dois ambientes",
                ["analysis/consolidacao_resultados/data/comparisons_batch_windows_mac.csv"],
                metric_defs,
            ),
            "evidence_coverage": reviewed_query(
                evidence.to_dict("records"),
                "Cobertura de evidências por execução",
                ["analysis/consolidacao_resultados/data/evidence_coverage.csv"],
                [],
            ),
            "class_pair_metrics": reviewed_query(
                class_pair_rows,
                "Métricas por classe dos pares exatos de batch",
                ["analysis/consolidacao_resultados/data/class_metrics.csv"],
                metric_defs,
            ),
            "confusion_pair_cells": reviewed_query(
                confusion_pair_rows,
                "Células das matrizes de confusão dos pares exatos de batch",
                ["analysis/consolidacao_resultados/data/confusion_matrices_long.csv"],
                [],
            ),
            "epoch_pair_metrics": reviewed_query(
                epoch_pair_rows,
                "Histórico por época dos pares exatos de batch",
                ["analysis/consolidacao_resultados/data/epoch_metrics.csv"],
                metric_defs,
            ),
            "reported_mac_batch": reviewed_query(
                reported_batch.to_dict("records"),
                "Células de batch reportadas no Mac em 14 de setembro",
                ["analysis/consolidacao_resultados/data/reported_mac_batch_cells.csv"],
                [],
            ),
            "reported_mac_quantization": reviewed_query(
                quant_status.to_dict("records"),
                "Status reportado da quantização no Mac em 14 de setembro",
                ["analysis/consolidacao_resultados/data/reported_mac_quantization_status.csv"],
                [],
            ),
        },
    }


def write_csv(df: pd.DataFrame, filename: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / filename, index=False, encoding="utf-8-sig", float_format="%.12g")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    head_commit = str(git(["rev-parse", "HEAD"])).strip()
    mac_commit = str(git(["rev-parse", MAC_REF])).strip()
    branch = str(git(["branch", "--show-current"])).strip()

    windows_sources, windows_inventory = initial_windows_inventory(head_commit)
    mac_sources, mac_inventory = mac_raw_sources(mac_commit)

    run_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    telemetry_long_rows: list[dict[str, Any]] = []
    telemetry_profile_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    for index, source in enumerate([*windows_sources, *mac_sources], start=1):
        print(f"[{index}/{len(windows_sources) + len(mac_sources)}] {source.platform} {source.run_root}")
        run, classes, confusion, epochs, telemetry_long, telemetry_profile = extract_raw_run(source)
        run_rows.append(run)
        class_rows.extend(classes)
        confusion_rows.extend(confusion)
        epoch_rows.extend(epochs)
        telemetry_long_rows.extend(telemetry_long)
        telemetry_profile_rows.extend(telemetry_profile)
        evidence_rows.append(evidence_row_from_run(run))

    mac_activation_runs, mac_activation_classes, mac_activation_telemetry, activation_inventory, activation_evidence = extract_mac_activation_final(mac_commit)
    run_rows.extend(mac_activation_runs)
    class_rows.extend(mac_activation_classes)
    telemetry_long_rows.extend(mac_activation_telemetry)
    evidence_rows.extend(activation_evidence)

    runs = pd.DataFrame(run_rows)
    classes = pd.DataFrame(class_rows)
    confusion = pd.DataFrame(confusion_rows)
    epochs = pd.DataFrame(epoch_rows)
    telemetry_long = pd.DataFrame(telemetry_long_rows)
    telemetry_profiles = pd.DataFrame(telemetry_profile_rows)
    evidence = pd.DataFrame(evidence_rows)
    inventory = pd.DataFrame([*windows_inventory, *mac_inventory, *activation_inventory])

    windows = runs[runs["platform"] == "Windows"].copy()
    mac = runs[runs["platform"] == "Mac"].copy()
    activation_pairs = pair_rows(windows, mac, "activation")
    batch_pairs = pair_rows(windows, mac, "batch")
    comparisons = pd.concat([batch_pairs, activation_pairs], ignore_index=True, sort=False)
    campaigns = campaign_summary(runs)

    raw_mac_batch_keys = {
        (str(row["dataset_key"]), int(row["batch_size"]))
        for _, row in mac[(mac["campaign"] == MAC_RAW_CAMPAIGN) & (mac["phase"] == "batch")].dropna(subset=["batch_size"]).iterrows()
    }
    reported_batch = reported_mac_batch_cells(raw_mac_batch_keys)
    quant_status = reported_mac_quantization_status()
    checks = validate_data(runs, classes, confusion, epochs, activation_pairs, batch_pairs)

    paired_uids = set(comparisons.get("windows_run_uid", [])) | set(comparisons.get("mac_run_uid", []))
    unmatched = runs[~runs["run_uid"].isin(paired_uids)].copy()
    unmatched["reason"] = np.where(
        unmatched["phase"].isin(["activation", "batch"]),
        "Condição correspondente não possui métrica final verificável no outro ambiente.",
        "Campanha sem protocolo equivalente disponível no outro ambiente.",
    )

    run_columns_first = [
        "run_uid", "platform", "host_runtime", "machine_label", "campaign", "phase", "dataset_key", "dataset", "variant",
        "batch_size", "activation", "quantization_variant", "normalization", "balance_mode", "seed", "attempt", "status",
        "evidence_level", "accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "loss", "test_samples",
        "epochs_completed", "training_seconds", "training_hours", "evaluation_seconds", "wall_seconds", "mean_epoch_seconds",
        "mean_train_examples_per_second",
    ]
    remaining = [column for column in runs.columns if column not in run_columns_first]
    runs = runs[[*run_columns_first, *remaining]].sort_values(["platform", "campaign", "phase", "dataset_key", "variant"]).reset_index(drop=True)

    write_csv(runs, "run_results.csv")
    write_csv(inventory.sort_values(["platform", "campaign", "source_path"]), "status_inventory.csv")
    write_csv(classes.sort_values(["platform", "campaign", "dataset_key", "variant", "class_index"]), "class_metrics.csv")
    write_csv(confusion.sort_values(["platform", "campaign", "dataset_key", "variant", "true_index", "pred_index"]), "confusion_matrices_long.csv")
    write_csv(epochs.sort_values(["platform", "campaign", "dataset_key", "variant", "epoch"]), "epoch_metrics.csv")
    write_csv(telemetry_long.sort_values(["platform", "campaign", "dataset_key", "variant", "metric", "stat"]), "telemetry_metrics_long.csv")
    write_csv(telemetry_profiles.sort_values(["platform", "campaign", "dataset_key", "variant", "time_bin"]), "telemetry_profiles.csv")
    write_csv(evidence.sort_values(["platform", "campaign", "dataset", "variant"]), "evidence_coverage.csv")
    write_csv(campaigns.sort_values(["platform", "campaign", "phase"]), "campaign_summary.csv")
    write_csv(activation_pairs.sort_values(["dataset_key", "activation"]), "comparisons_activation_windows_mac.csv")
    write_csv(batch_pairs.sort_values(["dataset_key", "batch_size_windows"]), "comparisons_batch_windows_mac.csv")
    write_csv(comparisons.sort_values(["pair_type", "dataset_key", "variant"]), "comparisons_windows_mac.csv")
    write_csv(unmatched.sort_values(["platform", "campaign", "dataset_key", "variant"]), "unmatched_runs.csv")
    write_csv(reported_batch.sort_values(["dataset_key", "batch_size"]), "reported_mac_batch_cells.csv")
    write_csv(quant_status.sort_values(["dataset_key", "variant"]), "reported_mac_quantization_status.csv")
    write_csv(checks, "validation_checks.csv")

    platform_summary: dict[str, Any] = {}
    for platform, group in runs.groupby("platform"):
        platform_summary[platform] = {
            "completed_runs_with_metrics": int(len(group)),
            "campaigns": int(group["campaign"].nunique()),
            "datasets": int(group["dataset_key"].nunique()),
            "epochs": int(pd.to_numeric(group["epochs_completed"], errors="coerce").fillna(0).sum()),
            "training_hours_observed": float(pd.to_numeric(group["training_seconds"], errors="coerce").fillna(0).sum() / 3600),
            "runs_with_class_metrics": int(group["has_class_metrics"].fillna(False).sum()),
            "runs_with_confusion_matrix": int(group["has_confusion_matrix"].fillna(False).sum()),
            "runs_with_epoch_metrics": int(group["has_epoch_metrics"].fillna(False).sum()),
            "runs_with_telemetry_summary": int(group["has_telemetry_summary"].fillna(False).sum()),
        }

    def pair_summary(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"pairs": 0}
        delta_f1 = pd.to_numeric(df["delta_macro_f1_pp_windows_minus_mac"], errors="coerce")
        delta_acc = pd.to_numeric(df["delta_accuracy_pp_windows_minus_mac"], errors="coerce")
        time_ratio = pd.to_numeric(df["training_time_ratio_windows_over_mac"], errors="coerce")
        return {
            "pairs": int(len(df)),
            "mean_delta_macro_f1_pp_windows_minus_mac": clean_scalar(delta_f1.mean()),
            "median_delta_macro_f1_pp_windows_minus_mac": clean_scalar(delta_f1.median()),
            "windows_macro_f1_wins": int((delta_f1 > 0).sum()),
            "mac_macro_f1_wins": int((delta_f1 < 0).sum()),
            "mean_delta_accuracy_pp_windows_minus_mac": clean_scalar(delta_acc.mean()),
            "median_training_time_ratio_windows_over_mac": clean_scalar(time_ratio.median()),
            "windows_faster_runs": int((time_ratio < 1).sum()),
            "mac_faster_runs": int((time_ratio > 1).sum()),
        }

    summary = {
        "snapshot_at": SNAPSHOT_AT,
        "branch": branch,
        "head_commit": head_commit,
        "mac_ref": MAC_REF,
        "mac_commit": mac_commit,
        "platforms": platform_summary,
        "comparisons": {
            "batch_exact": pair_summary(batch_pairs),
            "activation_descriptive": pair_summary(activation_pairs),
        },
        "reported_mac_gaps": {
            "batch_completed_reported": int((reported_batch["status"] == "completed").sum()),
            "batch_completed_with_raw_versioned": int(reported_batch["raw_artifact_versioned"].sum()),
            "batch_completed_report_only": int((~reported_batch["raw_artifact_versioned"]).sum()),
            "quantization_completed_reported": int((quant_status["status"] == "completed").sum()),
            "quantization_partial_stale": int((quant_status["status"] == "partial_stale").sum()),
            "quantization_not_started": int((quant_status["status"] == "not_started").sum()),
        },
        "evidence": {
            "class_metric_rows": int(len(classes)),
            "confusion_matrix_cells": int(len(confusion)),
            "epoch_rows": int(len(epochs)),
            "telemetry_metric_rows": int(len(telemetry_long)),
            "telemetry_profile_rows": int(len(telemetry_profiles)),
        },
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    capture_manifest = {
        "snapshot_at": SNAPSHOT_AT,
        "branch": branch,
        "head_commit": head_commit,
        "mac_ref": MAC_REF,
        "mac_commit": mac_commit,
        "windows_status_files_seen": len(windows_inventory),
        "windows_completed_runs_captured": len(windows_sources),
        "mac_raw_status_files_seen": len(mac_inventory),
        "mac_raw_completed_runs_captured": len(mac_sources),
        "mac_activation_consolidated_rows": len(mac_activation_runs),
        "notes": [
            "O snapshot usa o primeiro estado lido de cada status.json; runs em execução não são promovidos a resultado final durante a captura.",
            "Os outputs locais ignorados pelo Git foram lidos em modo somente leitura.",
            "Resultados Mac ausentes como artefato bruto permanecem classificados como evidência de relatório.",
        ],
    }
    (DATA_DIR / "capture_manifest.json").write_text(json.dumps(capture_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    snapshot = build_app_snapshot(runs, campaigns, activation_pairs, batch_pairs, evidence, classes, confusion, epochs, reported_batch, quant_status)
    (DATA_DIR / "report_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    hashes = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            hashes.append(f"{file_sha256(path)}  {path.name}")
    (DATA_DIR / "checksums.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
