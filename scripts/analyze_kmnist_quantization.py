"""Create a reproducible technical analysis for a completed KMNIST quantization run.

The script is intentionally independent from TensorFlow: it consumes the durable
JSON/CSV/NumPy artifacts produced by ``run_kmnist_quantization_benchmark.py``.
It writes audit tables, charts, a Markdown report and an executed notebook.

Example (WSL):

    /home/msduda/.venvs/tcc-benchmark/bin/python \
      scripts/analyze_kmnist_quantization.py \
      --input-root /mnt/e/tcc-benchmark/outputs/kmnist-quantization-2026-08-09
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from nbclient import NotebookClient


RUN_ID = "kmnist__unit_interval__all_raw__seed-42"
VARIANTS = ("fp32", "fp16", "int8_qat", "int4_qat", "int8_ptq")
TRAINING_VARIANTS = ("fp32", "fp16", "int8_qat", "int4_qat")
DISPLAY_NAMES = {
    "fp32": "FP32",
    "fp16": "FP16",
    "int8_qat": "INT8-QAT",
    "int4_qat": "INT4-QAT (emulado)",
    "int8_ptq": "INT8-PTQ",
}
COLORS = {
    "fp32": "#4C78A8",
    "fp16": "#72B7B2",
    "int8_qat": "#F58518",
    "int4_qat": "#E45756",
    "int8_ptq": "#54A24B",
}
GIB = 1024**3
MIB = 1024**2


def read_json(path: Path, default: Any | None = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def fmt_number(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def fmt_pct(value: Any, digits: int = 3) -> str:
    return fmt_number(None if value is None else 100 * float(value), digits, "%")


def fmt_bytes(value: Any) -> str:
    value = numeric(value)
    if value is None:
        return "—"
    if value >= GIB:
        return f"{value / GIB:.2f} GiB"
    return f"{value / MIB:.2f} MiB"


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame.loc[:, columns]
    if selected.empty:
        return "_Sem dados._"
    # Avoid an implicit dependency on the optional ``tabulate`` package so the
    # report remains runnable in the benchmark virtual environment.
    def cell(value: Any) -> str:
        if pd.isna(value):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    header = "| " + " | ".join(cell(column) for column in selected.columns) + " |"
    divider = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in selected.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def run_root(root: Path, variant: str) -> Path:
    return root / variant / "kmnist" / "runs" / RUN_ID


def artifact(root: Path, variant: str, name: str) -> Path:
    return run_root(root, variant) / "artifacts" / name


def extract_epoch_history(path: Path, variant: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame.insert(0, "variant", variant)
    if "epoch" not in frame:
        frame.insert(1, "epoch", np.arange(1, len(frame) + 1))
    else:
        epoch = pd.to_numeric(frame["epoch"], errors="coerce")
        # Keras CSV writers sometimes number epochs from zero.
        if epoch.min() == 0:
            epoch = epoch + 1
        frame["epoch"] = epoch.astype("Int64")
    return frame


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    return numeric((metrics.get("classification") or {}).get(key))


def telemetry_value(summary: dict[str, Any], key: str, statistic: str) -> float | None:
    return numeric((((summary.get("metrics") or {}).get(key) or {}).get(statistic)))


def litert_conversion(size: dict[str, Any], postprocess: dict[str, Any]) -> dict[str, Any]:
    conversion = size.get("litert") if isinstance(size.get("litert"), dict) else {}
    if not conversion and isinstance(postprocess.get("conversion"), dict):
        conversion = postprocess["conversion"]
    return conversion


def collect_variant(root: Path, variant: str, queue: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    base = run_root(root, variant)
    manifest = read_json(base / "manifest.json", {}) or {}
    status = read_json(base / "status.json", {}) or {}
    metrics = read_json(base / "artifacts" / "test_metrics.json", {}) or {}
    training = read_json(base / "logs" / "training_summary.json", {}) or {}
    telemetry = read_json(base / "telemetry" / "summary.json", {}) or {}
    environment = read_json(base / "telemetry" / "environment.json", {}) or {}
    size = read_json(base / "artifacts" / "model_size.json", {}) or {}
    benchmark = read_json(base / "artifacts" / "litert_benchmark.json", {}) or {}
    postprocess = read_json(base / "artifacts" / "quantization_postprocess.json", {}) or {}
    history = extract_epoch_history(base / "checkpoints" / "epoch_metrics.csv", variant)
    if history.empty:
        history = extract_epoch_history(base / "logs" / "epoch_metrics.csv", variant)

    classification = metrics.get("classification") or {}
    config = manifest.get("config") or {}
    protocol = config.get("protocol") or {}
    data = manifest.get("data_metadata") or {}
    runtime = manifest.get("runtime") or {}
    queue_record = (queue.get("variants") or {}).get(variant) or {}
    conversion = litert_conversion(size, postprocess)
    lite_classification = benchmark.get("classification") or {}
    records = {
        "variant": variant,
        "label": DISPLAY_NAMES[variant],
        "run_status": status.get("status", "missing"),
        "queue_status": queue_record.get("status_after_training", queue_record.get("status", "completed" if variant == "int8_ptq" else "missing")),
        "epochs_completed": numeric(training.get("epochs_completed")),
        "expected_epochs": 50 if variant in TRAINING_VARIANTS else 0,
        "accuracy": numeric(classification.get("accuracy")),
        "macro_precision": numeric(classification.get("macro_precision")),
        "macro_recall": numeric(classification.get("macro_recall")),
        "macro_f1": numeric(classification.get("macro_f1")),
        "macro_ovr_auc": numeric(classification.get("macro_ovr_auc")),
        "test_loss": numeric((metrics.get("keras_metrics") or {}).get("loss")),
        "metric_source": metrics.get("metric_source", "keras_checkpoint"),
        "training_seconds": numeric(training.get("training_seconds")),
        "fit_seconds_current_attempt": numeric(training.get("fit_seconds_current_attempt")),
        "evaluation_seconds": numeric(training.get("evaluation_seconds")),
        "wall_seconds_current_attempt": numeric(training.get("wall_seconds_current_attempt")),
        "queue_stage_seconds": numeric(queue_record.get("duration_seconds")),
        "mean_epoch_seconds": numeric(training.get("mean_epoch_seconds")),
        "train_examples_per_second": numeric(training.get("mean_train_examples_per_second")),
        "gpu_util_mean_percent": telemetry_value(telemetry, "gpu_utilization_percent", "mean"),
        "gpu_util_p95_percent": telemetry_value(telemetry, "gpu_utilization_percent", "p95"),
        "gpu_memory_peak_gib": (telemetry_value(telemetry, "gpu_memory_used_bytes", "max") or 0) / GIB if telemetry else None,
        "gpu_memory_mean_gib": (telemetry_value(telemetry, "gpu_memory_used_bytes", "mean") or 0) / GIB if telemetry else None,
        "process_rss_peak_gib": (telemetry_value(telemetry, "process_rss_bytes", "max") or 0) / GIB if telemetry else None,
        "process_rss_mean_gib": (telemetry_value(telemetry, "process_rss_bytes", "mean") or 0) / GIB if telemetry else None,
        "system_ram_peak_gib": (telemetry_value(telemetry, "ram_used_bytes", "max") or 0) / GIB if telemetry else None,
        "gpu_temperature_max_c": telemetry_value(telemetry, "gpu_temperature_c", "max"),
        "telemetry_samples": numeric(telemetry.get("sample_count")),
        "tf_allocator_peak_mib": max(
            (numeric(values.get("peak")) or 0) / MIB for values in (training.get("tensorflow_gpu_memory_after") or {}).values()
        ) if training.get("tensorflow_gpu_memory_after") else None,
        "serialized_model_mib": (numeric(size.get("serialized_model_bytes")) or 0) / MIB if size.get("serialized_model_bytes") else None,
        "serialized_litert_mib": (numeric(size.get("serialized_litert_bytes")) or 0) / MIB if size.get("serialized_litert_bytes") else None,
        "estimated_deployment_mib": (numeric(size.get("estimated_deployment_parameter_bytes")) or 0) / MIB if size.get("estimated_deployment_parameter_bytes") else None,
        "estimated_weight_bits": numeric(size.get("estimated_weight_bits")),
        "efficiency_interpretation": size.get("efficiency_interpretation", "not_available"),
        "litert_status": conversion.get("status", "not_exported"),
        "litert_validity": conversion.get("quantization_validity", size.get("quantization_validity", "not_exported")),
        "litert_input_dtypes": ", ".join(conversion.get("input_dtypes") or []),
        "litert_output_dtypes": ", ".join(conversion.get("output_dtypes") or []),
        "litert_float32_tensors": numeric((conversion.get("tensor_dtypes") or {}).get("float32")),
        "inference_runtime": benchmark.get("runtime"),
        "inference_latency_ms": numeric(benchmark.get("median_batch_latency_ms")),
        "inference_throughput_eps": numeric(benchmark.get("throughput_examples_per_second")),
        "inference_rss_peak_mib": (numeric(benchmark.get("process_rss_peak_bytes")) or 0) / MIB if benchmark.get("process_rss_peak_bytes") else None,
        "inference_vram": benchmark.get("inference_vram"),
        "conversion_error": str(postprocess.get("conversion_error"))[:512] if postprocess.get("conversion_error") else None,
        "split_fingerprint": manifest.get("split_fingerprint"),
        "source_fingerprint": config.get("source_fingerprint"),
        "dtype_policy": protocol.get("dtype_policy", (config.get("training") or {}).get("dtype_policy")),
        "qat_weight_bits": protocol.get("qat_weight_bits", (config.get("training") or {}).get("qat_weight_bits")),
        "batch_size": numeric((config.get("training") or {}).get("batch_size")),
        "augmentation_disabled": (config.get("training") or {}).get("augmentation") == {
            "flip_lr": False, "brightness_delta": 0.0, "contrast_lower": 1.0, "contrast_upper": 1.0,
            "translate_frac": 0.0, "zoom_min": 1.0, "zoom_max": 1.0, "noise_std": 0.0,
            "cutout_prob": 0.0, "cutout_max_frac": 0.0,
        },
        "memory_growth": all(item.get("enabled") for item in (runtime.get("memory_growth") or [])),
        "train_samples": numeric((((data.get("datasets") or {}).get("train") or {}).get("total_examples"))),
        "validation_samples": numeric((((data.get("datasets") or {}).get("validation") or {}).get("total_examples"))),
        "test_samples": numeric((((data.get("datasets") or {}).get("test") or {}).get("total_examples"))),
        "telemetry_epoch_end_events": numeric(((telemetry.get("event_counts") or {}).get("epoch_end"))),
        "gpu_name": ((environment.get("gpus") or [{}])[0]).get("name") if environment.get("gpus") else None,
    }
    raw = {
        "manifest": manifest, "status": status, "metrics": metrics, "training": training,
        "telemetry": telemetry, "environment": environment, "size": size, "benchmark": benchmark,
        "postprocess": postprocess, "classification": classification,
    }
    return records, history, raw


def add_logit_comparison(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    fp32_path = artifact(root, "fp32", "logits.npy")
    if not fp32_path.is_file():
        return summary
    reference = np.load(fp32_path)
    mse: dict[str, float | None] = {"fp32": 0.0}
    agreement: dict[str, float | None] = {"fp32": 1.0}
    reference_labels = np.argmax(reference, axis=1)
    for variant in VARIANTS:
        if variant == "fp32":
            continue
        candidate = artifact(root, variant, "logits.npy")
        if not candidate.is_file():
            candidate = artifact(root, variant, "litert_logits.npy")
        if not candidate.is_file():
            mse[variant], agreement[variant] = None, None
            continue
        values = np.load(candidate)
        if values.shape != reference.shape:
            mse[variant], agreement[variant] = None, None
            continue
        mse[variant] = float(np.mean((reference.astype(np.float64) - values.astype(np.float64)) ** 2))
        agreement[variant] = float(np.mean(reference_labels == np.argmax(values, axis=1)))
    summary["logit_mse_vs_fp32"] = summary["variant"].map(mse)
    summary["prediction_agreement_vs_fp32"] = summary["variant"].map(agreement)
    return summary


def validate_protocol(summary: pd.DataFrame, histories: pd.DataFrame, queue: dict[str, Any]) -> dict[str, Any]:
    expected = set(VARIANTS)
    present = set(summary["variant"])
    training = summary[summary["variant"].isin(TRAINING_VARIANTS)].copy()
    epoch_counts = histories.groupby("variant")["epoch"].nunique().to_dict() if not histories.empty else {}
    checks = {
        "top_level_status_completed": queue.get("status") == "completed",
        "zero_reported_failures": queue.get("failures") == 0,
        "all_five_variants_present": present == expected,
        "all_variant_statuses_completed": bool((summary["run_status"] == "completed").all()),
        "all_training_runs_have_50_epochs": all(epoch_counts.get(v, 0) == 50 for v in TRAINING_VARIANTS),
        "all_training_summaries_report_50_epochs": bool((training["epochs_completed"] == 50).all()),
        "fixed_batch_256": bool((training["batch_size"] == 256).all()),
        "augmentation_disabled": bool(training["augmentation_disabled"].all()),
        "memory_growth_enabled": bool(training["memory_growth"].all()),
        "same_split_fingerprint": training["split_fingerprint"].nunique(dropna=True) == 1,
        "same_source_fingerprint": training["source_fingerprint"].nunique(dropna=True) == 1,
        "same_70_15_15_sizes": bool(((training["train_samples"] == 49000) & (training["validation_samples"] == 10500) & (training["test_samples"] == 10500)).all()),
    }
    warnings: list[str] = []
    telemetry_events = training.set_index("variant")["telemetry_epoch_end_events"].dropna().to_dict()
    short_events = {key: int(value) for key, value in telemetry_events.items() if value != 50}
    if short_events:
        warnings.append(
            "A telemetria registrou menos de 50 eventos `epoch_end` em "
            + ", ".join(f"{DISPLAY_NAMES[k]} ({v})" for k, v in short_events.items())
            + ". Os CSVs de época e os `training_summary.json` registram as 50 épocas; por isso este é um aviso de instrumentação, não uma evidência de treino incompleto."
        )
    fp16 = summary.loc[summary["variant"] == "fp16"].iloc[0]
    if fp16["litert_status"] != "completed":
        warnings.append("A conversão LiteRT FP16 falhou; não há métrica de inferência LiteRT/CPU nem tamanho LiteRT para FP16.")
    int4 = summary.loc[summary["variant"] == "int4_qat"].iloc[0]
    if int4["efficiency_interpretation"] == "emulated_int4":
        warnings.append("INT4-QAT é fake quantização experimental/emulada; o tamanho de 4 bits é estimado e não representa um artefato LiteRT INT4 implantável.")
    ptq = summary.loc[summary["variant"] == "int8_ptq"].iloc[0]
    if ptq["litert_validity"] != "integer_only":
        warnings.append("O LiteRT INT8-PTQ contém tensores float32 auxiliares; ele é classificado como grafo híbrido/float_or_hybrid, não como INT8 integral.")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed": all(checks.values()),
        "epoch_row_counts": {key: int(value) for key, value in epoch_counts.items()},
        "warnings": warnings,
    }


def create_charts(summary: pd.DataFrame, histories: pd.DataFrame, output: Path) -> list[Path]:
    charts: list[Path] = []
    ordered = summary.set_index("variant").loc[list(VARIANTS)].reset_index()
    labels = ordered["label"].tolist()
    colors = [COLORS[v] for v in ordered["variant"]]
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for axis, metric, title in zip(axes, ("accuracy", "macro_f1", "macro_ovr_auc"), ("Acurácia", "Macro-F1", "AUC macro OvR")):
        values = ordered[metric] * 100
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylim(min(94, values.min() - 1), 100.1)
        axis.tick_params(axis="x", rotation=35)
        for x, value in enumerate(values):
            axis.text(x, value + 0.08, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    path = output / "01_test_metrics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    charts.append(path)

    if not histories.empty:
        metric = "val_macro_f1" if "val_macro_f1" in histories else "val_accuracy"
        fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for variant in TRAINING_VARIANTS:
            part = histories[histories["variant"] == variant]
            if not part.empty and metric in part:
                axis.plot(part["epoch"], part[metric] * 100, label=DISPLAY_NAMES[variant], color=COLORS[variant], linewidth=1.8)
        axis.set_title("Convergência na validação")
        axis.set_xlabel("Época")
        axis.set_ylabel("Macro-F1 de validação (%)" if metric == "val_macro_f1" else "Acurácia de validação (%)")
        axis.legend(ncol=2)
        path = output / "02_validation_convergence.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        charts.append(path)

    training = ordered[ordered["variant"].isin(TRAINING_VARIANTS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    axes[0].bar(training["label"], training["training_seconds"] / 60, color=[COLORS[v] for v in training["variant"]])
    axes[0].set_title("Tempo de treino (soma das épocas)")
    axes[0].set_ylabel("minutos")
    axes[0].tick_params(axis="x", rotation=30)
    for x, value in enumerate(training["training_seconds"] / 60):
        axes[0].text(x, value + 0.5, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(training["label"], training["gpu_memory_peak_gib"], color=[COLORS[v] for v in training["variant"]])
    axes[1].set_title("Pico de VRAM global durante treino")
    axes[1].set_ylabel("GiB")
    axes[1].tick_params(axis="x", rotation=30)
    for x, value in enumerate(training["gpu_memory_peak_gib"]):
        axes[1].text(x, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    path = output / "03_training_time_and_vram.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    charts.append(path)

    inference = ordered.dropna(subset=["inference_throughput_eps"]).copy()
    if not inference.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
        axes[0].bar(inference["label"], inference["inference_throughput_eps"], color=[COLORS[v] for v in inference["variant"]])
        axes[0].set_title("Throughput LiteRT em CPU")
        axes[0].set_ylabel("exemplos/s")
        axes[0].tick_params(axis="x", rotation=30)
        axes[1].bar(inference["label"], inference["serialized_litert_mib"], color=[COLORS[v] for v in inference["variant"]])
        axes[1].set_title("Tamanho serializado LiteRT")
        axes[1].set_ylabel("MiB")
        axes[1].tick_params(axis="x", rotation=30)
        path = output / "04_litert_efficiency.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        charts.append(path)
    return charts


def display_frame(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for column in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_ovr_auc", "prediction_agreement_vs_fp32"):
        if column in result:
            result[column] = result[column].map(lambda x: fmt_pct(x, 3))
    for column in ("training_seconds", "queue_stage_seconds", "inference_latency_ms", "inference_throughput_eps", "logit_mse_vs_fp32"):
        if column in result:
            result[column] = result[column].map(lambda x: fmt_number(x, 3))
    return result


def write_markdown(
    output: Path,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    histories: pd.DataFrame,
    quality: dict[str, Any],
    queue: dict[str, Any],
    charts: list[Path],
) -> Path:
    baseline = summary.loc[summary["variant"] == "fp32"].iloc[0]
    best = summary.loc[summary["accuracy"].idxmax()]
    fp32_training = float(baseline["training_seconds"])
    int8_ptq = summary.loc[summary["variant"] == "int8_ptq"].iloc[0]
    int4 = summary.loc[summary["variant"] == "int4_qat"].iloc[0]
    fp16 = summary.loc[summary["variant"] == "fp16"].iloc[0]

    metric_table = summary[["label", "accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_ovr_auc", "test_loss", "logit_mse_vs_fp32", "prediction_agreement_vs_fp32"]].copy()
    metric_table.columns = ["Variante", "Acurácia", "Precisão macro", "Recall macro", "Macro-F1", "AUC macro OvR", "Loss", "MSE logits vs FP32", "Acordo vs FP32"]
    for column in ("Acurácia", "Precisão macro", "Recall macro", "Macro-F1", "AUC macro OvR", "Acordo vs FP32"):
        metric_table[column] = metric_table[column].map(lambda x: fmt_pct(x, 3))
    for column in ("Loss", "MSE logits vs FP32"):
        metric_table[column] = metric_table[column].map(lambda x: fmt_number(x, 6))

    train_table = summary[summary["variant"].isin(TRAINING_VARIANTS)][["label", "epochs_completed", "training_seconds", "fit_seconds_current_attempt", "mean_epoch_seconds", "train_examples_per_second", "gpu_util_mean_percent", "gpu_memory_peak_gib", "process_rss_peak_gib", "gpu_temperature_max_c"]].copy()
    train_table.columns = ["Variante", "Épocas", "Tempo das épocas (s)", "Wall/fit atual (s)", "Média/época (s)", "Treino (ex/s)", "GPU média (%)", "VRAM pico (GiB)", "RSS processo pico (GiB)", "Temp. GPU pico (°C)"]
    for column in train_table.columns[2:]:
        train_table[column] = train_table[column].map(lambda x: fmt_number(x, 2))

    deploy_table = summary[["label", "serialized_model_mib", "serialized_litert_mib", "estimated_deployment_mib", "estimated_weight_bits", "efficiency_interpretation", "litert_status", "litert_validity", "inference_latency_ms", "inference_throughput_eps", "inference_rss_peak_mib"]].copy()
    deploy_table.columns = ["Variante", "Checkpoint (MiB)", "LiteRT (MiB)", "Pesos estimados (MiB)", "Bits dos pesos", "Interpretação", "Conversão", "Validade", "Latência/batch (ms)", "Throughput (ex/s)", "RSS inferência (MiB)"]
    for column in ("Checkpoint (MiB)", "LiteRT (MiB)", "Pesos estimados (MiB)", "Bits dos pesos", "Latência/batch (ms)", "Throughput (ex/s)", "RSS inferência (MiB)"):
        deploy_table[column] = deploy_table[column].map(lambda x: fmt_number(x, 3))

    deltas_table = deltas[["label", "accuracy_delta_pp", "macro_f1_delta_pp", "auc_delta_pp", "logit_mse_vs_fp32", "prediction_agreement_vs_fp32"]].copy()
    deltas_table.columns = ["Variante", "Δ acurácia (p.p.)", "Δ Macro-F1 (p.p.)", "Δ AUC (p.p.)", "MSE logits", "Acordo"]
    for column in ("Δ acurácia (p.p.)", "Δ Macro-F1 (p.p.)", "Δ AUC (p.p.)"):
        deltas_table[column] = deltas_table[column].map(lambda x: fmt_number(x, 3))
    deltas_table["MSE logits"] = deltas_table["MSE logits"].map(lambda x: fmt_number(x, 6))
    deltas_table["Acordo"] = deltas_table["Acordo"].map(lambda x: fmt_pct(x, 3))

    chart_links = "\n\n".join(f"![{chart.stem}]({chart.name})" for chart in charts)
    q_lines = "\n".join(f"- {'✅' if passed else '❌'} `{name}`" for name, passed in quality["checks"].items())
    warning_lines = "\n".join(f"- {warning}" for warning in quality["warnings"])
    int8_speedup = float(int8_ptq["inference_throughput_eps"]) / float(baseline["inference_throughput_eps"])
    int8_size_reduction = 1 - float(int8_ptq["serialized_litert_mib"]) / float(baseline["serialized_litert_mib"])
    best_delta = (float(best["accuracy"]) - float(baseline["accuracy"])) * 100
    int4_time_delta = 1 - float(int4["training_seconds"]) / fp32_training
    fp16_time_delta = 1 - float(fp16["training_seconds"]) / fp32_training
    fp16_time_sentence = (
        f"reduziu o tempo somado das épocas em {fmt_pct(fp16_time_delta, 1)} frente ao FP32"
        if fp16_time_delta >= 0
        else f"teve tempo somado das épocas {fmt_pct(abs(fp16_time_delta), 1)} maior que o FP32"
    )

    content = f"""# Benchmark de quantização KMNIST — análise técnica

Gerado em {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} a partir de `{output.parent}`. Este relatório é descritivo: cada variante foi treinada uma vez com seed 42; diferenças entre variantes não devem ser interpretadas como inferência estatística ou causal isolada.

## Resumo técnico

Os cinco tratamentos foram concluídos (`status` global: **{queue.get('status')}**, falhas: **{queue.get('failures')}**). A melhor acurácia de teste foi **{fmt_pct(best['accuracy'])}** em **{best['label']}**, {fmt_number(best_delta, 3)} p.p. em relação ao FP32 ({fmt_pct(baseline['accuracy'])}). A QAT INT8 preservou qualidade próxima ao FP32 ({fmt_number((float(summary.loc[summary['variant'] == 'int8_qat', 'accuracy'].iloc[0]) - float(baseline['accuracy'])) * 100, 3)} p.p.), enquanto a PTQ INT8 reduziu o LiteRT de {fmt_number(baseline['serialized_litert_mib'], 3)} para {fmt_number(int8_ptq['serialized_litert_mib'], 3)} MiB ({fmt_pct(int8_size_reduction, 1)} menor) e elevou o throughput LiteRT/CPU de {fmt_number(baseline['inference_throughput_eps'], 1)} para {fmt_number(int8_ptq['inference_throughput_eps'], 1)} exemplos/s ({fmt_number(int8_speedup, 2)}×). Essa PTQ tem perda de {fmt_number((float(int8_ptq['accuracy']) - float(baseline['accuracy'])) * 100, 3)} p.p. de acurácia.

INT4-QAT obteve o melhor resultado numérico, mas é uma implementação **emulada** de fake quantização com pesos mestres FP32: não é evidência de implantação INT4 suportada. FP16 {fp16_time_sentence}, porém sua conversão LiteRT falhou e, portanto, não tem resultado de inferência CPU comparável.

## Evidências visuais

{chart_links}

## Escopo, dados e definições

- **Dados:** KMNIST, 70.000 exemplos brutos (`all_raw`), 10 classes, entrada 64×64×1 com normalização `unit_interval`.
- **Partição:** estratificada 70/15/15, seed 42: 49.000 treino, 10.500 validação e 10.500 teste; 1.050 exemplos por classe em validação e teste.
- **Treino:** 50 épocas, batch size 256, CNN estruturalmente idêntica, sem augmentation, `TF_FORCE_GPU_ALLOW_GROWTH=true` e `memory_growth` ativado antes do TensorFlow.
- **Variantes:** FP32, FP16 (`mixed_float16`), W8A8 QAT, W4A8 QAT emulada e INT8-PTQ do checkpoint FP32 com 1.024 exemplos de calibração.
- **Métricas de teste:** acurácia; precision, recall e F1 macro; AUC ROC macro one-vs-rest. MSE de logits é a média do erro quadrático entre logits da variante e logits FP32 no mesmo conjunto de 10.500 exemplos. Para QAT treinada do zero, esse MSE mede divergência final completa, não somente o efeito de quantização.
- **Desempenho de implantação:** LiteRT Python em CPU, batch 256, 10 warm-ups e 30 passagens cronometradas. VRAM não se aplica a essa trilha CPU.

## Qualidade e integridade dos dados

{q_lines}

Contagem de linhas de épocas por variante: `{json.dumps(quality['epoch_row_counts'], ensure_ascii=False)}`.

{warning_lines}

## Qualidade preditiva no teste

{markdown_table(metric_table)}

FP32 é a referência. INT8-QAT ficou muito próximo da referência em acurácia e Macro-F1. INT8-PTQ perdeu mais qualidade e concentrou a maior degradação de recall na classe 4 (consultar `classification_report.json` por variante para o detalhamento por classe). INT4-QAT obteve valores superiores nesta única execução, o que é compatível tanto com variação do processo de treino quanto com a regularização induzida pelo fake quant; sem repetições independentes, não se pode atribuir a melhoria ao uso de 4 bits.

### Deltas contra FP32

{markdown_table(deltas_table)}

## Tempo de treinamento e uso de hardware

{markdown_table(train_table)}

`Tempo das épocas` soma as durações por época persistidas no histórico, robusta a retomadas. `Wall/fit atual` é o tempo de `model.fit` da tentativa atual e inclui sobrecargas não atribuídas às épocas. As métricas de VRAM são globais pela NVML (incluem o processo e contexto da GPU); o pico do alocador TensorFlow deve ser lido separadamente porque mede somente o allocator TensorFlow. A telemetria amostra a cada cinco segundos, então picos muito curtos podem não ter sido observados.

INT8-QAT e INT4-QAT usaram aproximadamente {fmt_pct(int4_time_delta, 1)} menos tempo de épocas que FP32 nesta máquina. Como se trata de execuções únicas e a instrumentação inclui E/S e inicialização, isso é uma observação operacional, não uma garantia de aceleração generalizável.

## Tamanho e implantação LiteRT/CPU

{markdown_table(deploy_table)}

A variante INT8-PTQ possui entrada/saída int8 e tamanho físico LiteRT de {fmt_number(int8_ptq['serialized_litert_mib'], 3)} MiB, mas os metadados mostram 40 tensores float32 auxiliares: foi corretamente marcada como `float_or_hybrid`, não como grafo INT8 integral. A variante INT8-QAT também deve ser interpretada pela validade de conversão gravada nos metadados, não somente pelo rótulo da variante. INT4-QAT não tem exportação LiteRT e o valor de quatro bits é uma estimativa dos pesos quantizáveis. FP16 tem checkpoint treinado, porém falhou ao converter para LiteRT; por isso latência, throughput e RSS de inferência estão ausentes, em vez de serem imputados.

## Metodologia e rastreabilidade

Os dados deste relatório vêm de `manifest.json`, `status.json`, `training_summary.json`, `epoch_metrics.csv`, `test_metrics.json`, `model_size.json`, `litert_benchmark.json`, `quantization_postprocess.json` e `telemetry/summary.json` de cada run. As tabelas CSV em `analysis/` são a camada tabular auditável. O notebook entregue relê essas tabelas, reproduz os gráficos e registra sua própria execução.

## Limitações, incerteza e checagens de robustez

- Há uma única seed por variante; não foram calculados intervalos de confiança, testes de hipótese ou significância.
- QAT foi treinada do zero, portanto diferenças para FP32 combinam quantização, ordem estocástica de treino e possíveis diferenças numéricas do backend.
- CPU LiteRT e GPU TensorFlow são trilhas de hardware distintas; não compare latência LiteRT/CPU com throughput de treino GPU como se fosse a mesma carga.
- FP16 não é uma medida de implantação neste experimento porque o conversor não gerou FlatBuffer válido.
- INT4-QAT é experimental/emulado; tamanho serializado de checkpoint não é tamanho de um binário INT4 implantável.
- A PTQ INT8 é híbrida em termos de tensores e não deve ser reportada como INT8 integral.

## Próximos passos recomendados

1. Repetir cada tratamento com ao menos 3–5 seeds e reportar média, desvio-padrão e intervalos de confiança dos deltas contra FP32.
2. Corrigir ou isolar a rota de exportação FP16 antes de compará-la no runtime de implantação.
3. Caso o alvo seja hardware INT8 estrito, inspecionar e eliminar os tensores float remanescentes da PTQ ou declarar explicitamente o artefato como híbrido.
4. Para INT4, migrar para um backend que aceite pesos INT4 reais e medir o binário/latência nesse backend; não extrapolar do fake quant atual.

## Questões em aberto

- A vantagem numérica observada no INT4-QAT se repete sob seeds independentes?
- Qual é o custo de qualidade da PTQ ao aumentar a calibração de 1.024 exemplos para um subconjunto maior?
- Em qual dispositivo-alvo (por exemplo, ARM ou NPU) a redução de tamanho INT8 se converte de fato em menor latência e energia?
"""
    path = output / "kmnist_quantization_report.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_notebook(output: Path) -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()},
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Benchmark de quantização KMNIST — notebook reprodutível\n\n"
            "Este notebook lê as tabelas auditáveis geradas a partir dos artefatos do benchmark, confirma as verificações de integridade e recria os gráficos principais."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
            "ANALYSIS = Path.cwd()\n"
            "if not (ANALYSIS / 'run_summary.csv').exists():\n"
            "    ANALYSIS = Path('/mnt/e/tcc-benchmark/outputs/kmnist-quantization-2026-08-09/analysis')\n"
            "assert (ANALYSIS / 'run_summary.csv').exists(), f'Pasta analysis não encontrada: {ANALYSIS}'\n"
            "runs = pd.read_csv(ANALYSIS / 'run_summary.csv')\n"
            "deltas = pd.read_csv(ANALYSIS / 'metric_deltas_vs_fp32.csv')\n"
            "epochs = pd.read_csv(ANALYSIS / 'training_epochs.csv')\n"
            "quality = json.loads((ANALYSIS / 'data_quality.json').read_text(encoding='utf-8'))\n"
            "summary = json.loads((ANALYSIS / 'analysis_summary.json').read_text(encoding='utf-8'))\n"
            "print('Pasta de análise:', ANALYSIS)\n"
            "print('Validação passou:', quality['passed'])\n"
            "pd.DataFrame({'checagem': quality['checks'].keys(), 'resultado': quality['checks'].values()})"
        ),
        nbf.v4.new_markdown_cell("## Métricas de teste e diferenças para FP32"),
        nbf.v4.new_code_cell(
            "metrics_view = runs[['label', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_ovr_auc', 'logit_mse_vs_fp32', 'prediction_agreement_vs_fp32']].copy()\n"
            "for col in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'macro_ovr_auc', 'prediction_agreement_vs_fp32']:\n"
            "    metrics_view[col] = metrics_view[col].map(lambda x: f'{x:.3%}' if pd.notna(x) else '—')\n"
            "metrics_view['logit_mse_vs_fp32'] = metrics_view['logit_mse_vs_fp32'].map(lambda x: f'{x:.6f}' if pd.notna(x) else '—')\n"
            "metrics_view"
        ),
        nbf.v4.new_code_cell(
            "colors = {'FP32':'#4C78A8','FP16':'#72B7B2','INT8-QAT':'#F58518','INT4-QAT (emulado)':'#E45756','INT8-PTQ':'#54A24B'}\n"
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "for ax, metric, title in zip(axes, ['accuracy','macro_f1','macro_ovr_auc'], ['Acurácia','Macro-F1','AUC macro OvR']):\n"
            "    values = runs[metric] * 100\n"
            "    ax.bar(runs['label'], values, color=[colors[x] for x in runs['label']])\n"
            "    ax.set_title(title); ax.tick_params(axis='x', rotation=35); ax.set_ylim(min(94, values.min()-1), 100.1)\n"
            "plt.tight_layout()"
        ),
        nbf.v4.new_markdown_cell("## Convergência e telemetria de treino"),
        nbf.v4.new_code_cell(
            "metric = 'val_macro_f1' if 'val_macro_f1' in epochs.columns else 'val_accuracy'\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for variant, part in epochs.groupby('variant'):\n"
            "    label = runs.loc[runs.variant.eq(variant), 'label'].iloc[0]\n"
            "    ax.plot(part['epoch'], part[metric] * 100, label=label)\n"
            "ax.set(xlabel='Época', ylabel=f'{metric} (%)', title='Convergência na validação')\n"
            "ax.legend(); plt.tight_layout()"
        ),
        nbf.v4.new_code_cell(
            "train_view = runs[runs.variant != 'int8_ptq'][['label','training_seconds','mean_epoch_seconds','train_examples_per_second','gpu_util_mean_percent','gpu_memory_peak_gib','process_rss_peak_gib']].copy()\n"
            "train_view.round(2)"
        ),
        nbf.v4.new_markdown_cell("## Implantação LiteRT/CPU"),
        nbf.v4.new_code_cell(
            "deploy_view = runs[['label','serialized_litert_mib','estimated_deployment_mib','litert_status','litert_validity','inference_latency_ms','inference_throughput_eps','inference_rss_peak_mib']].copy()\n"
            "deploy_view.round(3)"
        ),
        nbf.v4.new_markdown_cell(
            "## Conclusões e limites\n\n"
            "O dicionário abaixo contém as conclusões calculadas pelo pipeline. O relatório Markdown fornece a interpretação completa e as ressalvas: uma única seed, INT4 emulado, FP16 sem LiteRT e INT8-PTQ classificado como híbrido."
        ),
        nbf.v4.new_code_cell("summary['key_findings']"),
    ]
    path = output / "kmnist_quantization_analysis.ipynb"
    nbf.write(notebook, path)
    client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(output)}})
    client.execute(cwd=str(output))
    nbf.write(notebook, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="Pasta que contém quantization-status.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destino; padrão: <input-root>/analysis.")
    args = parser.parse_args()
    root = args.input_root.resolve()
    output = (args.output_dir or root / "analysis").resolve()
    if not (root / "quantization-status.json").is_file():
        parser.error(f"quantization-status.json não encontrado em {root}")
    output.mkdir(parents=True, exist_ok=True)
    queue = read_json(root / "quantization-status.json", {}) or {}

    records: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    raw: dict[str, Any] = {}
    for variant in VARIANTS:
        record, history, raw_variant = collect_variant(root, variant, queue)
        records.append(record)
        if not history.empty:
            history_frames.append(history)
        raw[variant] = raw_variant
    summary = pd.DataFrame(records)
    histories = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    summary = add_logit_comparison(root, summary)
    baseline = summary.loc[summary["variant"] == "fp32"].iloc[0]
    deltas = summary[["variant", "label", "accuracy", "macro_f1", "macro_ovr_auc", "logit_mse_vs_fp32", "prediction_agreement_vs_fp32"]].copy()
    deltas["accuracy_delta_pp"] = (deltas["accuracy"] - float(baseline["accuracy"])) * 100
    deltas["macro_f1_delta_pp"] = (deltas["macro_f1"] - float(baseline["macro_f1"])) * 100
    deltas["auc_delta_pp"] = (deltas["macro_ovr_auc"] - float(baseline["macro_ovr_auc"])) * 100
    quality = validate_protocol(summary, histories, queue)

    order = {name: index for index, name in enumerate(VARIANTS)}
    summary = summary.sort_values("variant", key=lambda values: values.map(order)).reset_index(drop=True)
    deltas = deltas.sort_values("variant", key=lambda values: values.map(order)).reset_index(drop=True)
    summary.to_csv(output / "run_summary.csv", index=False)
    deltas.to_csv(output / "metric_deltas_vs_fp32.csv", index=False)
    histories.to_csv(output / "training_epochs.csv", index=False)
    (output / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    charts = create_charts(summary, histories, output)

    best = summary.loc[summary["accuracy"].idxmax()]
    ptq = summary.loc[summary["variant"] == "int8_ptq"].iloc[0]
    findings = [
        f"Melhor acurácia: {best['label']} ({best['accuracy']:.3%}).",
        f"INT8-QAT: {summary.loc[summary.variant.eq('int8_qat'), 'accuracy'].iloc[0] - baseline['accuracy']:+.3%} de diferença de acurácia contra FP32.",
        f"INT8-PTQ: {ptq['serialized_litert_mib']:.3f} MiB e {ptq['inference_throughput_eps']:.1f} exemplos/s em LiteRT/CPU; validade {ptq['litert_validity']}.",
        "INT4-QAT é um resultado numérico emulado, não um artefato INT4 LiteRT implantável.",
    ]
    analysis_summary = {
        "input_root": str(root),
        "output_dir": str(output),
        "queue_status": queue.get("status"),
        "quality_passed": quality["passed"],
        "key_findings": findings,
        "generated_charts": [chart.name for chart in charts],
        "source_artifacts": [
            "manifest.json", "status.json", "training_summary.json", "epoch_metrics.csv",
            "test_metrics.json", "model_size.json", "litert_benchmark.json", "telemetry/summary.json",
        ],
    }
    (output / "analysis_summary.json").write_text(json.dumps(analysis_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = write_markdown(output, summary, deltas, histories, quality, queue, charts)
    notebook = write_notebook(output)
    print(json.dumps({"report": str(report), "notebook": str(notebook), "output": str(output), "quality_passed": quality["passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
