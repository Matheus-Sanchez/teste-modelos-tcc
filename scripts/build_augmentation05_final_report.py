"""Build the final local-Mac report for the activation experiment at augmentation 0.5.

The source output directories are intentionally not committed because they are large.
This script extracts the final metrics and telemetry from those directories, writes a
small auditable data snapshot, produces figures, generates a Markdown report, and
creates an executed Jupyter notebook that can regenerate the figures from the
published CSV snapshots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import nbformat as nbf
    from nbclient import NotebookClient
except ImportError:  # pragma: no cover - reported with an actionable message in main.
    nbf = None
    NotebookClient = None


DATASET_ORDER = [
    "mnist",
    "fashion_mnist",
    "kmnist",
    "emnist_balanced",
    "cifar10",
    "cifar100_coarse",
    "svhn",
    "gtsrb",
    "fer2013",
]
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
ACTIVATION_ORDER = ["relu", "sigmoid", "softmax"]
ACTIVATION_LABELS = {"relu": "ReLU", "sigmoid": "Sigmoid", "softmax": "Softmax"}
PALETTE = {"relu": "#2463EB", "sigmoid": "#D6A534", "softmax": "#C86B32"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def gib(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value) / 1024**3


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, separator, *body])


def collect_results(root: Path) -> pd.DataFrame:
    """Read one final metric record per dataset × activation condition."""
    rows: list[dict[str, Any]] = []
    for metric_path in sorted(root.rglob("artifacts/test_metrics.json")):
        relative = metric_path.relative_to(root).parts
        if len(relative) < 7:
            continue
        dataset_key, activation = relative[:2]
        run_dir = metric_path.parents[1]
        metric_data = read_json(metric_path)
        classification = metric_data.get("classification", {})
        keras_metrics = metric_data.get("keras_metrics", {})
        training = read_json(run_dir / "logs" / "training_summary.json")
        status = read_json(run_dir / "status.json")
        summary_path = root / dataset_key / activation / dataset_key / "summary.json"
        summary_run = read_json(summary_path).get("runs", [{}])[0]
        telemetry_path = run_dir / "telemetry" / "summary.json"
        telemetry = read_json(telemetry_path) if telemetry_path.exists() else {}
        telemetry_metrics = telemetry.get("metrics", {})

        def telemetry_stat(metric: str, statistic: str) -> float | None:
            value = telemetry_metrics.get(metric, {}).get(statistic)
            return float(value) if value is not None else None

        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": DATASET_LABELS.get(dataset_key, dataset_key),
                "activation_key": activation,
                "activation": ACTIVATION_LABELS.get(activation, activation),
                "status": status.get("status"),
                "epochs_completed": training.get("epochs_completed"),
                "test_accuracy": classification.get("accuracy"),
                "test_balanced_accuracy": classification.get("balanced_accuracy"),
                "test_macro_f1": classification.get("macro_f1"),
                "test_macro_precision": classification.get("macro_precision"),
                "test_macro_recall": classification.get("macro_recall"),
                "test_loss": keras_metrics.get("loss"),
                "test_samples": summary_run.get("test_samples"),
                "training_seconds": training.get("training_seconds", training.get("total_seconds")),
                "evaluation_seconds": training.get("evaluation_seconds"),
                "mean_epoch_seconds": training.get("mean_epoch_seconds"),
                "mean_train_examples_per_second": training.get("mean_train_examples_per_second"),
                "telemetry_samples": telemetry.get("sample_count", summary_run.get("telemetry_samples")),
                "gpu_util_mean_pct": telemetry_stat("gpu_utilization_percent", "mean"),
                "gpu_util_max_pct": telemetry_stat("gpu_utilization_percent", "max"),
                "gpu_memory_mean_bytes": telemetry_stat("gpu_memory_used_bytes", "mean"),
                "peak_gpu_memory_bytes": summary_run.get("peak_gpu_memory_used_bytes", telemetry_stat("gpu_memory_used_bytes", "max")),
                "process_cpu_mean_pct": telemetry_stat("process_cpu_percent", "mean"),
                "peak_process_cpu_pct": telemetry_stat("process_cpu_percent", "max"),
                "process_rss_mean_bytes": telemetry_stat("process_rss_bytes", "mean"),
                "peak_process_rss_bytes": summary_run.get("peak_process_rss_bytes", telemetry_stat("process_rss_bytes", "max")),
                "ram_mean_pct": telemetry_stat("ram_percent", "mean"),
                "peak_ram_percent": telemetry_stat("ram_percent", "max"),
                "thermal_pressure": summary_run.get("thermal_pressure"),
                "metric_path": str(metric_path.relative_to(root.parent.parent.parent)),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise FileNotFoundError(f"Nenhum artifacts/test_metrics.json foi encontrado em {root}")
    result["dataset_key"] = pd.Categorical(result["dataset_key"], DATASET_ORDER, ordered=True)
    result["activation_key"] = pd.Categorical(result["activation_key"], ACTIVATION_ORDER, ordered=True)
    return result.sort_values(["dataset_key", "activation_key"]).reset_index(drop=True)


def sample_weighted_mean(frame: pd.DataFrame, metric: str) -> float:
    valid = frame[[metric, "telemetry_samples"]].dropna()
    return float(np.average(valid[metric], weights=valid["telemetry_samples"]))


def collect_hardware_metadata(root: Path) -> dict[str, Any]:
    environment_path = next(root.rglob("telemetry/environment.json"), None)
    if environment_path is None:
        return {}
    environment = read_json(environment_path)
    hardware = environment.get("hardware", {})
    gpu = hardware.get("gpus", [{}])[0]
    packages = environment.get("packages", {})
    return {
        "cpu_logical_cores": hardware.get("cpu_count_logical"),
        "ram_gib": gib(hardware.get("ram_total_bytes")),
        "gpu_name": gpu.get("name"),
        "gpu_cores": gpu.get("gpu_core_count"),
        "gpu_backend": hardware.get("gpu_backend"),
        "gpu_memory_kind": gpu.get("memory_kind"),
        "metal_version": hardware.get("metal_version"),
        "macos_version": hardware.get("macos_version"),
        "python_version": environment.get("python", {}).get("version"),
        "tensorflow": packages.get("tensorflow"),
        "tensorflow_metal": packages.get("tensorflow_metal"),
    }


def collect_analysis(project: Path) -> dict[str, Any]:
    new_root = project / "outputs" / "controlled-augmentation05-activations-mac2" / "activations"
    old_root = project / "outputs" / "controlled-augmentation2-activations-mac2" / "activations"
    runs = collect_results(new_root)
    old_runs = collect_results(old_root)

    baseline = old_runs[
        ["dataset_key", "activation_key", "test_accuracy", "test_macro_f1", "mean_epoch_seconds", "training_seconds"]
    ].rename(
        columns={
            "test_accuracy": "old_accuracy",
            "test_macro_f1": "old_macro_f1",
            "mean_epoch_seconds": "old_mean_epoch_seconds",
            "training_seconds": "old_training_seconds",
        }
    )
    comparison = runs.merge(baseline, on=["dataset_key", "activation_key"], how="inner")
    comparison["model"] = comparison["dataset"] + " · " + comparison["activation"]
    comparison["delta_accuracy_pp"] = (comparison["test_accuracy"] - comparison["old_accuracy"]) * 100
    comparison["delta_macro_f1_pp"] = (comparison["test_macro_f1"] - comparison["old_macro_f1"]) * 100
    comparison["time_reduction_pct"] = (
        (comparison["old_mean_epoch_seconds"] - comparison["mean_epoch_seconds"])
        / comparison["old_mean_epoch_seconds"]
        * 100
    )
    comparison = comparison.sort_values(["dataset_key", "activation_key"]).reset_index(drop=True)

    total_training_seconds = float(runs["training_seconds"].sum())
    total_evaluation_seconds = float(runs["evaluation_seconds"].sum())
    weighted_epoch_seconds = total_training_seconds / float(runs["epochs_completed"].sum())
    activation_winners = (
        runs.loc[runs.groupby("dataset_key", observed=True)["test_macro_f1"].idxmax(), ["dataset_key", "activation_key"]]
        .groupby("activation_key", observed=True)
        .size()
        .reindex(ACTIVATION_ORDER, fill_value=0)
        .to_dict()
    )
    softmax = runs.loc[runs["activation_key"] == "softmax", "test_macro_f1"]
    thermal_counts = runs["thermal_pressure"].fillna("indisponível").value_counts().to_dict()
    hardware_usage = {
        "sample_count": int(runs["telemetry_samples"].sum()),
        "gpu_util_mean_pct": sample_weighted_mean(runs, "gpu_util_mean_pct"),
        "gpu_util_max_pct": float(runs["gpu_util_max_pct"].max()),
        "gpu_memory_mean_gib": sample_weighted_mean(runs, "gpu_memory_mean_bytes") / 1024**3,
        "gpu_memory_peak_gib": float(runs["peak_gpu_memory_bytes"].max()) / 1024**3,
        "process_cpu_mean_pct": sample_weighted_mean(runs, "process_cpu_mean_pct"),
        "process_cpu_peak_pct": float(runs["peak_process_cpu_pct"].max()),
        "process_rss_mean_gib": sample_weighted_mean(runs, "process_rss_mean_bytes") / 1024**3,
        "process_rss_peak_gib": float(runs["peak_process_rss_bytes"].max()) / 1024**3,
        "ram_mean_pct": sample_weighted_mean(runs, "ram_mean_pct"),
        "ram_peak_pct": float(runs["peak_ram_percent"].max()),
        "thermal_pressure_counts": thermal_counts,
    }
    overview = {
        "runs": int(len(runs)),
        "completed_runs": int((runs["status"] == "completed").sum()),
        "datasets": int(runs["dataset_key"].nunique()),
        "activations": int(runs["activation_key"].nunique()),
        "epochs_completed": int(runs["epochs_completed"].sum()),
        "training_seconds": total_training_seconds,
        "training_hours": total_training_seconds / 3600,
        "evaluation_seconds": total_evaluation_seconds,
        "weighted_epoch_seconds": weighted_epoch_seconds,
        "best_run": runs.loc[runs["test_macro_f1"].idxmax(), "dataset"]
        + " / "
        + runs.loc[runs["test_macro_f1"].idxmax(), "activation"],
        "best_macro_f1": float(runs["test_macro_f1"].max()),
        "softmax_macro_f1_min": float(softmax.min()),
        "softmax_macro_f1_max": float(softmax.max()),
        "activation_winners": activation_winners,
        "paired_comparisons": int(len(comparison)),
        "mean_time_reduction_pct": float(comparison["time_reduction_pct"].mean()),
        "mean_delta_macro_f1_pp": float(comparison["delta_macro_f1_pp"].mean()),
        "hardware_usage": hardware_usage,
        "hardware": collect_hardware_metadata(new_root),
    }
    return {"runs": runs, "comparison": comparison, "overview": overview}


def configure_chart_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "axes.edgecolor": "#9CA3AF",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def grouped_horizontal_chart(
    runs: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    path: Path,
    percent_axis: bool = False,
) -> None:
    datasets = [DATASET_LABELS[key] for key in DATASET_ORDER]
    y = np.arange(len(datasets))
    height = 0.23
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    for index, activation in enumerate(ACTIVATION_ORDER):
        values = (
            runs.loc[runs["activation_key"] == activation]
            .set_index("dataset_key")
            .reindex(DATASET_ORDER)[metric]
            .to_numpy()
        )
        ax.barh(
            y + (index - 1) * height,
            values,
            height=height,
            color=PALETTE[activation],
            label=ACTIVATION_LABELS[activation],
            edgecolor="#1F2937",
            linewidth=0.35,
        )
    ax.set_yticks(y, datasets)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=16)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    if percent_axis:
        ax.set_xlim(0, 1.02)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.text(0.125, 0.01, "27 runs finais; 100 epochs por condição; seed 42.", color="#6B7280", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, path)


def timing_chart(runs: pd.DataFrame, path: Path) -> None:
    datasets = [DATASET_LABELS[key] for key in DATASET_ORDER]
    y = np.arange(len(datasets))
    height = 0.23
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    for index, activation in enumerate(ACTIVATION_ORDER):
        values = (
            runs.loc[runs["activation_key"] == activation]
            .set_index("dataset_key")
            .reindex(DATASET_ORDER)["training_seconds"]
            .to_numpy()
            / 3600
        )
        ax.barh(
            y + (index - 1) * height,
            values,
            height=height,
            color=PALETTE[activation],
            label=ACTIVATION_LABELS[activation],
            edgecolor="#1F2937",
            linewidth=0.35,
        )
    ax.set_yticks(y, datasets)
    ax.invert_yaxis()
    ax.set_xlabel("Horas de treinamento (soma das 100 épocas)")
    ax.set_title("Tempo de treinamento por dataset e ativação", loc="left", pad=16)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.text(0.125, 0.01, "A avaliação final é reportada separadamente e não entra nas barras.", color="#6B7280", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, path)


def comparison_speed_chart(comparison: pd.DataFrame, path: Path) -> None:
    labels = comparison["model"].tolist()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for index, row in comparison.iterrows():
        position = list(comparison.index).index(index)
        ax.plot(
            [row["old_mean_epoch_seconds"], row["mean_epoch_seconds"]],
            [position, position],
            color="#9CA3AF",
            linewidth=2,
            zorder=1,
        )
        ax.scatter(row["old_mean_epoch_seconds"], position, s=58, color="#6B7280", label="extra_fraction = 2,0" if position == 0 else None, zorder=2)
        ax.scatter(row["mean_epoch_seconds"], position, s=58, color="#2463EB", label="extra_fraction = 0,5" if position == 0 else None, zorder=2)
        ax.text(
            max(row["old_mean_epoch_seconds"], row["mean_epoch_seconds"]) + 5,
            position,
            f"−{row['time_reduction_pct']:.1f}%",
            va="center",
            fontsize=9,
            color="#374151",
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Segundos por época")
    ax.set_title("Tempo médio por época nos pares com baseline", loc="left", pad=16)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    figure_legend = fig.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
    )
    figure_legend.set_in_layout(False)
    fig.text(0.125, 0.01, "Rótulos indicam redução de tempo ao passar de 2,0 para 0,5.", color="#6B7280", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save_figure(fig, path)


def comparison_quality_chart(comparison: pd.DataFrame, path: Path) -> None:
    ordered = comparison.sort_values("delta_macro_f1_pp")
    colors = ["#2463EB" if value >= 0 else "#C86B32" for value in ordered["delta_macro_f1_pp"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.barh(ordered["model"], ordered["delta_macro_f1_pp"], color=colors, edgecolor="#1F2937", linewidth=0.35)
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_xlabel("Variação do macro-F1 de teste (pontos percentuais; 0,5 − 2,0)")
    ax.set_title("Variação de macro-F1 nos pares com baseline", loc="left", pad=16)
    ax.set_xlim(ordered["delta_macro_f1_pp"].min() - 0.45, ordered["delta_macro_f1_pp"].max() + 0.25)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, ordered["delta_macro_f1_pp"]):
        x = value + (0.05 if value >= 0 else -0.05)
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    fig.text(0.125, 0.01, "Apenas cinco condições têm métricas finais nos dois diretórios de augmentation.", color="#6B7280", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, path)


def hardware_chart(runs: pd.DataFrame, overview: dict[str, Any], path: Path) -> None:
    ordered = runs.copy()
    ordered["label"] = ordered["dataset"].str.replace(" Balanced", "", regex=False) + "\n" + ordered["activation"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5), gridspec_kw={"width_ratios": [1.05, 1]})
    for activation in ACTIVATION_ORDER:
        subset = ordered.loc[ordered["activation_key"] == activation]
        axes[0].scatter(
            subset["gpu_util_mean_pct"],
            subset["mean_train_examples_per_second"],
            s=65,
            color=PALETTE[activation],
            edgecolor="#1F2937",
            linewidth=0.45,
            label=ACTIVATION_LABELS[activation],
            alpha=0.9,
        )
    axes[0].set_xlabel("Utilização média da GPU (%)")
    axes[0].set_ylabel("Exemplos de treino por segundo")
    axes[0].set_title("Throughput versus uso médio da GPU", loc="left", pad=14)
    axes[0].grid(color="#E5E7EB", linewidth=0.8)
    axes[0].legend(frameon=False, loc="best")

    y = np.arange(len(ordered))
    colors = [PALETTE[key] for key in ordered["activation_key"]]
    axes[1].barh(
        y,
        ordered["peak_gpu_memory_bytes"] / 1024**3,
        color=colors,
        edgecolor="#1F2937",
        linewidth=0.3,
    )
    axes[1].set_yticks(y, ordered["label"], fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Pico de memória GPU/unificada (GiB)")
    axes[1].set_title("Pico de memória por run", loc="left", pad=14)
    axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axes[1].set_axisbelow(True)
    usage = overview["hardware_usage"]
    fig.text(
        0.125,
        0.01,
        "Telemetria a cada 5 s. Média ponderada: "
        f"GPU {usage['gpu_util_mean_pct']:.1f}% · RAM do processo {usage['process_rss_mean_gib']:.2f} GiB · "
        f"pico de memória GPU {usage['gpu_memory_peak_gib']:.2f} GiB.",
        color="#6B7280",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, path)


def activation_summary_chart(runs: pd.DataFrame, path: Path) -> None:
    grouped = (
        runs.groupby("activation_key", observed=True)
        .agg(
            mean_macro_f1=("test_macro_f1", "mean"),
            median_macro_f1=("test_macro_f1", "median"),
            mean_epoch_seconds=("mean_epoch_seconds", "mean"),
        )
        .reindex(ACTIVATION_ORDER)
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].bar(
        [ACTIVATION_LABELS[key] for key in grouped["activation_key"]],
        grouped["mean_macro_f1"],
        color=[PALETTE[key] for key in grouped["activation_key"]],
        edgecolor="#1F2937",
        linewidth=0.4,
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].set_ylabel("Média simples entre datasets")
    axes[0].set_title("Macro-F1 médio por ativação", loc="left", pad=14)
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[1].bar(
        [ACTIVATION_LABELS[key] for key in grouped["activation_key"]],
        grouped["mean_epoch_seconds"],
        color=[PALETTE[key] for key in grouped["activation_key"]],
        edgecolor="#1F2937",
        linewidth=0.4,
    )
    axes[1].set_ylabel("Segundos por época")
    axes[1].set_title("Tempo médio por época", loc="left", pad=14)
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[1].set_axisbelow(True)
    fig.text(
        0.125,
        0.01,
        "As médias entre datasets são descritivas: os datasets têm números de classes e dificuldades diferentes.",
        color="#6B7280",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, path)


def render_figures(runs: pd.DataFrame, comparison: pd.DataFrame, overview: dict[str, Any], figure_dir: Path) -> list[Path]:
    configure_chart_style()
    figure_paths = [
        figure_dir / "01_macro_f1_por_dataset_ativacao.png",
        figure_dir / "02_accuracy_por_dataset_ativacao.png",
        figure_dir / "03_tempo_treinamento_por_dataset_ativacao.png",
        figure_dir / "04_resumo_por_ativacao.png",
        figure_dir / "05_comparacao_tempo_augmentation.png",
        figure_dir / "06_comparacao_macro_f1_augmentation.png",
        figure_dir / "07_uso_hardware.png",
    ]
    grouped_horizontal_chart(runs, "test_macro_f1", "Macro-F1 de teste por dataset e ativação", "Macro-F1 de teste", figure_paths[0], percent_axis=True)
    grouped_horizontal_chart(runs, "test_accuracy", "Accuracy de teste por dataset e ativação", "Accuracy de teste", figure_paths[1], percent_axis=True)
    timing_chart(runs, figure_paths[2])
    activation_summary_chart(runs, figure_paths[3])
    comparison_speed_chart(comparison, figure_paths[4])
    comparison_quality_chart(comparison, figure_paths[5])
    hardware_chart(runs, overview, figure_paths[6])
    return figure_paths


def write_data_snapshot(analysis: dict[str, Any], output_dir: Path) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis["runs"].to_csv(data_dir / "resultados_augmentation05.csv", index=False)
    analysis["comparison"].to_csv(data_dir / "comparacao_augmentation20_vs_05.csv", index=False)
    safe_overview = analysis["overview"].copy()
    (data_dir / "resumo_augmentation05.json").write_text(
        json.dumps(safe_overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_report(analysis: dict[str, Any], output_dir: Path) -> Path:
    runs = analysis["runs"]
    comparison = analysis["comparison"]
    overview = analysis["overview"]
    hardware = overview["hardware"]
    usage = overview["hardware_usage"]
    best = runs.loc[runs["test_macro_f1"].idxmax()]
    winners = overview["activation_winners"]
    result_rows = [
        [
            row.dataset,
            row.activation,
            pct(row.test_accuracy),
            pct(row.test_balanced_accuracy),
            pct(row.test_macro_f1),
            num(row.mean_epoch_seconds, 1),
            num(row.training_seconds / 3600, 2),
        ]
        for row in runs.itertuples()
    ]
    comparison_rows = [
        [
            row.model,
            pct(row.old_accuracy),
            pct(row.test_accuracy),
            f"{row.delta_accuracy_pp:+.2f}",
            pct(row.old_macro_f1),
            pct(row.test_macro_f1),
            f"{row.delta_macro_f1_pp:+.2f}",
            f"{row.old_mean_epoch_seconds:.1f} → {row.mean_epoch_seconds:.1f}",
            f"{row.time_reduction_pct:.1f}%",
        ]
        for row in comparison.itertuples()
    ]
    thermal = ", ".join(f"{name}: {count}" for name, count in sorted(usage["thermal_pressure_counts"].items()))
    hardware_rows = [
        ["CPU lógico", f"{hardware.get('cpu_logical_cores')} núcleos"],
        ["Memória unificada", f"{hardware.get('ram_gib'):.0f} GiB"],
        ["GPU", f"{hardware.get('gpu_name')} · {hardware.get('gpu_cores')} núcleos"],
        ["Backend", f"{hardware.get('gpu_backend')} · {hardware.get('metal_version')}"],
        ["Sistema", f"macOS {hardware.get('macos_version')} · Python {hardware.get('python_version', '').split()[0]}"],
        ["Framework", f"TensorFlow {hardware.get('tensorflow')} · tensorflow-metal {hardware.get('tensorflow_metal')}"],
        ["Amostras de telemetria", f"{usage['sample_count']:,} a cada 5 s"],
        ["GPU — utilização", f"média ponderada {usage['gpu_util_mean_pct']:.1f}% · máximo {usage['gpu_util_max_pct']:.1f}%"],
        ["GPU — memória compartilhada", f"média ponderada {usage['gpu_memory_mean_gib']:.2f} GiB · pico {usage['gpu_memory_peak_gib']:.2f} GiB"],
        ["Processo — CPU", f"média ponderada {usage['process_cpu_mean_pct']:.1f}% · pico {usage['process_cpu_peak_pct']:.1f}% (multicore)"],
        ["Processo — RSS", f"média ponderada {usage['process_rss_mean_gib']:.2f} GiB · pico {usage['process_rss_peak_gib']:.2f} GiB"],
        ["RAM do sistema", f"média ponderada {usage['ram_mean_pct']:.1f}% · pico {usage['ram_peak_pct']:.1f}%"],
        ["Pressão térmica", thermal],
    ]
    report = f"""# Relatório final — ativações com augmentation 0,5 (Mac M4)

**Escopo:** apenas a rodada local deste Mac: 9 datasets × 3 ativações, seed 42, 100 epochs por condição. Testes feitos em outro Mac não entram nas contagens, tempos nem comparações.

## Resumo técnico

- A rodada foi concluída: **{overview['completed_runs']}/{overview['runs']} runs**, **{overview['epochs_completed']:,} epochs** e **{overview['training_hours']:.2f} horas** de treinamento acumulado. Não há run pendente.
- O melhor resultado é **{best.dataset}/{best.activation}**, com macro-F1 de teste de **{pct(best.test_macro_f1)}** e accuracy de **{pct(best.test_accuracy)}**.
- A ativação Sigmoid venceu {winners.get('sigmoid', 0)} de 9 datasets por macro-F1; ReLU venceu {winners.get('relu', 0)}; Softmax não venceu nenhum. O intervalo de macro-F1 do Softmax foi **{pct(overview['softmax_macro_f1_min'])}–{pct(overview['softmax_macro_f1_max'])}**, caracterizando desempenho degenerado neste protocolo.
- Nos **{overview['paired_comparisons']} pares** com baseline final em augmentation 2,0, `extra_fraction=0,5` reduziu o tempo médio por epoch em **{overview['mean_time_reduction_pct']:.1f}%**. A variação média simples de macro-F1 foi **{overview['mean_delta_macro_f1_pp']:+.2f} p.p.**, portanto a qualidade não melhorou de forma uniforme.

## Resultados por dataset e ativação

![Macro-F1 de teste por dataset e ativação](figures/01_macro_f1_por_dataset_ativacao.png)

O ranking é majoritariamente favorável ao Sigmoid, exceto no GTSRB, em que ReLU apresentou macro-F1 maior. A comparação entre datasets deve ser lida com cautela porque os conjuntos têm diferentes números de classes, distribuições e níveis de dificuldade.

![Accuracy de teste por dataset e ativação](figures/02_accuracy_por_dataset_ativacao.png)

{markdown_table(["Dataset", "Ativação", "Accuracy", "Balanced accuracy", "Macro-F1", "s/epoch", "Tempo treino (h)"], result_rows)}

Dados tabulares completos: [`data/resultados_augmentation05.csv`](data/resultados_augmentation05.csv).

## Tempo de treinamento e comparação com augmentation 2,0

![Tempo de treinamento por dataset e ativação](figures/03_tempo_treinamento_por_dataset_ativacao.png)

O total de **{overview['training_hours']:.2f} h** é a soma do tempo de treinamento registrado por cada run, não tempo de relógio com paralelismo. O tempo médio ponderado foi de **{overview['weighted_epoch_seconds']:.1f} s/epoch**; a avaliação final acrescentou **{overview['evaluation_seconds'] / 60:.1f} min** acumulados.

![Resumo por ativação](figures/04_resumo_por_ativacao.png)

As médias de macro-F1 por ativação são apenas descritivas — não constituem uma medida de desempenho global comum, pois os datasets não têm a mesma dificuldade nem a mesma taxonomia de classes.

![Tempo médio por época nos pares com baseline](figures/05_comparacao_tempo_augmentation.png)

![Variação de macro-F1 nos pares com baseline](figures/06_comparacao_macro_f1_augmentation.png)

{markdown_table(["Condição", "Acc. 2,0", "Acc. 0,5", "Δ acc. (p.p.)", "F1 2,0", "F1 0,5", "Δ F1 (p.p.)", "s/epoch 2,0 → 0,5", "Redução"], comparison_rows)}

Dados comparativos: [`data/comparacao_augmentation20_vs_05.csv`](data/comparacao_augmentation20_vs_05.csv). Só entram condições com métricas finais persistidas nos dois diretórios: MNIST (três ativações) e Fashion-MNIST (ReLU e Sigmoid).

## Hardware e uso observado

{markdown_table(["Métrica", "Valor"], hardware_rows)}

![Uso de hardware por run](figures/07_uso_hardware.png)

O backend reportado pelos próprios runs foi Apple Metal com memória unificada. Os picos de memória devem ser lidos como uso da memória compartilhada do driver/SoC, não como VRAM dedicada. O percentual de CPU é do processo e pode exceder 100% em uma máquina multicore. As estatísticas de uso são agregações de amostras a cada 5 segundos; elas descrevem execução observada, mas não medem energia elétrica nem permitem inferir causalidade sobre throughput.

## Escopo, dados e método

- **Protocolo:** `extra_fraction=0,5`, batch de ativação 256, `mixed_float16`, seed 42, normalização `unit_interval`, `all_raw`, 100 epochs, mesmo conjunto de transformações da configuração `configs/controlled-augmentation05-mac-m4.yaml`.
- **Métricas de qualidade:** accuracy, balanced accuracy e macro-F1 são calculadas no conjunto de teste final; macro-F1 é a média não ponderada de F1 por classe.
- **Tempo:** `training_seconds` e `mean_epoch_seconds` vêm de `logs/training_summary.json`; avaliação final vem do mesmo arquivo.
- **Hardware:** telemetria das 27 execuções (`telemetry/summary.json`), com backend e versões em `telemetry/environment.json`.
- **Comparação 2,0 vs. 0,5:** diferença é novo menos antigo para qualidade (pontos percentuais) e redução relativa do tempo por epoch para desempenho operacional.

## Limitações e verificações de robustez

- Há **uma única seed (42)** por condição. As diferenças são descritivas, não estimativas de significância estatística.
- Apenas cinco pares têm baseline final de augmentation 2,0; não é válido extrapolar a comparação direta para os demais 22 runs.
- A média entre datasets não substitui análise dentro de cada dataset; classes, resolução, volume e dificuldade variam.
- Todos os 27 status estão como `completed`, cada `training_summary.json` registra 100 epochs e cada condição possui `artifacts/test_metrics.json` e telemetria. A pressão térmica agregada permaneceu nominal nas 27 condições.
- O escopo exclui deliberadamente resultados produzidos em outro Mac.

## Próximos passos recomendados

1. Repetir as condições mais promissoras e os casos fracos com múltiplas seeds para quantificar variabilidade.
2. Tratar Softmax como condição de falha do protocolo atual: revisar sua posição na arquitetura e validar logits/perdas antes de novos treinamentos extensos.
3. Completar baselines 2,0 que faltam caso a decisão exija comparar todos os datasets diretamente.
4. Para uso prático, priorizar Sigmoid como primeiro candidato e ReLU como alternativa para GTSRB, sempre validando com replicações.

## Artefatos reprodutíveis

- [Notebook executado](../../notebooks/augmentation05_final_analysis.ipynb)
- [Script de geração](../../scripts/build_augmentation05_final_report.py)
- [Snapshot de métricas](data/resultados_augmentation05.csv)
- [Resumo consolidado](data/resumo_augmentation05.json)

Para reconstruir a partir dos outputs locais, execute:

```bash
.venv-mac/bin/python -m pip install -r requirements/reporting-notebook.txt
.venv-mac/bin/python scripts/build_augmentation05_final_report.py --execute
```
"""
    report_path = output_dir / "README.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def build_notebook(project: Path, analysis: dict[str, Any], notebook_path: Path) -> None:
    if nbf is None:
        raise RuntimeError("Instale nbformat, nbclient e ipykernel antes de gerar o notebook.")
    overview = analysis["overview"]
    best = analysis["runs"].loc[analysis["runs"]["test_macro_f1"].idxmax()]
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Análise final — ativações com augmentation 0,5 (Mac M4)\n\n"
            "## tl;dr\n\n"
            f"A rodada local está completa: **{overview['completed_runs']}/27 runs**, {overview['epochs_completed']:,} epochs e "
            f"**{overview['training_hours']:.2f} h** de treinamento acumulado. O melhor resultado é "
            f"**{best.dataset}/{best.activation}** (macro-F1 de teste {pct(best.test_macro_f1)}). "
            f"Nos cinco pares comparáveis com augmentation 2,0, a versão 0,5 reduziu o tempo médio por epoch em "
            f"**{overview['mean_time_reduction_pct']:.1f}%**, com efeito misto em macro-F1."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "### Key Assumptions\n\n"
            "- O escopo é somente o Mac M4 local: 9 datasets × 3 ativações, seed 42 e 100 epochs por condição.\n"
            "- Os dados publicados no diretório `docs/augmentation05_final/data/` são snapshots derivados dos artefatos finais locais.\n"
            "- Os outputs brutos não são versionados por volume; para reextrair os dados é necessário ter os diretórios `outputs/controlled-augmentation*-activations-mac2/`.\n"
            "- A comparação entre 0,5 e 2,0 usa somente os cinco pares com métricas finais nos dois lados."
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Load published snapshots"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "PROJECT = Path.cwd()\n"
            "REPORT_DIR = PROJECT / 'docs' / 'augmentation05_final'\n"
            "DATA_DIR = REPORT_DIR / 'data'\n"
            "FIGURE_DIR = REPORT_DIR / 'figures'\n"
            "sys.path.insert(0, str(PROJECT / 'scripts'))\n"
            "from build_augmentation05_final_report import render_figures\n\n"
            "runs = pd.read_csv(DATA_DIR / 'resultados_augmentation05.csv')\n"
            "comparison = pd.read_csv(DATA_DIR / 'comparacao_augmentation20_vs_05.csv')\n"
            "overview = json.loads((DATA_DIR / 'resumo_augmentation05.json').read_text(encoding='utf-8'))\n"
            "runs.head()"
        ),
        nbf.v4.new_markdown_cell("### 2. Validate final artifacts"),
        nbf.v4.new_code_cell(
            "assert len(runs) == 27, f'Esperava 27 runs; encontrei {len(runs)}'\n"
            "assert (runs['status'] == 'completed').all(), 'Há run sem status completed'\n"
            "assert (runs['epochs_completed'] == 100).all(), 'Há run sem 100 epochs'\n"
            "assert runs['test_macro_f1'].notna().all(), 'Há macro-F1 final ausente'\n"
            "print(f\"Runs: {len(runs)}; completed: {(runs.status == 'completed').sum()}; epochs: {int(runs.epochs_completed.sum())}\")\n"
            "print(f\"Comparações pareadas: {len(comparison)}; telemetria: {overview['hardware_usage']['sample_count']:,} amostras\")"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 3. Complete final-result table"),
        nbf.v4.new_code_cell(
            "display_columns = ['dataset', 'activation', 'test_accuracy', 'test_balanced_accuracy', 'test_macro_f1', 'mean_epoch_seconds', 'training_seconds']\n"
            "runs[display_columns].assign(training_hours=lambda frame: frame.training_seconds / 3600).drop(columns='training_seconds').round(4)"
        ),
        nbf.v4.new_markdown_cell("### 4. Regenerate all report figures"),
        nbf.v4.new_code_cell(
            "figure_paths = render_figures(runs, comparison, overview, FIGURE_DIR)\n"
            "for figure_path in figure_paths:\n"
            "    display(Image(filename=str(figure_path), width=920))\n"
            "print(f'{len(figure_paths)} gráficos regenerados em {FIGURE_DIR.relative_to(PROJECT)}')"
        ),
        nbf.v4.new_markdown_cell("### 5. Paired comparison against augmentation 2.0"),
        nbf.v4.new_code_cell(
            "comparison_columns = ['model', 'old_accuracy', 'test_accuracy', 'delta_accuracy_pp', 'old_macro_f1', 'test_macro_f1', 'delta_macro_f1_pp', 'old_mean_epoch_seconds', 'mean_epoch_seconds', 'time_reduction_pct']\n"
            "comparison[comparison_columns].round(4)"
        ),
        nbf.v4.new_markdown_cell("### 6. Hardware and observed usage"),
        nbf.v4.new_code_cell(
            "hardware = overview['hardware']\n"
            "usage = overview['hardware_usage']\n"
            "pd.DataFrame([\n"
            "    ('GPU', f\"{hardware['gpu_name']} ({hardware['gpu_cores']} cores), {hardware['metal_version']}\"),\n"
            "    ('Backend', hardware['gpu_backend']),\n"
            "    ('Unified memory', f\"{hardware['ram_gib']:.0f} GiB\"),\n"
            "    ('GPU utilization', f\"mean {usage['gpu_util_mean_pct']:.1f}% / peak {usage['gpu_util_max_pct']:.1f}%\"),\n"
            "    ('GPU shared memory', f\"mean {usage['gpu_memory_mean_gib']:.2f} GiB / peak {usage['gpu_memory_peak_gib']:.2f} GiB\"),\n"
            "    ('Thermal pressure', str(usage['thermal_pressure_counts'])),\n"
            "], columns=['Metric', 'Observed value'])"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- Sigmoid teve o maior macro-F1 em 8 dos 9 datasets; ReLU foi melhor no GTSRB.\n"
            "- Softmax deve ser interpretado como condição degenerada neste protocolo, não como alternativa competitiva.\n"
            "- A redução de `extra_fraction` diminuiu materialmente o custo de treino nos pares disponíveis, mas não preservou a qualidade de maneira uniforme.\n"
            "- Há apenas uma seed por condição. Use múltiplas seeds antes de transformar as diferenças em uma decisão definitiva."
        ),
    ]
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, notebook_path)


def execute_notebook(notebook_path: Path, project: Path) -> None:
    if nbf is None or NotebookClient is None:
        raise RuntimeError("Instale nbformat, nbclient e ipykernel antes de executar o notebook.")
    notebook = nbf.read(notebook_path, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(project)}})
    client.execute()
    nbf.write(notebook, notebook_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--execute", action="store_true", help="Execute the generated notebook top to bottom.")
    args = parser.parse_args()
    project = args.project.resolve()
    report_dir = project / "docs" / "augmentation05_final"
    notebook_path = project / "notebooks" / "augmentation05_final_analysis.ipynb"

    analysis = collect_analysis(project)
    if analysis["overview"]["completed_runs"] != 27 or analysis["overview"]["runs"] != 27:
        raise RuntimeError("A rodada augmentation 0,5 não está completa: o relatório final não será gerado.")
    write_data_snapshot(analysis, report_dir)
    render_figures(analysis["runs"], analysis["comparison"], analysis["overview"], report_dir / "figures")
    report_path = write_report(analysis, report_dir)
    build_notebook(project, analysis, notebook_path)
    if args.execute:
        execute_notebook(notebook_path, project)
    print(report_path)
    print(notebook_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
