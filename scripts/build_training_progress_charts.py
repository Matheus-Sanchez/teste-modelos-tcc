"""Build reproducible progress charts for the controlled training report."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "controlled-augmentation2-mac-m4-aug05"
CHARTS = ROOT / "analysis_reports" / "charts"
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
DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "kmnist": "KMNIST",
    "emnist_balanced": "EMNIST balanced",
    "cifar10": "CIFAR-10",
    "cifar100_coarse": "CIFAR-100 coarse",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "fer2013": "FER2013",
}
BATCHES = (32, 64, 128, 256)
BATCH_COLORS = {32: "#2f6f9f", 64: "#4c9f9b", 128: "#d79b2b", 256: "#c96f3d"}
STATE_COLORS = {
    "completed": "#2f6f9f",
    "active": "#d79b2b",
    "partial": "#8e6aa8",
    "pending": "#d9dee5",
}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def status_path(dataset: str, batch: int) -> Path:
    run = f"{dataset}__unit_interval__all_raw__seed-42"
    return OUTPUT / "batch" / f"batch-{batch:03d}" / dataset / "runs" / run / "status.json"


def run_root(dataset: str, batch: int) -> Path:
    return status_path(dataset, batch).parent


def active_status_file() -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for dataset in DATASETS:
        for batch in BATCHES:
            path = status_path(dataset, batch)
            if read_json(path).get("status") != "running":
                continue
            telemetry = path.parent / "telemetry" / "samples.csv"
            if telemetry.exists():
                candidates.append((telemetry.stat().st_mtime, path))
    return max(candidates, default=(0.0, None))[1]


def cell_state(dataset: str, batch: int, active: Path | None) -> str:
    path = status_path(dataset, batch)
    status = read_json(path).get("status")
    if status == "completed":
        return "completed"
    if status == "running":
        return "active" if active == path else "partial"
    return "pending"


def build_matrix_chart(active: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=160)
    ax.set_facecolor("#f7f8fa")
    for row, dataset in enumerate(DATASETS):
        for col, batch in enumerate(BATCHES):
            state = cell_state(dataset, batch, active)
            ax.add_patch(
                plt.Rectangle(
                    (col - 0.43, row - 0.38),
                    0.86,
                    0.76,
                    facecolor=STATE_COLORS[state],
                    edgecolor="white",
                    linewidth=1.4,
                )
            )
            if state == "completed":
                label = "✓"
            elif state == "active":
                label = "ATIVO"
            elif state == "partial":
                label = "PARCIAL"
            else:
                label = "—"
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8.5 if state in {"active", "partial"} else 15,
                color="white" if state != "pending" else "#6b7280",
                fontweight="bold" if state != "pending" else "normal",
            )
    ax.set_xlim(-0.5, len(BATCHES) - 0.5)
    ax.set_ylim(len(DATASETS) - 0.5, -0.5)
    ax.set_xticks(range(len(BATCHES)), [f"Batch {b}" for b in BATCHES])
    ax.set_yticks(range(len(DATASETS)), [DATASET_LABELS[d] for d in DATASETS])
    ax.tick_params(length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("Estado das 36 células de treinamento", x=0.11, y=0.98, ha="left", fontsize=16, fontweight="bold", color="#25313c")
    fig.text(0.11, 0.915, "✓ concluído · ATIVO em atualização · PARCIAL com checkpoint antigo · — ainda não iniciado", fontsize=9, color="#5d6873")
    ax.legend(
        handles=[Patch(facecolor=color, label=label) for label, color in (("Concluído", STATE_COLORS["completed"]), ("Ativo", STATE_COLORS["active"]), ("Parcial antigo", STATE_COLORS["partial"]), ("Pendente", STATE_COLORS["pending"]))],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(CHARTS / "treinamento_matriz_status.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def completed_timing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for batch in BATCHES:
            root = run_root(dataset, batch)
            status = read_json(root / "status.json").get("status")
            summary = read_json(root / "logs" / "training_summary.json")
            if status != "completed" or not summary.get("mean_epoch_seconds"):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "batch": batch,
                    "label": f"{DATASET_LABELS[dataset]} · B{batch}",
                    "seconds": float(summary["mean_epoch_seconds"]),
                }
            )
    return rows


def build_timing_chart() -> None:
    rows = completed_timing_rows()
    rows.sort(key=lambda row: (DATASETS.index(str(row["dataset"])), BATCHES.index(int(row["batch"]))))
    labels = [str(row["label"]) for row in rows]
    values = [float(row["seconds"]) for row in rows]
    colors = [BATCH_COLORS[int(row["batch"])] for row in rows]
    fig, ax = plt.subplots(figsize=(12.5, 8.0), dpi=160)
    positions = list(range(len(rows)))
    ax.barh(positions, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Segundos médios por época")
    fig.suptitle("Tempo médio por época nas células concluídas", x=0.11, y=0.98, ha="left", fontsize=16, fontweight="bold", color="#25313c")
    fig.text(0.11, 0.925, f"{len(rows)} células com 100 épocas e avaliação final registrada", fontsize=9, color="#5d6873")
    ax.grid(axis="x", color="#d9dee5", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for y, value in zip(positions, values):
        ax.text(value + max(values) * 0.012, y, f"{value:.0f}s", va="center", fontsize=8, color="#3f4b56")
    ax.legend(
        handles=[Patch(facecolor=BATCH_COLORS[b], label=f"Batch {b}") for b in BATCHES],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(CHARTS / "treinamento_tempo_por_epoca.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_hardware_chart(active: Path | None) -> None:
    if active is None:
        return
    telemetry = active.parent / "telemetry" / "samples.csv"
    rows = read_csv_rows(telemetry)
    elapsed = [float(row["elapsed_seconds"]) / 3600 for row in rows]
    gpu = [float(row["gpu_utilization_percent"]) for row in rows]
    process_cpu = [float(row["process_cpu_percent"]) for row in rows]
    ram = [float(row["ram_percent"]) for row in rows]
    active_dataset = active.parents[2].name
    active_batch = int(active.parents[3].name.replace("batch-", ""))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), dpi=160, sharex=True)
    axes[0].plot(elapsed, gpu, color="#2f6f9f", linewidth=1.5, label="GPU Metal")
    axes[0].set_ylabel("GPU (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].set_title(f"Telemetria do lote ativo: {DATASET_LABELS.get(active_dataset, active_dataset)} · batch {active_batch}", loc="left", fontsize=16, fontweight="bold", color="#25313c")
    axes[0].grid(color="#d9dee5", linewidth=0.8, alpha=0.8)
    axes[1].plot(elapsed, process_cpu, color="#c96f3d", linewidth=1.2, label="CPU do processo")
    axes[1].plot(elapsed, ram, color="#8e6aa8", linewidth=1.2, label="RAM reportada")
    axes[1].set_xlabel("Horas desde o início do lote")
    axes[1].set_ylabel("Uso (%)")
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].grid(color="#d9dee5", linewidth=0.8, alpha=0.8)
    axes[1].text(0, -0.32, "A CPU do processo pode superar 100% porque o monitor soma uso em múltiplos núcleos.", transform=axes[1].transAxes, fontsize=9, color="#5d6873")
    for ax in axes:
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(CHARTS / "treinamento_hardware_lote_ativo.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_learning_curve(active: Path | None) -> None:
    if active is None:
        return
    metrics = active.parent / "checkpoints" / "epoch_metrics.csv"
    rows = read_csv_rows(metrics)
    epochs = [int(row["epoch"]) for row in rows]
    val_f1 = [float(row["val_macro_f1"]) for row in rows]
    val_acc = [float(row["val_accuracy"]) for row in rows]
    active_dataset = active.parents[2].name
    active_batch = int(active.parents[3].name.replace("batch-", ""))
    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=160)
    ax.plot(epochs, val_f1, color="#2f6f9f", linewidth=2.0, label="Validação · Macro-F1")
    ax.plot(epochs, val_acc, color="#d79b2b", linewidth=2.0, label="Validação · Accuracy")
    ax.scatter([epochs[-1]], [val_f1[-1]], color="#2f6f9f", s=38, zorder=3)
    ax.scatter([epochs[-1]], [val_acc[-1]], color="#d79b2b", s=38, zorder=3)
    ax.set_xlabel("Época persistida")
    ax.set_ylabel("Métrica de validação")
    ax.set_ylim(0, 1)
    fig.suptitle(f"Curva de validação: {DATASET_LABELS.get(active_dataset, active_dataset)} · batch {active_batch}", x=0.08, y=0.98, ha="left", fontsize=16, fontweight="bold", color="#25313c")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.8)
    ax.legend(loc="lower right", frameon=False)
    fig.text(0.08, 0.915, "A métrica de teste só estará disponível após a conclusão das 100 épocas.", fontsize=9, color="#5d6873")
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(CHARTS / "treinamento_curva_validacao_lote_ativo.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    active = active_status_file()
    build_matrix_chart(active)
    build_timing_chart()
    build_hardware_chart(active)
    build_learning_curve(active)
    print(f"active={active}")
    for path in sorted(CHARTS.glob("treinamento_*.png")):
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
