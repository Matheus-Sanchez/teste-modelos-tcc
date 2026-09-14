from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


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
ACTIVATIONS = ("relu", "sigmoid", "softmax")
BATCH_SIZES = (32, 64, 128, 256)
EXPECTED_EPOCHS = 100
STALE_RUNNING_AFTER_HOURS = 1.0

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


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_epochs(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def classify_status(row: dict[str, Any], observed_at: datetime) -> None:
    """Mark an unrefreshed `running` marker as stale without mutating source files."""
    row["reported_status"] = row["status"]
    updated_at = parse_timestamp(row.get("updated_at"))
    age_hours = (observed_at - updated_at).total_seconds() / 3600.0 if updated_at else None
    row["status_age_hours"] = round(age_hours, 3) if age_hours is not None else None
    if row["status"] == "running" and age_hours is not None and age_hours > STALE_RUNNING_AFTER_HOURS:
        row["status"] = "stale"


def epoch_stats(run_root: Path) -> dict[str, Any]:
    # The current runner writes epoch_metrics.csv under checkpoints. Older
    # runs may have it under logs, so retain that fallback for portability.
    paths = (run_root / "checkpoints" / "epoch_metrics.csv", run_root / "logs" / "epoch_metrics.csv")
    rows: list[dict[str, str]] = []
    used: Path | None = None
    for path in paths:
        rows = read_epochs(path)
        if rows:
            used = path
            break
    seconds: list[float] = []
    last_macro: float | None = None
    last_seconds: float | None = None
    for row in rows:
        try:
            value = float(row.get("epoch_seconds", ""))
            if math.isfinite(value):
                seconds.append(value)
        except (TypeError, ValueError):
            pass
        try:
            value = float(row.get("val_macro_f1", ""))
            if math.isfinite(value):
                last_macro = value
        except (TypeError, ValueError):
            pass
        try:
            value = float(row.get("epoch_seconds", ""))
            if math.isfinite(value):
                last_seconds = value
        except (TypeError, ValueError):
            pass
    return {
        "epochs_completed": len(rows),
        "mean_epoch_seconds": sum(seconds) / len(seconds) if seconds else None,
        "last_val_macro_f1": last_macro,
        "last_epoch_seconds": last_seconds,
        "metrics_path": str(used) if used else None,
    }


def run_record(run_root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    status_payload = read_json(run_root / "status.json")
    stats = epoch_stats(run_root)
    status = str(status_payload.get("status", "missing"))
    epochs = int(stats["epochs_completed"] or 0)
    return {
        **identity,
        "status": status,
        "epochs_completed": epochs,
        "expected_epochs": EXPECTED_EPOCHS,
        "fraction": round(min(epochs, EXPECTED_EPOCHS) / EXPECTED_EPOCHS, 6),
        "mean_epoch_seconds": stats["mean_epoch_seconds"],
        "last_val_macro_f1": stats["last_val_macro_f1"],
        "last_epoch_seconds": stats["last_epoch_seconds"],
        "run_root": str(run_root),
        "metrics_path": stats["metrics_path"],
        "updated_at": status_payload.get("updated_at"),
    }


def batch_root(root: Path, dataset: str, batch_size: int) -> Path:
    return root / "batch" / f"batch-{batch_size:03d}" / dataset / "runs" / f"{dataset}__unit_interval__all_raw__seed-42"


def activation_root(root: Path, dataset: str, activation: str) -> Path:
    return root / "activations" / dataset / activation / dataset / "runs" / f"{dataset}__unit_interval__all_raw__seed-42"


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def source(label: str, files: list[str], definitions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "label": label,
        "files": files,
        "filters": [
            "Suíte controlled-augmentation05-batch-activation",
            "seed 42, normalização unit_interval, 100 épocas esperadas",
            "Quantização ignorada explicitamente por --skip-quantization",
        ],
        "metricDefinitions": definitions,
        "evidenceFlow": [
            {"title": "Leitura", "detail": "Leitura dos status.json e dos históricos checkpoints/epoch_metrics.csv disponíveis no output local."},
            {"title": "Agregação", "detail": "Contagem de execuções por status e cálculo de frações a partir de épocas concluídas; nenhuma época ausente foi convertida em zero."},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reviewed snapshot for the live training progress report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    now = datetime.now(timezone.utc)

    batch_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for batch_size in BATCH_SIZES:
            batch_rows.append(
                run_record(
                    batch_root(root, dataset, batch_size),
                    {"dataset": dataset, "dataset_label": DATASET_LABELS[dataset], "batch_size": batch_size},
                )
            )

    activation_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for activation in ACTIVATIONS:
            activation_rows.append(
                run_record(
                    activation_root(root, dataset, activation),
                    {"dataset": dataset, "dataset_label": DATASET_LABELS[dataset], "activation": activation},
                )
            )

    for row in [*batch_rows, *activation_rows]:
        classify_status(row, now)

    batch_completed = sum(row["status"] == "completed" for row in batch_rows)
    activation_completed = sum(row["status"] == "completed" for row in activation_rows)
    running = [row for row in activation_rows if row["status"] == "running"]
    stale = [row for row in activation_rows if row["status"] == "stale"]
    failed = [row for row in [*batch_rows, *activation_rows] if row["status"] == "failed"]
    current = running[0] if running else None
    training_state = "interrupted" if stale and not running else ("running" if running else "not_started")
    title_prefix = {
        "interrupted": "O treinamento está interrompido",
        "running": "O treinamento está em andamento",
        "not_started": "O treinamento aguarda início",
    }[training_state]

    # Estimate missing-run duration from the strongest local evidence: the
    # same dataset's completed activation runs, then its batch runs, then the
    # global average. This is explicitly a scenario, not an observed metric.
    dataset_rates: dict[str, float] = {}
    for dataset in DATASETS:
        values = [
            row["mean_epoch_seconds"]
            for row in activation_rows
            if row["dataset"] == dataset and row["status"] == "completed" and row["mean_epoch_seconds"]
        ]
        if not values:
            values = [
                row["mean_epoch_seconds"]
                for row in batch_rows
                if row["dataset"] == dataset and row["status"] == "completed" and row["mean_epoch_seconds"]
            ]
        if values:
            dataset_rates[dataset] = sum(values) / len(values)
    all_rates = [row for row in [*activation_rows, *batch_rows] if row["status"] == "completed" and row["mean_epoch_seconds"]]
    global_rate = sum(row["mean_epoch_seconds"] for row in all_rates) / len(all_rates) if all_rates else 60.0

    remaining_rows: list[dict[str, Any]] = []
    remaining_seconds = 0.0
    for row in activation_rows:
        if row["status"] == "completed":
            continue
        rate = row["mean_epoch_seconds"] or dataset_rates.get(row["dataset"], global_rate)
        remaining_epochs = max(EXPECTED_EPOCHS - int(row["epochs_completed"]), 0)
        # A small fixed allowance covers evaluation, LiteRT postprocess, and
        # process startup. It is not folded into the epoch-rate measurements.
        overhead = 900.0
        estimate = remaining_epochs * float(rate) + overhead
        remaining_seconds += estimate
        remaining_rows.append(
            {
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "activation": row["activation"],
                "status": row["status"],
                "epochs_completed": row["epochs_completed"],
                "remaining_epochs": remaining_epochs,
                "estimated_epoch_seconds": round(float(rate), 3),
                "estimated_remaining_hours": round(estimate / 3600.0, 3),
                "estimate_type": "observed_rate_extrapolation",
            }
        )

    # Use a 15% band for run-to-run variation and postprocessing overhead.
    low_hours = remaining_seconds / 3600.0 * 0.85
    high_hours = remaining_seconds / 3600.0 * 1.15
    eta_low = now + timedelta(hours=low_hours)
    eta_high = now + timedelta(hours=high_hours)

    summary = {
        "as_of": now.isoformat(),
        "batch_completed": batch_completed,
        "batch_planned": len(batch_rows),
        "activation_completed": activation_completed,
        "activation_planned": len(activation_rows),
        "execution_completed": batch_completed + activation_completed,
        "execution_planned": len(batch_rows) + len(activation_rows),
        "execution_fraction": round((batch_completed + activation_completed) / (len(batch_rows) + len(activation_rows)), 6),
        "active_run_count": len(running),
        "stale_run_count": len(stale),
        "failed_run_count": len(failed),
        "training_state": training_state,
        "stale_after_hours": STALE_RUNNING_AFTER_HOURS,
        "quantization_status": "skipped",
        "pipeline_status_file": str(root / "pipeline-status.json"),
        "pipeline_status": read_json(root / "pipeline-status.json").get("status", "stale_or_missing"),
        "current_dataset": current["dataset"] if current else None,
        "current_activation": current["activation"] if current else None,
        "current_epochs": current["epochs_completed"] if current else None,
        "current_expected_epochs": EXPECTED_EPOCHS if current else None,
        "current_fraction": current["fraction"] if current else None,
        "current_mean_epoch_seconds": current["mean_epoch_seconds"] if current else None,
        "current_last_val_macro_f1": current["last_val_macro_f1"] if current else None,
        "remaining_execution_count": len(remaining_rows),
        "remaining_hours_low": round(low_hours, 2),
        "remaining_hours_high": round(high_hours, 2),
        "eta_low": eta_low.isoformat(),
        "eta_high": eta_high.isoformat(),
        "estimate_note": "Faixa baseada em extrapolação dos tempos de época observados, com 15% de variação e 15 min de overhead por execução; não é garantia de prazo.",
    }

    datasets: list[dict[str, Any]] = []
    for dataset in DATASETS:
        rows = [row for row in activation_rows if row["dataset"] == dataset]
        datasets.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "completed": sum(row["status"] == "completed" for row in rows),
                "running": sum(row["status"] == "running" for row in rows),
                "stale": sum(row["status"] == "stale" for row in rows),
                "pending": sum(row["status"] in {"missing", "pending"} for row in rows),
                "failed": sum(row["status"] == "failed" for row in rows),
                "planned": len(rows),
            }
        )

    files = [
        str(root / "pipeline-status.json"),
        *[row["run_root"] + "\\status.json" for row in batch_rows if row["status"] != "missing"],
        *[row["run_root"] + "\\status.json" for row in activation_rows if row["status"] != "missing"],
        *[row["metrics_path"] for row in [*batch_rows, *activation_rows] if row.get("metrics_path")],
    ]
    definitions = [
        {"label": "Execuções concluídas", "definition": "Quantidade de runs com status.json igual a completed.", "componentIds": ["progress-summary", "progress-by-stage", "progress-dataset-table"]},
        {"label": "Épocas concluídas", "definition": "Número de linhas registradas em checkpoints/epoch_metrics.csv (ou logs/epoch_metrics.csv no fallback).", "componentIds": ["progress-summary", "current-run"]},
        {"label": "Macro-F1 de validação", "definition": "Valor val_macro_f1 da última época registrada; para runs concluídos, é o valor da última época, não necessariamente o melhor checkpoint.", "componentIds": ["current-run", "activation-table"]},
        {"label": "Previsão de retomada", "definition": "Cenário condicionado à retomada imediata: extrapolação das épocas restantes pelos tempos médios observados, acrescida de overhead fixo por execução e faixa de 15%.", "componentIds": ["forecast-summary", "remaining-table"]},
    ]

    payload = {
        "surface": "report",
        "title": f"{title_prefix}: {batch_completed + activation_completed} de {len(batch_rows) + len(activation_rows)} execuções concluídas",
        "generatedAt": now.isoformat(),
        "asOf": now.date().isoformat(),
        "status": "ready",
        "buildStatus": "creating",
        "queries": {
            "summary": {"rows": [summary], "source": source("Resumo vivo do pipeline", files, definitions)},
            "batch_status": {"rows": batch_rows, "source": source("Status dos 36 runs de batch", files, definitions)},
            "activation_status": {"rows": activation_rows, "source": source("Status das 27 ativações", files, definitions)},
            "dataset_summary": {"rows": datasets, "source": source("Resumo por dataset", files, definitions)},
            "remaining": {"rows": remaining_rows, "source": source("Execuções ainda não concluídas", files, definitions)},
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=clean) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False, indent=2, default=clean))


if __name__ == "__main__":
    main()
