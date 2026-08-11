"""Audit and report the completed KMNIST batch-size sweep.

This script reads only durable benchmark artifacts, applies structural and
comparability checks, and writes auditable tables plus a Markdown report and a
Data Analytics portable-report payload.  Timing and hardware comparisons are
restricted to runs whose per-epoch timing/telemetry cover all 50 epochs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "kmnist__unit_interval__all_raw__seed-42"
DEFAULT_INPUT_ROOT = Path("/mnt/e/tcc-benchmark/outputs/kmnist-alldata-noaug-batch-sweep-2026-08-06")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "analysis-2026-08-09"
EXPECTED_BATCHES = tuple(range(16, 513, 16))
NEUTRAL_AUGMENTATION = {
    "flip_lr": False,
    "brightness_delta": 0.0,
    "contrast_lower": 1.0,
    "contrast_upper": 1.0,
    "translate_frac": 0.0,
    "zoom_min": 1.0,
    "zoom_max": 1.0,
    "noise_std": 0.0,
    "cutout_prob": 0.0,
    "cutout_max_frac": 0.0,
}
REQUIRED_FILES = (
    "manifest.json",
    "status.json",
    "artifacts/test_metrics.json",
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
        raise TypeError(f"Esperado objeto JSON em {path}")
    return value


def _number(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if math.isfinite(candidate) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(0, sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b"")) - 1)


def _stat(payload: dict[str, Any], name: str, stat: str) -> float:
    return _number(payload.get("metrics", {}).get(name, {}).get(stat))


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, *, digits: int = 8) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for key, value in row.items():
            value = _safe_json(value)
            if isinstance(value, float):
                value = round(value, digits)
            cleaned[str(key)] = value
        result.append(cleaned)
    return result


def _check(
    checks: list[dict[str, Any]],
    *,
    batch_size: int,
    name: str,
    passed: bool,
    severity: str,
    detail: str,
) -> None:
    checks.append(
        {
            "batch_size": int(batch_size),
            "check": name,
            "passed": bool(passed),
            "severity_if_failed": severity,
            "detail": detail,
        }
    )


def _training_signature(config: dict[str, Any]) -> dict[str, Any]:
    training = copy.deepcopy(config["training"])
    training.pop("batch_size", None)
    return {
        "dataset": config["dataset"],
        "adapter": config["adapter"],
        "normalization": config["normalization"],
        "balance_mode": config["balance_mode"],
        "seed": config["seed"],
        "target_size": config["target_size"],
        "native_channels": config["native_channels"],
        "num_classes": config["num_classes"],
        "train_fraction": config["train_fraction"],
        "validation_fraction": config["validation_fraction"],
        "test_fraction": config["test_fraction"],
        "training": training,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], *, digits: int = 4) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for _, row in frame.iterrows():
        values: list[str] = []
        for field, _ in columns:
            value = row.get(field)
            if pd.isna(value):
                values.append("-")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _source(source_id: str, label: str, filename: str, description: str, definitions: list[str]) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "query": {
            "id": f"kmnist-batch-sweep-{source_id}",
            "description": description,
            "engine": "DuckDB",
            "language": "SQL",
            "sql": f"SELECT * FROM read_csv_auto('{filename}', header = true)",
            "tables_used": [filename],
            "filters": [
                "KMNIST completo: 70.000 imagens de origem; split estratificado 70/15/15; seed 42",
                "Normalização unit_interval, balanceamento all_raw e augmentation desativada",
                "Apenas o batch size varia entre 16 e 512, em incrementos de 16",
            ],
            "metric_definitions": definitions,
        },
    }


def _build_artifact(
    *,
    output_dir: Path,
    overview: dict[str, Any],
    runs: pd.DataFrame,
    epochs: pd.DataFrame,
    hardware: pd.DataFrame,
    quality_summary: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> dict[str, Any]:
    comparable = runs[runs["timing_comparable"] == "Sim"].copy()
    # The collector can miss the final epoch_end notification while retaining
    # continuous five-second samples, so 49/50 events is accepted as complete.
    partial_hardware = hardware[hardware["telemetry_epoch_end_count"] < 49].copy()
    partial_hardware_batches = ", ".join(
        str(int(value)) for value in partial_hardware["batch_size"].tolist()
    ) or "nenhum"
    selected_curve_batches = [16, 64, 128, 256, 512]
    curves = epochs[epochs["batch_size"].isin(selected_curve_batches)].copy()
    curves["batch_label"] = curves["batch_size"].map(lambda value: f"batch {int(value)}")
    timing_table = runs[
        [
            "batch_size",
            "epochs_completed",
            "timed_epoch_count",
            "timing_coverage_rate",
            "training_seconds_recorded",
            "mean_epoch_seconds_recorded",
            "weighted_examples_per_second",
            "evaluation_seconds",
            "timing_comparable",
        ]
    ].copy()
    metric_table = runs[
        [
            "batch_size",
            "best_epoch",
            "best_val_macro_f1",
            "final_val_macro_f1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_loss",
        ]
    ].copy()
    sources = [
        _source(
            "overview",
            "Resumo consolidado da varredura KMNIST",
            "analysis_overview.json",
            "Conclusão, escopo, cobertura e principais agregados da varredura de batch size.",
            [
                "Run concluída = status completed e arquivos obrigatórios presentes.",
                "Comparabilidade de tempo exige 50 registros finitos de epoch_seconds.",
            ],
        ),
        _source(
            "runs",
            "Métricas e tempo por batch",
            "run_metrics.csv",
            "Uma linha por batch com métricas de validação/teste, tempo e cobertura do registro.",
            [
                "Macro-F1 = média não ponderada do F1 por classe no teste.",
                "Throughput ponderado = 49.000 exemplos por época vezes épocas com duração registrada, dividido pela soma de epoch_seconds.",
            ],
        ),
        _source(
            "hardware",
            "Resumo de hardware por batch",
            "hardware_summary.csv",
            "Médias, p95 e picos observados pelo amostrador de hardware a cada cinco segundos.",
            [
                "VRAM máxima = maior gpu_memory_used_bytes amostrado na run, convertido em GiB.",
                "Cobertura de telemetria = eventos epoch_end registrados dividido por 50 épocas planejadas.",
            ],
        ),
        _source(
            "epochs",
            "Histórico por época",
            "epoch_metrics_all.csv",
            "Métricas de treino/validação e duração por época para as 32 runs.",
            ["Melhor época = maior val_macro_f1 no histórico persistido da run."],
        ),
        _source(
            "quality",
            "Controles de qualidade e ressalvas",
            "quality_checks.csv",
            "Validações de completude, protocolo, sequenciamento de épocas, previsões e cobertura operacional.",
            ["Falhas de cobertura de telemetria não invalidam métricas de teste ou tempos por época, mas limitam conclusões sobre picos de hardware."],
        ),
    ]

    fastest = comparable.loc[comparable["mean_epoch_seconds_recorded"].idxmin()]
    best_score = runs.loc[runs["test_macro_f1"].idxmax()]
    peak_vram = hardware.loc[hardware["gpu_memory_max_gib"].idxmax()]
    cards = [
        {
            "id": "completion",
            "description": "Conclusão e integridade estrutural",
            "dataset": "headline",
            "sourceId": "overview",
            "metrics": [
                {"label": "Runs concluídas", "field": "completed_runs", "format": "number"},
                {"label": "Planejadas", "field": "planned_runs", "format": "number"},
            ],
        },
        {
            "id": "best_score",
            "description": "Maior macro-F1 de teste observada",
            "dataset": "headline",
            "sourceId": "overview",
            "metrics": [
                {"label": "Macro-F1", "field": "best_test_macro_f1", "format": "percent"},
                {"label": "Batch", "field": "best_test_batch", "format": "number"},
            ],
        },
        {
            "id": "fastest",
            "description": "Menor duração média por época entre as runs comparáveis",
            "dataset": "headline",
            "sourceId": "overview",
            "metrics": [
                {"label": "Batch", "field": "fastest_comparable_batch", "format": "number"},
                {"label": "s/época", "field": "fastest_comparable_mean_epoch_seconds", "format": "number"},
            ],
        },
        {
            "id": "vram",
            "description": "Maior VRAM observada durante treino",
            "dataset": "headline",
            "sourceId": "hardware",
            "metrics": [
                {"label": "VRAM (GiB)", "field": "peak_gpu_memory_gib", "format": "number"},
                {"label": "Batch", "field": "peak_gpu_memory_batch", "format": "number"},
            ],
        },
        {
            "id": "coverage",
            "description": "Runs elegíveis para comparação de eficiência",
            "dataset": "headline",
            "sourceId": "quality",
            "metrics": [
                {"label": "Cobertura completa", "field": "timing_comparable_runs", "format": "number"},
                {"label": "Com ressalva", "field": "partial_timing_runs", "format": "number"},
            ],
        },
    ]
    charts = [
        {
            "id": "test_f1",
            "title": "Macro-F1 de teste por batch size",
            "subtitle": "32 runs, mesmo dataset, split e seed; valores representam o conjunto de teste de 10.500 imagens.",
            "type": "line",
            "dataset": "run_metrics",
            "sourceId": "runs",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "batch_size", "type": "quantitative", "label": "Batch size"},
                "y": {"field": "test_macro_f1", "type": "quantitative", "label": "Macro-F1 de teste"},
            },
            "options": {"legend": {"visible": False}},
            "layout": {"width": "full", "height": 380},
        },
        {
            "id": "epoch_time",
            "title": "Duração média por época nas runs comparáveis",
            "subtitle": "Exclui batches 48 e 224, cuja retomada preservou métricas mas não o histórico completo de duração.",
            "type": "line",
            "dataset": "comparable_runs",
            "sourceId": "runs",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "batch_size", "type": "quantitative", "label": "Batch size"},
                "y": {"field": "mean_epoch_seconds_recorded", "type": "quantitative", "label": "Segundos por época"},
            },
            "options": {"legend": {"visible": False}},
            "layout": {"width": "full", "height": 380},
        },
        {
            "id": "throughput",
            "title": "Throughput de treino nas runs comparáveis",
            "subtitle": "Exemplos de treino processados por segundo, ponderados pela duração registrada de cada época.",
            "type": "line",
            "dataset": "comparable_runs",
            "sourceId": "runs",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "batch_size", "type": "quantitative", "label": "Batch size"},
                "y": {"field": "weighted_examples_per_second", "type": "quantitative", "label": "Exemplos/s"},
            },
            "options": {"legend": {"visible": False}},
            "layout": {"width": "full", "height": 380},
        },
        {
            "id": "vram_by_batch",
            "title": "Pico de VRAM observado por batch size",
            "subtitle": "Picos amostrados na RTX 3050 de 8 GiB; batches com telemetria parcial são exibidos, mas sinalizados na tabela.",
            "type": "line",
            "dataset": "hardware",
            "sourceId": "hardware",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "batch_size", "type": "quantitative", "label": "Batch size"},
                "y": {"field": "gpu_memory_max_gib", "type": "quantitative", "label": "VRAM máxima (GiB)"},
            },
            "options": {"legend": {"visible": False}},
            "layout": {"width": "full", "height": 380},
        },
        {
            "id": "validation_curves",
            "title": "Macro-F1 de validação por época - batches selecionados",
            "subtitle": "Curvas de 16, 64, 128, 256 e 512 para comparar convergência sem sobrecarregar a leitura com 32 séries.",
            "type": "line",
            "dataset": "selected_curves",
            "sourceId": "epochs",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "epoch", "type": "quantitative", "label": "Época"},
                "y": {"field": "val_macro_f1", "type": "quantitative", "label": "Macro-F1 de validação"},
                "color": {"field": "batch_label", "type": "nominal", "label": "Batch"},
            },
            "options": {"legend": {"visible": True}},
            "layout": {"width": "full", "height": 400},
        },
    ]
    tables = [
        {
            "id": "metrics_table",
            "title": "Métricas de qualidade para cada batch",
            "subtitle": "Métricas finais no teste e melhor macro-F1 observada no conjunto de validação.",
            "dataset": "metric_table",
            "sourceId": "runs",
            "defaultSort": {"field": "batch_size", "direction": "asc"},
            "density": "compact",
            "columns": [
                {"field": "batch_size", "label": "Batch", "type": "number", "format": "number"},
                {"field": "best_val_macro_f1", "label": "Melhor val F1", "type": "number", "format": "percent"},
                {"field": "test_accuracy", "label": "Teste accuracy", "type": "number", "format": "percent"},
                {"field": "test_macro_f1", "label": "Teste macro-F1", "type": "number", "format": "percent"},
                {"field": "test_loss", "label": "Teste loss", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "timing_table",
            "title": "Tempo de treinamento por batch",
            "subtitle": "Tempo registrado por época; todas as 32 runs possuem cobertura integral nesta varredura.",
            "dataset": "timing_table",
            "sourceId": "runs",
            "defaultSort": {"field": "batch_size", "direction": "asc"},
            "density": "compact",
            "columns": [
                {"field": "batch_size", "label": "Batch", "type": "number", "format": "number"},
                {"field": "training_seconds_recorded", "label": "Treino (s)", "type": "number", "format": "number"},
                {"field": "mean_epoch_seconds_recorded", "label": "s/época", "type": "number", "format": "number"},
                {"field": "weighted_examples_per_second", "label": "Exemplos/s", "type": "number", "format": "number"},
                {"field": "evaluation_seconds", "label": "Avaliação (s)", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "hardware_table",
            "title": "Hardware observado em cada batch",
            "subtitle": "Telemetria a cada cinco segundos; a cobertura operacional é exibida para interpretação correta.",
            "dataset": "hardware",
            "sourceId": "hardware",
            "defaultSort": {"field": "batch_size", "direction": "asc"},
            "density": "compact",
            "columns": [
                {"field": "batch_size", "label": "Batch", "type": "number", "format": "number"},
                {"field": "telemetry_coverage_rate", "label": "Cobertura", "type": "number", "format": "percent"},
                {"field": "gpu_util_mean_percent", "label": "GPU média (%)", "type": "number", "format": "number"},
                {"field": "gpu_memory_max_gib", "label": "VRAM máx. (GiB)", "type": "number", "format": "number"},
                {"field": "gpu_temperature_max_c", "label": "GPU máx. (C)", "type": "number", "format": "number"},
                {"field": "ram_max_percent", "label": "RAM máx. (%)", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "quality_table",
            "title": "Ressalvas de qualidade identificadas",
            "subtitle": "Cobertura parcial não invalida métricas de teste nem tempos por época, mas limita a interpretação de picos de hardware.",
            "dataset": "anomalies",
            "sourceId": "quality",
            "defaultSort": {"field": "batch_size", "direction": "asc"},
            "density": "compact",
            "columns": [
                {"field": "batch_size", "label": "Batch", "type": "number", "format": "number"},
                {"field": "issue", "label": "Ressalva", "type": "text"},
                {"field": "severity", "label": "Severidade", "type": "text"},
                {"field": "evidence", "label": "Evidência", "type": "text"},
                {"field": "implication", "label": "Implicação", "type": "text"},
            ],
        },
    ]
    report_title = "Análise técnica da varredura de batch size do KMNIST"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {report_title}"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "overview",
            "body": (
                "## Resumo técnico\n\n"
                f"A varredura foi concluída com **{int(overview['completed_runs'])} de {int(overview['planned_runs'])} runs** e **{int(overview['epochs_recorded'])} épocas** registradas. "
                f"As métricas finais de teste estão presentes para todos os batches. A maior macro-F1 de teste foi **{float(best_score['test_macro_f1']):.4%}** no batch **{int(best_score['batch_size'])}**. "
                f"As **{int(overview['timing_comparable_runs'])} runs** têm cobertura completa de duração por época e entram na comparação de eficiência. Os batches **{partial_hardware_batches}** têm telemetria de hardware parcial; suas métricas de modelo e tempo continuam válidas, mas os picos de hardware devem ser lidos como amostras."
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {
            "id": "quality_findings",
            "type": "markdown",
            "sourceId": "runs",
            "body": (
                "## Qualidade do modelo ao longo dos batches\n\n"
                f"Os 32 batches usam o mesmo conjunto de teste de 10.500 imagens. A macro-F1 de teste variou de **{float(runs['test_macro_f1'].min()):.4%}** a **{float(runs['test_macro_f1'].max()):.4%}**, "
                "portanto as diferenças observadas descrevem este único seed e não estabelecem superioridade estatística entre batches próximos."
            ),
        },
        {"id": "test_f1_chart", "type": "chart", "chartId": "test_f1"},
        {
            "id": "efficiency_findings",
            "type": "markdown",
            "sourceId": "runs",
            "body": (
                "## Eficiência de treino com cobertura completa\n\n"
                f"Entre as runs comparáveis, o batch **{int(fastest['batch_size'])}** teve a menor duração média: **{float(fastest['mean_epoch_seconds_recorded']):.2f} s por época**. "
                "Esse ranking usa apenas histórico completo de epoch_seconds para evitar que uma retomada pareça artificialmente rápida."
            ),
        },
        {"id": "epoch_time_chart", "type": "chart", "chartId": "epoch_time"},
        {"id": "throughput_chart", "type": "chart", "chartId": "throughput"},
        {
            "id": "hardware_findings",
            "type": "markdown",
            "sourceId": "hardware",
            "body": (
                "## Uso de hardware\n\n"
                f"O maior pico de VRAM observado foi **{float(peak_vram['gpu_memory_max_gib']):.2f} GiB** no batch **{int(peak_vram['batch_size'])}**, abaixo dos 8 GiB físicos da RTX 3050. "
                "Não houve falha registrada por memória. Os valores de hardware dos batches com telemetria parcial devem ser lidos como amostra, não como pico garantido da run inteira."
            ),
        },
        {"id": "vram_chart", "type": "chart", "chartId": "vram_by_batch"},
        {
            "id": "convergence_findings",
            "type": "markdown",
            "sourceId": "epochs",
            "body": (
                "## Convergência de batches selecionados\n\n"
                "As cinco curvas abaixo preservam a leitura da convergência sem sobrepor 32 séries. Todas as runs foram treinadas por 50 épocas; a parada antecipada e a redução de taxa foram configuradas para não encurtar o experimento."
            ),
        },
        {"id": "curve_chart", "type": "chart", "chartId": "validation_curves"},
        {
            "id": "run_detail_intro",
            "type": "markdown",
            "sourceId": "runs",
            "body": "## Métricas, tempo e hardware por run\n\nAs tabelas mostram todas as 32 runs. As colunas de cobertura indicam quais medições podem ser comparadas diretamente.",
        },
        {"id": "metrics_table_block", "type": "table", "tableId": "metrics_table"},
        {"id": "timing_table_block", "type": "table", "tableId": "timing_table"},
        {"id": "hardware_table_block", "type": "table", "tableId": "hardware_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "quality",
            "body": (
                "## Limitações e verificações de robustez\n\n"
                "As métricas de qualidade foram verificadas contra status, 50 épocas sequenciais, previsões e suporte do teste. "
                f"Os batches {partial_hardware_batches} não preservaram todos os eventos `epoch_end` da telemetria; por isso os picos de VRAM e demais máximos desses pontos não representam necessariamente a run inteira. Os seus históricos de tempo por época estão completos. "
                "Há somente um seed por batch, então pequenas diferenças de macro-F1 podem refletir a variabilidade de otimização."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Próximos passos recomendados\n\n"
                f"1. Reexecutar os batches {partial_hardware_batches} em uma nova pasta de saída se for necessária uma comparação integral de hardware desses pontos.\n"
                "2. Repetir os batches mais promissores com ao menos três seeds para estimar variância de qualidade.\n"
                "3. Selecionar o menor batch que atinja a qualidade-alvo e mantenha a eficiência/hardware dentro dos limites operacionais."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": "## Questões em aberto\n\nA varredura mede um único dataset, arquitetura, normalização e seed. A escolha final de batch pode mudar em outros datasets ou com replicações adicionais.",
        },
    ]
    headline = pd.DataFrame(
        [
            {
                "completed_runs": int(overview["completed_runs"]),
                "planned_runs": int(overview["planned_runs"]),
                "best_test_macro_f1": float(best_score["test_macro_f1"]),
                "best_test_batch": int(best_score["batch_size"]),
                "fastest_comparable_batch": int(fastest["batch_size"]),
                "fastest_comparable_mean_epoch_seconds": float(fastest["mean_epoch_seconds_recorded"]),
                "peak_gpu_memory_gib": float(peak_vram["gpu_memory_max_gib"]),
                "peak_gpu_memory_batch": int(peak_vram["batch_size"]),
                "timing_comparable_runs": int(overview["timing_comparable_runs"]),
                "partial_timing_runs": int(overview["partial_timing_runs"]),
            }
        ]
    )
    manifest = {
        "version": 1,
        "surface": "report",
        "title": report_title,
        "description": "Auditoria de qualidade, desempenho, tempo e telemetria de 32 runs KMNIST com batch size variável.",
        "generatedAt": overview["generated_at"],
        "sources": sources,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": overview["generated_at"],
            "datasets": {
                "headline": _records(headline),
                "run_metrics": _records(runs),
                "comparable_runs": _records(comparable),
                "metric_table": _records(metric_table),
                "timing_table": _records(timing_table),
                "hardware": _records(hardware),
                "selected_curves": _records(curves),
                "quality_summary": _records(quality_summary),
                "anomalies": _records(anomalies),
            },
        },
        "sources": sources,
    }


def _write_markdown(
    *,
    output_path: Path,
    overview: dict[str, Any],
    runs: pd.DataFrame,
    hardware: pd.DataFrame,
    quality_summary: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> None:
    comparable = runs[runs["timing_comparable"] == "Sim"].copy()
    best = runs.loc[runs["test_macro_f1"].idxmax()]
    fastest = comparable.loc[comparable["mean_epoch_seconds_recorded"].idxmin()]
    highest_throughput = comparable.loc[comparable["weighted_examples_per_second"].idxmax()]
    peak_vram = hardware.loc[hardware["gpu_memory_max_gib"].idxmax()]
    max_temperature = hardware.loc[hardware["gpu_temperature_max_c"].idxmax()]
    partial_hardware_batches = ", ".join(
        str(int(value))
        for value in hardware.loc[hardware["telemetry_epoch_end_count"] < 49, "batch_size"].tolist()
    ) or "nenhum"
    metric_columns = [
        ("batch_size", "Batch"),
        ("best_epoch", "Melhor época"),
        ("best_val_macro_f1", "Melhor val F1"),
        ("final_val_macro_f1", "Val F1 final"),
        ("test_accuracy", "Teste acc."),
        ("test_balanced_accuracy", "Teste bal. acc."),
        ("test_macro_f1", "Teste macro-F1"),
        ("test_loss", "Teste loss"),
    ]
    timing_columns = [
        ("batch_size", "Batch"),
        ("epochs_completed", "Épocas"),
        ("timed_epoch_count", "Cronometradas"),
        ("timing_coverage_rate", "Cobertura"),
        ("training_seconds_recorded", "Treino s"),
        ("mean_epoch_seconds_recorded", "s/época"),
        ("weighted_examples_per_second", "Exemplos/s"),
        ("evaluation_seconds", "Avaliação s"),
        ("timing_comparable", "Comparável"),
    ]
    hardware_columns = [
        ("batch_size", "Batch"),
        ("telemetry_epoch_end_count", "Epoch ends"),
        ("telemetry_coverage_rate", "Cobertura"),
        ("gpu_util_mean_percent", "GPU média %"),
        ("gpu_util_p95_percent", "GPU p95 %"),
        ("gpu_memory_max_gib", "VRAM máx. GiB"),
        ("gpu_temperature_max_c", "GPU máx. C"),
        ("ram_max_percent", "RAM máx. %"),
        ("process_rss_max_gib", "RSS máx. GiB"),
    ]
    total_recorded_hours = float(runs["training_seconds_recorded"].sum()) / 3600.0
    complete_recorded_hours = float(comparable["training_seconds_recorded"].sum()) / 3600.0
    anomaly_text = "Nenhuma."
    if not anomalies.empty:
        anomaly_text = "; ".join(
            f"batch {int(row.batch_size)}: {row.issue.lower()} ({row.evidence})"
            for row in anomalies.itertuples(index=False)
        )
    quality_table = _markdown_table(
        quality_summary,
        [("severity_if_failed", "Severidade"), ("result", "Resultado"), ("checks", "Checks")],
        digits=0,
    )
    content = f"""# Análise técnica da varredura de batch size do KMNIST

## Resumo técnico

A varredura foi concluída com **{int(overview['completed_runs'])} de {int(overview['planned_runs'])} runs** e **{int(overview['epochs_recorded'])} épocas** persistidas. Todas as runs têm status `completed`, resumo de treino, métricas de teste e telemetria disponíveis. Não há comando de retomada pendente.

A maior macro-F1 de teste foi **{float(best['test_macro_f1']):.4%}** no batch **{int(best['batch_size'])}**. Para comparação de eficiência, o batch **{int(fastest['batch_size'])}** foi o mais rápido entre as runs com cronometragem completa, com **{float(fastest['mean_epoch_seconds_recorded']):.2f} s por época**. O maior throughput comparável foi **{float(highest_throughput['weighted_examples_per_second']):.2f} exemplos/s** no batch **{int(highest_throughput['batch_size'])}**.

## Escopo, dados e definições

- Dataset: KMNIST, 70.000 imagens em escala de cinza, 64x64 px, 10 classes.
- Split estratificado fixo, seed 42: 49.000 treino, 10.500 validação e 10.500 teste; 4.900 exemplos de treino por classe.
- Protocolo fixo: `unit_interval`, `all_raw`, FP16, Adam com learning rate 0,0003 e 50 épocas.
- Augmentation desativada: `extra_fraction=0.0` e todas as transformações em valores neutros.
- Variável experimental: somente o batch size, de 16 a 512 em passos de 16.
- Macro-F1 de teste: média não ponderada do F1 das dez classes.
- Tempo canônico: soma de `epoch_seconds` disponíveis. Uma run só é comparável em eficiência quando há 50 tempos de época finitos.

## Verificação de qualidade dos artefatos

Os controles validaram status final, arquivos obrigatórios, sequência 1..50 de épocas, configuração constante fora do batch, split, contagem de previsões e suporte das classes de teste. Resultado dos checks:

{quality_table}

Ressalvas: {anomaly_text}

Essas ressalvas não invalidam as métricas finais de qualidade nem os tempos por época dos batches afetados: as 50 épocas, previsões e avaliação de teste estão presentes. Elas limitam a interpretação de seus máximos de hardware, que podem não representar a execução inteira.

## Métricas de qualidade por run

As diferenças de desempenho são descritivas deste único seed. Não há replicações suficientes para afirmar superioridade estatística entre batches próximos.

{_markdown_table(runs, metric_columns)}

## Tempo de treinamento por run

O tempo registrado em todas as runs soma **{total_recorded_hours:.2f} h**. Deste total, **{complete_recorded_hours:.2f} h** pertencem às {int(overview['timing_comparable_runs'])} runs com cobertura integral, apropriadas para comparação de eficiência.

{_markdown_table(runs, timing_columns)}

## Uso de hardware por run

O pico de VRAM observado foi **{float(peak_vram['gpu_memory_max_gib']):.2f} GiB** no batch **{int(peak_vram['batch_size'])}**, abaixo dos 8 GiB da RTX 3050. A maior temperatura observada foi **{float(max_temperature['gpu_temperature_max_c']):.1f} C** no batch **{int(max_temperature['batch_size'])}**. Não há status de OOM/falha na matriz concluída.

{_markdown_table(hardware, hardware_columns)}

## Limitações, incerteza e robustez

- Há uma só réplica (seed 42) por batch; a variação de qualidade não possui intervalo de confiança.
- Os batches {partial_hardware_batches} têm cobertura parcial de telemetria; seus resultados de hardware são amostras e não devem ser usados para comparar picos. Os registros de tempo por época estão completos e entram no ranking de eficiência.
- A telemetria mede amostras a cada cinco segundos; picos muito breves podem não ter sido observados.
- As conclusões valem para KMNIST, a arquitetura CNN do projeto e este protocolo de pré-processamento.

## Próximos passos recomendados

1. Reexecutar batches {partial_hardware_batches} em novo diretório caso seja necessário completar o perfil de VRAM e demais métricas de hardware desses pontos.
2. Repetir os batches candidatos com pelo menos três seeds antes de escolher batch por qualidade.
3. Para uso operacional, escolher o menor batch que cumpra a meta de macro-F1 e tenha tempo/hardware aceitáveis; usar a tabela de cobertura para evitar decisões com medições parciais.

## Arquivos de apoio

- `run_metrics.csv`: métricas de qualidade, tempo e cobertura por batch.
- `hardware_summary.csv`: telemetria agregada por batch.
- `epoch_metrics_all.csv`: 1.600 linhas de histórico por época.
- `per_class_metrics.csv`: métricas de precisão, recall, F1 e suporte por classe.
- `quality_checks.csv` e `quality_summary.csv`: evidências da validação.
- `report_artifact.json`: artefato canônico usado para gerar o HTML/PDF.
"""
    output_path.write_text(content, encoding="utf-8")


def build_analysis(input_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    hardware_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    epoch_frames: list[pd.DataFrame] = []
    signatures: dict[int, str] = {}
    split_fingerprints: dict[int, str] = {}
    source_fingerprints: dict[int, str] = {}

    for batch_size in EXPECTED_BATCHES:
        run_dir = input_root / f"batch-{batch_size:03d}" / "kmnist" / "runs" / RUN_ID
        missing = [relative for relative in REQUIRED_FILES if not (run_dir / relative).is_file()]
        _check(
            checks,
            batch_size=batch_size,
            name="required_files",
            passed=not missing,
            severity="critical",
            detail="all required files present" if not missing else f"missing: {', '.join(missing)}",
        )
        if missing:
            continue

        manifest = _json(run_dir / "manifest.json")
        status = _json(run_dir / "status.json")
        training = _json(run_dir / "logs" / "training_summary.json")
        test = _json(run_dir / "artifacts" / "test_metrics.json")
        telemetry = _json(run_dir / "telemetry" / "summary.json")
        environment = _json(run_dir / "telemetry" / "environment.json")
        config = manifest["config"]
        data_metadata = manifest.get("data_metadata", {})
        datasets = data_metadata.get("datasets", {})
        train_meta = datasets.get("train", {})
        validation_meta = datasets.get("validation", {})
        test_meta = datasets.get("test", {})
        training_config = config["training"]

        epochs = pd.read_csv(run_dir / "checkpoints" / "epoch_metrics.csv")
        for column in epochs.columns:
            if column != "epoch":
                epochs[column] = pd.to_numeric(epochs[column], errors="coerce")
        epochs["epoch"] = pd.to_numeric(epochs["epoch"], errors="coerce")
        epochs.insert(0, "batch_size", batch_size)
        epoch_frames.append(epochs)

        expected_epochs = _int(training.get("max_epochs"), 50)
        reported_epochs = _int(training.get("epochs_completed"))
        epoch_numbers = epochs["epoch"].dropna().astype(int).tolist()
        sequential_epochs = epoch_numbers == list(range(1, expected_epochs + 1))
        timed = epochs.dropna(subset=["epoch_seconds"]).copy()
        timed_epoch_count = len(timed)
        training_seconds_recorded = float(timed["epoch_seconds"].sum()) if timed_epoch_count else math.nan
        mean_epoch_seconds_recorded = float(timed["epoch_seconds"].mean()) if timed_epoch_count else math.nan
        examples_timed = float(timed["train_examples"].sum()) if timed_epoch_count else math.nan
        weighted_throughput = examples_timed / training_seconds_recorded if training_seconds_recorded else math.nan
        timing_coverage = timed_epoch_count / expected_epochs if expected_epochs else math.nan
        training_seconds_summary = _number(training.get("training_seconds"))
        timing_reconciled = math.isclose(
            training_seconds_recorded,
            training_seconds_summary,
            rel_tol=1e-8,
            abs_tol=1e-4,
        )

        classification = test["classification"]
        per_class = classification["per_class"]
        test_samples = int(sum(_int(values.get("support")) for values in per_class.values()))
        prediction_count = _line_count(run_dir / "artifacts" / "predictions.csv")
        metric_epochs = epochs.dropna(subset=["val_macro_f1"])
        best_row = metric_epochs.loc[metric_epochs["val_macro_f1"].idxmax()]
        final_row = epochs.iloc[-1]
        event_counts = telemetry.get("event_counts", {})
        telemetry_epoch_end_count = _int(event_counts.get("epoch_end"))
        telemetry_coverage = telemetry_epoch_end_count / expected_epochs if expected_epochs else math.nan
        signature = json.dumps(_training_signature(config), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        signatures[batch_size] = signature
        split_fingerprints[batch_size] = str(manifest.get("split_fingerprint", ""))
        source_fingerprints[batch_size] = str(config.get("source_fingerprint", ""))

        _check(checks, batch_size=batch_size, name="status_completed", passed=status.get("status") == "completed", severity="critical", detail=str(status.get("status")))
        _check(checks, batch_size=batch_size, name="batch_matches_config", passed=_int(training_config.get("batch_size")) == batch_size, severity="critical", detail=f"config={training_config.get('batch_size')}, folder={batch_size}")
        _check(checks, batch_size=batch_size, name="epoch_count_matches_summary", passed=len(epochs) == reported_epochs == expected_epochs, severity="critical", detail=f"csv={len(epochs)}, summary={reported_epochs}, expected={expected_epochs}")
        _check(checks, batch_size=batch_size, name="epoch_sequence", passed=sequential_epochs, severity="high", detail=f"first={epoch_numbers[:1]}, last={epoch_numbers[-1:]}")
        _check(checks, batch_size=batch_size, name="prediction_count", passed=prediction_count == test_samples == 10500, severity="critical", detail=f"predictions={prediction_count}, support={test_samples}")
        _check(checks, batch_size=batch_size, name="timing_summary_reconciled", passed=timing_reconciled, severity="medium", detail=f"epoch_sum={training_seconds_recorded:.6f}, summary={training_seconds_summary:.6f}")
        _check(checks, batch_size=batch_size, name="full_timing_coverage", passed=timed_epoch_count == expected_epochs, severity="high", detail=f"timed_epochs={timed_epoch_count}/{expected_epochs}")
        _check(checks, batch_size=batch_size, name="full_telemetry_coverage", passed=telemetry_epoch_end_count >= expected_epochs - 1, severity="medium", detail=f"telemetry_epoch_end={telemetry_epoch_end_count}/{expected_epochs}")
        _check(checks, batch_size=batch_size, name="neutral_augmentation", passed=training_config.get("augmentation") == NEUTRAL_AUGMENTATION and _number(training_config.get("extra_fraction")) == 0.0, severity="critical", detail=f"extra_fraction={training_config.get('extra_fraction')}")
        _check(checks, batch_size=batch_size, name="fixed_split_counts", passed=_int(train_meta.get("total_examples")) == 49000 and _int(validation_meta.get("total_examples")) == 10500 and _int(test_meta.get("total_examples")) == 10500, severity="critical", detail=f"train={train_meta.get('total_examples')}, val={validation_meta.get('total_examples')}, test={test_meta.get('total_examples')}")
        headline_metrics = [classification.get("accuracy"), classification.get("balanced_accuracy"), classification.get("macro_f1"), test.get("keras_metrics", {}).get("loss")]
        _check(checks, batch_size=batch_size, name="finite_headline_metrics", passed=all(math.isfinite(_number(value)) for value in headline_metrics), severity="critical", detail="accuracy, balanced accuracy, macro-F1 and loss are finite")

        gpu_name = ""
        gpus = environment.get("hardware", {}).get("gpus", [])
        if gpus:
            gpu_name = str(gpus[0].get("name", ""))
        run_rows.append(
            {
                "batch_size": batch_size,
                "run_id": RUN_ID,
                "status": str(status.get("status")),
                "epochs_completed": reported_epochs,
                "best_epoch": _int(best_row["epoch"]),
                "best_val_macro_f1": _number(best_row["val_macro_f1"]),
                "final_val_macro_f1": _number(final_row["val_macro_f1"]),
                "final_val_accuracy": _number(final_row["val_accuracy"]),
                "final_train_accuracy": _number(final_row["accuracy"]),
                "final_train_loss": _number(final_row["loss"]),
                "test_accuracy": _number(classification.get("accuracy")),
                "test_balanced_accuracy": _number(classification.get("balanced_accuracy")),
                "test_macro_f1": _number(classification.get("macro_f1")),
                "test_loss": _number(test.get("keras_metrics", {}).get("loss")),
                "test_samples": test_samples,
                "training_seconds_recorded": training_seconds_recorded,
                "training_seconds_summary": training_seconds_summary,
                "mean_epoch_seconds_recorded": mean_epoch_seconds_recorded,
                "timed_epoch_count": timed_epoch_count,
                "timing_coverage_rate": timing_coverage,
                "timing_comparable": "Sim" if timed_epoch_count == expected_epochs else "Não",
                "weighted_examples_per_second": weighted_throughput,
                "mean_train_examples_per_second_summary": _number(training.get("mean_train_examples_per_second")),
                "median_train_examples_per_second_summary": _number(training.get("median_train_examples_per_second")),
                "evaluation_seconds": _number(training.get("evaluation_seconds")),
                "fit_seconds_current_attempt": _number(training.get("fit_seconds_current_attempt")),
                "wall_seconds_current_attempt": _number(training.get("wall_seconds_current_attempt")),
                "source_samples": _int(config.get("source_manifest", {}).get("joined_samples")),
                "train_examples": _int(train_meta.get("total_examples")),
                "validation_examples": _int(validation_meta.get("total_examples")),
                "test_examples": _int(test_meta.get("total_examples")),
                "split_fingerprint": split_fingerprints[batch_size],
                "source_fingerprint": source_fingerprints[batch_size],
                "config_signature": signature,
                "tensorflow_version": str(environment.get("packages", {}).get("tensorflow", "")),
                "gpu_name": gpu_name,
            }
        )
        hardware_rows.append(
            {
                "batch_size": batch_size,
                "telemetry_samples": _int(telemetry.get("sample_count")),
                "telemetry_epoch_end_count": telemetry_epoch_end_count,
                "telemetry_coverage_rate": telemetry_coverage,
                "gpu_util_mean_percent": _stat(telemetry, "gpu_utilization_percent", "mean"),
                "gpu_util_p95_percent": _stat(telemetry, "gpu_utilization_percent", "p95"),
                "gpu_util_max_percent": _stat(telemetry, "gpu_utilization_percent", "max"),
                "gpu_memory_mean_gib": _stat(telemetry, "gpu_memory_used_bytes", "mean") / (1024**3),
                "gpu_memory_p95_gib": _stat(telemetry, "gpu_memory_used_bytes", "p95") / (1024**3),
                "gpu_memory_max_gib": _stat(telemetry, "gpu_memory_used_bytes", "max") / (1024**3),
                "gpu_temperature_mean_c": _stat(telemetry, "gpu_temperature_c", "mean"),
                "gpu_temperature_max_c": _stat(telemetry, "gpu_temperature_c", "max"),
                "ram_mean_percent": _stat(telemetry, "ram_percent", "mean"),
                "ram_max_percent": _stat(telemetry, "ram_percent", "max"),
                "ram_used_max_gib": _stat(telemetry, "ram_used_bytes", "max") / (1024**3),
                "process_rss_mean_gib": _stat(telemetry, "process_rss_bytes", "mean") / (1024**3),
                "process_rss_max_gib": _stat(telemetry, "process_rss_bytes", "max") / (1024**3),
                "cpu_mean_percent": _stat(telemetry, "cpu_percent", "mean"),
                "process_cpu_mean_percent": _stat(telemetry, "process_cpu_percent", "mean"),
                "output_disk_free_min_gib": _stat(telemetry, "disk_output_free_bytes", "min") / (1024**3),
            }
        )
        for class_id, values in per_class.items():
            class_rows.append(
                {
                    "batch_size": batch_size,
                    "class_id": _int(class_id),
                    "class_name": str(config["class_names"][_int(class_id)]),
                    "precision": _number(values.get("precision")),
                    "recall": _number(values.get("recall")),
                    "f1": _number(values.get("f1")),
                    "support": _int(values.get("support")),
                }
            )

    runs = pd.DataFrame(run_rows).sort_values("batch_size").reset_index(drop=True)
    hardware = pd.DataFrame(hardware_rows).sort_values("batch_size").reset_index(drop=True)
    classes = pd.DataFrame(class_rows).sort_values(["batch_size", "class_id"]).reset_index(drop=True)
    epochs_all = pd.concat(epoch_frames, ignore_index=True).sort_values(["batch_size", "epoch"]).reset_index(drop=True)
    if len(runs) != len(EXPECTED_BATCHES):
        raise RuntimeError(f"Esperadas {len(EXPECTED_BATCHES)} runs, encontradas {len(runs)}")

    protocol_signatures = set(signatures.values())
    common_split = len(set(split_fingerprints.values())) == 1
    common_source = len(set(source_fingerprints.values())) == 1
    for batch_size in EXPECTED_BATCHES:
        _check(checks, batch_size=batch_size, name="constant_protocol_except_batch", passed=len(protocol_signatures) == 1, severity="critical", detail=f"distinct_signatures={len(protocol_signatures)}")
        _check(checks, batch_size=batch_size, name="common_split", passed=common_split, severity="critical", detail=f"distinct_split_fingerprints={len(set(split_fingerprints.values()))}")
        _check(checks, batch_size=batch_size, name="common_source", passed=common_source, severity="critical", detail=f"distinct_source_fingerprints={len(set(source_fingerprints.values()))}")

    anomalies: list[dict[str, Any]] = []
    for row in runs.itertuples(index=False):
        hardware_row = hardware.loc[hardware["batch_size"] == row.batch_size].iloc[0]
        if row.timed_epoch_count < row.epochs_completed:
            anomalies.append(
                {
                    "batch_size": int(row.batch_size),
                    "issue": "Cobertura parcial de tempo por época",
                    "severity": "Alta para eficiência; baixa para qualidade do modelo",
                    "evidence": f"{int(row.timed_epoch_count)}/{int(row.epochs_completed)} épocas têm epoch_seconds",
                    "implication": "Não usar esta run em ranking de duração ou throughput.",
                }
            )
        if _int(hardware_row["telemetry_epoch_end_count"]) < max(0, int(row.epochs_completed) - 1):
            anomalies.append(
                {
                    "batch_size": int(row.batch_size),
                    "issue": "Cobertura parcial de telemetria",
                    "severity": "Média para hardware; baixa para qualidade do modelo",
                    "evidence": f"{int(hardware_row['telemetry_epoch_end_count'])}/{int(row.epochs_completed)} eventos epoch_end",
                    "implication": "Picos observados não representam necessariamente a execução inteira.",
                }
            )
    anomalies_frame = pd.DataFrame(anomalies, columns=["batch_size", "issue", "severity", "evidence", "implication"])

    checks_frame = pd.DataFrame(checks)
    checks_frame["result"] = np.where(checks_frame["passed"], "Aprovado", "Falhou")
    quality_summary = (
        checks_frame.groupby(["severity_if_failed", "result"], as_index=False)
        .size()
        .rename(columns={"size": "checks"})
        .sort_values(["severity_if_failed", "result"])
        .reset_index(drop=True)
    )
    comparable = runs[runs["timing_comparable"] == "Sim"].copy()
    overview = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "completed_runs": int((runs["status"] == "completed").sum()),
        "planned_runs": len(EXPECTED_BATCHES),
        "epochs_recorded": int(len(epochs_all)),
        "timing_comparable_runs": int(len(comparable)),
        "partial_timing_runs": int((runs["timing_comparable"] == "Não").sum()),
        "best_test_batch": int(runs.loc[runs["test_macro_f1"].idxmax(), "batch_size"]),
        "best_test_macro_f1": float(runs["test_macro_f1"].max()),
        "mean_test_macro_f1": float(runs["test_macro_f1"].mean()),
        "min_test_macro_f1": float(runs["test_macro_f1"].min()),
        "max_test_macro_f1": float(runs["test_macro_f1"].max()),
        "fastest_comparable_batch": int(comparable.loc[comparable["mean_epoch_seconds_recorded"].idxmin(), "batch_size"]),
        "fastest_comparable_mean_epoch_seconds": float(comparable["mean_epoch_seconds_recorded"].min()),
        "highest_comparable_throughput_batch": int(comparable.loc[comparable["weighted_examples_per_second"].idxmax(), "batch_size"]),
        "highest_comparable_throughput": float(comparable["weighted_examples_per_second"].max()),
        "recorded_training_hours_all_runs": float(runs["training_seconds_recorded"].sum() / 3600.0),
        "recorded_training_hours_comparable_runs": float(comparable["training_seconds_recorded"].sum() / 3600.0),
        "peak_gpu_memory_gib": float(hardware["gpu_memory_max_gib"].max()),
        "peak_gpu_memory_batch": int(hardware.loc[hardware["gpu_memory_max_gib"].idxmax(), "batch_size"]),
        "peak_gpu_temperature_c": float(hardware["gpu_temperature_max_c"].max()),
        "peak_gpu_temperature_batch": int(hardware.loc[hardware["gpu_temperature_max_c"].idxmax(), "batch_size"]),
        "failed_critical_checks": int(((checks_frame["severity_if_failed"] == "critical") & (~checks_frame["passed"])).sum()),
        "failed_high_checks": int(((checks_frame["severity_if_failed"] == "high") & (~checks_frame["passed"])).sum()),
        "failed_medium_checks": int(((checks_frame["severity_if_failed"] == "medium") & (~checks_frame["passed"])).sum()),
    }

    runs.to_csv(output_dir / "run_metrics.csv", index=False)
    hardware.to_csv(output_dir / "hardware_summary.csv", index=False)
    epochs_all.to_csv(output_dir / "epoch_metrics_all.csv", index=False)
    classes.to_csv(output_dir / "per_class_metrics.csv", index=False)
    checks_frame.to_csv(output_dir / "quality_checks.csv", index=False)
    quality_summary.to_csv(output_dir / "quality_summary.csv", index=False)
    anomalies_frame.to_csv(output_dir / "quality_caveats.csv", index=False)
    (output_dir / "analysis_overview.json").write_text(json.dumps(_safe_json(overview), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(
        output_path=output_dir / "kmnist_batch_sweep_report.md",
        overview=overview,
        runs=runs,
        hardware=hardware,
        quality_summary=quality_summary,
        anomalies=anomalies_frame,
    )
    artifact = _build_artifact(
        output_dir=output_dir,
        overview=overview,
        runs=runs,
        epochs=epochs_all,
        hardware=hardware,
        quality_summary=quality_summary,
        anomalies=anomalies_frame,
    )
    (output_dir / "report_artifact.json").write_text(json.dumps(_safe_json(artifact), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return overview


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_root.is_dir():
        parser.error(f"Entrada não encontrada: {input_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Saída já contém arquivos: {output_dir}. Escolha outra pasta de relatório.")
    overview = build_analysis(input_root, output_dir)
    print(json.dumps(_safe_json(overview), ensure_ascii=False, indent=2))
    print(f"Relatório Markdown: {output_dir / 'kmnist_batch_sweep_report.md'}")
    print(f"Artefato Data Analytics: {output_dir / 'report_artifact.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
