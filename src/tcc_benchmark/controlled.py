"""Pure helpers shared by the sequential controlled benchmark supervisor."""

from __future__ import annotations

import csv
import html
import io
from pathlib import Path
from typing import Any, Iterable, Mapping

from .state import atomic_write_bytes, atomic_write_json, atomic_write_text


def number(value: Any) -> float | None:
    """Return a finite float, keeping incomplete artefacts out of selection."""

    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if candidate == candidate and candidate not in {float("inf"), float("-inf")} else None


def completed_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row.get("status")) == "completed" and number(row.get("macro_f1")) is not None]


def select_batch(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Maximise Macro-F1; break an exact tie with mean epoch duration."""

    candidates = completed_rows(rows)
    if not candidates:
        raise ValueError("Nenhuma run de batch concluída possui Macro-F1.")
    return min(
        candidates,
        key=lambda row: (
            -float(number(row["macro_f1"]) or 0.0),
            float(number(row.get("mean_epoch_seconds")) or float("inf")),
            int(row["batch_size"]),
        ),
    )


def select_quantization(rows: Iterable[Mapping[str, Any]], *, max_macro_f1_drop: float = 0.01) -> dict[str, Any]:
    """Select the fastest variant within the permitted FP32 Macro-F1 drop."""

    candidates = completed_rows(rows)
    fp32 = next((row for row in candidates if row.get("variant") == "fp32"), None)
    if fp32 is None:
        raise ValueError("A seleção de quantização exige o resultado FP32 concluído.")
    threshold = float(number(fp32["macro_f1"]) or 0.0) - float(max_macro_f1_drop)
    eligible = [row for row in candidates if float(number(row["macro_f1"]) or -1.0) >= threshold]
    if not eligible:
        raise ValueError("Nenhuma variante de quantização ficou dentro da tolerância de Macro-F1.")
    winner = min(
        eligible,
        key=lambda row: (
            float(number(row.get("median_batch_latency_ms")) or float("inf")),
            float(number(row.get("serialized_litert_bytes")) or number(row.get("serialized_model_bytes")) or float("inf")),
            -float(number(row.get("throughput_examples_per_second")) or 0.0),
            -float(number(row.get("macro_f1")) or 0.0),
            str(row.get("variant")),
        ),
    )
    return {**winner, "macro_f1_threshold": threshold, "max_macro_f1_drop": float(max_macro_f1_drop)}


def select_activation(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Maximise Macro-F1; use training duration only as deterministic tie-breaker."""

    candidates = completed_rows(rows)
    if not candidates:
        raise ValueError("Nenhuma run de ativação concluída possui Macro-F1.")
    return min(
        candidates,
        key=lambda row: (
            -float(number(row["macro_f1"]) or 0.0),
            float(number(row.get("mean_epoch_seconds")) or float("inf")),
            str(row.get("hidden_activation")),
        ),
    )


def write_phase_report(root: str | Path, *, phase: str, rows: Iterable[Mapping[str, Any]], winners: Mapping[str, Mapping[str, Any]]) -> dict[str, Path]:
    """Write portable CSV/JSON/HTML/PNG reports without requiring a notebook."""

    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    values = [dict(row) for row in rows]
    fields = sorted({str(field) for row in values for field in row}) or ["status"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(values)
    csv_path = atomic_write_text(destination / f"{phase}_comparison.csv", stream.getvalue())
    json_path = atomic_write_json(destination / f"{phase}_comparison.json", {"phase": phase, "rows": values, "winners": dict(winners)})

    headers = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape('' if row.get(field) is None else str(row.get(field)))}</td>" for field in fields) + "</tr>"
        for row in values
    )
    winner_items = "".join(
        f"<li><strong>{html.escape(dataset)}</strong>: {html.escape(str(winner.get('batch_size', winner.get('variant', winner.get('hidden_activation', '')))))}</li>"
        for dataset, winner in sorted(winners.items())
    )
    html_path = atomic_write_text(
        destination / f"{phase}_comparison.html",
        "<!doctype html><meta charset='utf-8'>"
        f"<title>Benchmark controlado — {html.escape(phase)}</title><h1>Benchmark controlado — {html.escape(phase)}</h1>"
        f"<h2>Vencedores</h2><ul>{winner_items}</ul><table border='1'><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>",
    )
    png_path = destination / f"{phase}_macro_f1.png"
    _write_macro_f1_plot(png_path, values, phase)
    return {"csv": csv_path, "json": json_path, "html": html_path, "png": png_path}


def _write_macro_f1_plot(path: Path, rows: list[dict[str, Any]], phase: str) -> None:
    """Render a compact summary figure, tolerating a missing matplotlib install."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        atomic_write_bytes(path, b"")
        return
    usable = [row for row in rows if number(row.get("macro_f1")) is not None]
    figure, axis = plt.subplots(figsize=(max(7, len(usable) * 0.4), 4.5))
    labels = [f"{row.get('dataset', '')}\n{row.get('batch_size', row.get('variant', row.get('hidden_activation', '')))}" for row in usable]
    axis.bar(range(len(usable)), [float(number(row["macro_f1"]) or 0.0) for row in usable], color="#2563eb")
    axis.set_xticks(range(len(usable)), labels, rotation=55, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Macro-F1")
    axis.set_title(f"{phase}: Macro-F1")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    atomic_write_bytes(path, buffer.getvalue())


__all__ = [
    "completed_rows",
    "number",
    "select_activation",
    "select_batch",
    "select_quantization",
    "write_phase_report",
]
