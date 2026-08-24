"""Relatórios duráveis e sem dependência de TensorFlow para cada benchmark."""

from __future__ import annotations

import csv
import html
import io
import math
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .state import RunPaths, atomic_write_bytes, atomic_write_json, atomic_write_text, canonical_json, read_json, utc_now


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    return value


def _safe_number(value: Any) -> float | int | str | None:
    value = _to_builtin(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return int(value) if isinstance(value, int) else numeric
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return numeric if math.isfinite(numeric) else None


def _argmax(values: Sequence[Any]) -> int:
    if not values:
        return 0
    converted = [_safe_number(item) for item in values]
    numeric = [float(item) if isinstance(item, (int, float)) else float("-inf") for item in converted]
    return max(range(len(numeric)), key=numeric.__getitem__)


def _label_vector(values: Any) -> list[Any]:
    """Accept scalar labels, column labels, one-hot vectors or score matrices."""

    raw = _to_builtin(values)
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        return [raw]
    result: list[Any] = []
    for value in raw:
        item = _to_builtin(value)
        if isinstance(item, (list, tuple)):
            if len(item) == 0:
                result.append(None)
            elif len(item) == 1:
                result.append(_to_builtin(item[0]))
            else:
                result.append(_argmax(item))
        else:
            result.append(item)
    return result


def _sortable_label(value: Any) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except (TypeError, ValueError):
        return (1, str(value))


def _labels_and_names(
    y_true: Sequence[Any], y_pred: Sequence[Any], class_names: Sequence[str] | Mapping[Any, str] | None
) -> tuple[list[Any], dict[Any, str]]:
    mapping: dict[Any, str] = {}
    labels: list[Any] = []
    if isinstance(class_names, Mapping):
        labels.extend(class_names.keys())
        mapping = {key: str(value) for key, value in class_names.items()}
    elif class_names is not None:
        labels.extend(range(len(class_names)))
        mapping = {index: str(name) for index, name in enumerate(class_names)}
    labels.extend(y_true)
    labels.extend(y_pred)
    deduplicated: list[Any] = []
    for label in labels:
        if label not in deduplicated:
            deduplicated.append(label)
    deduplicated.sort(key=_sortable_label)
    for label in deduplicated:
        mapping.setdefault(label, str(label))
    return deduplicated, mapping


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    class_names: Sequence[str] | Mapping[Any, str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, balanced accuracy, macro-F1 and a full confusion matrix.

    It intentionally avoids a hard dependency on scikit-learn so reports can
    still be generated on a machine used only to inspect completed runs.
    """

    actual = _label_vector(y_true)
    predicted = _label_vector(y_pred)
    if len(actual) != len(predicted):
        raise ValueError(f"y_true ({len(actual)}) e y_pred ({len(predicted)}) têm tamanhos diferentes")
    labels, names = _labels_and_names(actual, predicted, class_names)
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(actual, predicted):
        matrix[label_index[truth]][label_index[prediction]] += 1

    per_class: list[dict[str, Any]] = []
    recalls_with_support: list[float] = []
    f1_values: list[float] = []
    correct = 0
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(labels)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(labels)) if column != index)
        support = sum(matrix[index])
        precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
        recall = 0.0 if support == 0 else true_positive / support
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        if support:
            recalls_with_support.append(recall)
        f1_values.append(f1)
        correct += true_positive
        per_class.append(
            {
                "class_index": index,
                "label": label,
                "class_name": names[label],
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    n_samples = len(actual)
    return {
        "n_samples": n_samples,
        "accuracy": None if n_samples == 0 else correct / n_samples,
        "balanced_accuracy": None if not recalls_with_support else sum(recalls_with_support) / len(recalls_with_support),
        "macro_f1": None if not f1_values else sum(f1_values) / len(f1_values),
        "labels": labels,
        "class_names": [names[label] for label in labels],
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _to_builtin(row.get(field)) for field in fieldnames})
    return buffer.getvalue()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    return atomic_write_text(path, _csv_text(rows, fieldnames))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_fallback_heatmap(path: Path, matrix: Sequence[Sequence[int]]) -> Path:
    """Small dependency-free PNG heatmap for headless/minimal installations."""

    count = max(1, len(matrix))
    cell = max(3, min(18, 720 // count))
    width = height = max(1, count * cell)
    maximum = max((max(row) if row else 0 for row in matrix), default=0)
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # PNG filter: none
        row_index = min(count - 1, y // cell)
        for x in range(width):
            column_index = min(count - 1, x // cell)
            value = matrix[row_index][column_index] if maximum else 0
            strength = 0 if maximum == 0 else int(255 * math.sqrt(value / maximum))
            # White (zero) to readable deep blue (max).
            raw.extend((255 - strength, 255 - strength // 2, 255))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    payload += _png_chunk(b"IEND", b"")
    return atomic_write_bytes(path, payload)


def write_confusion_matrix_png(
    path: str | Path, matrix: Sequence[Sequence[int]], class_names: Sequence[str]
) -> Path:
    """Write a labelled matplotlib matrix when possible, otherwise valid PNG."""

    destination = Path(path)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        size = max(5.0, min(18.0, 0.28 * max(1, len(class_names)) + 3.0))
        figure, axis = plt.subplots(figsize=(size, size))
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xlabel("Classe predita")
        axis.set_ylabel("Classe real")
        if len(class_names) <= 30:
            ticks = list(range(len(class_names)))
            axis.set_xticks(ticks, labels=class_names, rotation=90, fontsize=7)
            axis.set_yticks(ticks, labels=class_names, fontsize=7)
        else:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return destination
    except Exception:
        return _write_fallback_heatmap(destination, matrix)


def _history_rows(history: Any) -> list[dict[str, Any]]:
    if history is None:
        return []
    history = _to_builtin(history)
    if isinstance(history, Mapping):
        possible_history = history.get("history") if isinstance(history.get("history"), Mapping) else history
        keys = [str(key) for key, value in possible_history.items() if isinstance(_to_builtin(value), (list, tuple))]
        if keys:
            length = max(len(_to_builtin(possible_history[key])) for key in keys)
            rows: list[dict[str, Any]] = []
            for index in range(length):
                row: dict[str, Any] = {"epoch": index + 1}
                for key in keys:
                    values = _to_builtin(possible_history[key])
                    row[key] = values[index] if index < len(values) else None
                rows.append(row)
            return rows
        return [{str(key): value for key, value in possible_history.items()}]
    if isinstance(history, (list, tuple)):
        return [
            ({"epoch": index + 1, **{str(key): value for key, value in item.items()}} if isinstance(item, Mapping) else {"epoch": index + 1, "value": item})
            for index, item in enumerate(history)
        ]
    return [{"value": history}]


def summarize_training_history(history: Any) -> dict[str, Any]:
    """Derive total and mean epoch duration from Keras-style history rows."""

    rows = _history_rows(history)
    durations: list[float] = []
    for row in rows:
        value = _safe_number(row.get("epoch_seconds"))
        if isinstance(value, (int, float)):
            durations.append(float(value))
    return {
        "epochs_recorded": len(rows),
        "total_seconds": sum(durations) if durations else None,
        "mean_epoch_seconds": (sum(durations) / len(durations)) if durations else None,
        "min_epoch_seconds": min(durations) if durations else None,
        "max_epoch_seconds": max(durations) if durations else None,
    }


def _write_history_png(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path | None:
    numeric_keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key != "epoch" and isinstance(_safe_number(value), (int, float)) and key not in numeric_keys:
                numeric_keys.append(key)
    if not rows or not numeric_keys:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(9, 4.5))
        x = [row.get("epoch", index + 1) for index, row in enumerate(rows)]
        for key in numeric_keys[:8]:
            y = [_safe_number(row.get(key)) for row in rows]
            values = [float(value) if isinstance(value, (int, float)) else float("nan") for value in y]
            axis.plot(x, values, label=key)
        axis.set_xlabel("Época")
        axis.set_ylabel("Métrica")
        axis.legend(loc="best", fontsize=8)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path
    except Exception:
        return None


def _metric_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in ("n_samples", "loss", "accuracy", "balanced_accuracy", "macro_f1"):
        if key in metrics:
            rows.append({"metric": key, "value": metrics[key]})
    return rows


def _render_run_html(
    *,
    title: str,
    metrics: Mapping[str, Any],
    per_class: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None,
    training_summary: Mapping[str, Any] | None,
    telemetry_summary: Mapping[str, Any] | None,
    has_history_png: bool,
) -> str:
    metric_table = "".join(
        f"<tr><th>{html.escape(str(row['metric']))}</th><td>{html.escape(str(row['value']))}</td></tr>"
        for row in _metric_rows(metrics)
    )
    class_table = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(field, '')))}</td>"
            for field in ("class_index", "class_name", "support", "precision", "recall", "f1")
        )
        + "</tr>"
        for row in per_class
    )
    config_html = html.escape(canonical_json(config or {}))
    training_html = html.escape(canonical_json(training_summary or {}))
    telemetry_html = html.escape(canonical_json(telemetry_summary or {}))
    history_image = '<h2>Histórico</h2><img src="history.png" alt="Curvas do treinamento">' if has_history_png else ""
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem;max-width:1200px}}table{{border-collapse:collapse;margin:.7rem 0}}th,td{{border:1px solid #ccc;padding:.35rem .55rem;text-align:right}}th{{background:#f3f5f7}}td:nth-child(2),td:nth-child(3){{text-align:left}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;background:#f6f8fa;padding:1rem}}</style>
</head><body><h1>{html.escape(title)}</h1>
<h2>Métricas de teste</h2><table>{metric_table}</table>
<h2>Matriz de confusão</h2><img src="confusion_matrix.png" alt="Matriz de confusão">
{history_image}
<h2>Métricas por classe</h2><table><thead><tr><th>Índice</th><th>Classe</th><th>Suporte</th><th>Precisão</th><th>Recall</th><th>F1</th></tr></thead><tbody>{class_table}</tbody></table>
<h2>Configuração</h2><pre>{config_html}</pre>
<h2>Treinamento</h2><pre>{training_html}</pre>
<h2>Telemetria</h2><pre>{telemetry_html}</pre>
</body></html>"""


def write_run_report(
    run_dir: str | Path,
    *,
    y_true: Any,
    y_pred: Any,
    class_names: Sequence[str] | Mapping[Any, str] | None = None,
    test_metrics: Mapping[str, Any] | None = None,
    history: Any = None,
    training_summary: Mapping[str, Any] | None = None,
    telemetry_summary: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Create all evaluation artifacts for one run and return their locations.

    ``test_metrics`` can add Keras loss or other test-only values.  Calculated
    classification values remain authoritative if the same key is supplied.
    """

    paths = RunPaths.from_root(run_dir).ensure()
    manifest = read_json(paths.manifest, default={}) or {}
    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    if test_metrics:
        extras = {str(key): _to_builtin(value) for key, value in test_metrics.items() if key not in metrics}
        metrics.update(extras)
    metrics["generated_at"] = utc_now()
    metrics["run_id"] = manifest.get("run_id", paths.root.name)

    if config is None and isinstance(manifest.get("config"), Mapping):
        config = manifest["config"]
    if telemetry_summary is None:
        telemetry_summary = read_json(paths.telemetry / "summary.json", default={}) or {}
    if environment is None:
        environment = read_json(paths.telemetry / "environment.json", default={}) or {}

    history_rows = _history_rows(history)
    derived_training = summarize_training_history(history_rows)
    effective_training = {**derived_training, **dict(training_summary or {})}
    report_payload = {
        "schema_version": 1,
        "generated_at": metrics["generated_at"],
        "run_id": metrics["run_id"],
        "test": {key: value for key, value in metrics.items() if key not in {"per_class", "confusion_matrix", "labels", "class_names"}},
        "per_class": metrics["per_class"],
        "config": config or {},
        "training": effective_training,
        "telemetry": dict(telemetry_summary or {}),
        "environment": dict(environment or {}),
    }

    artifacts: dict[str, Path] = {}
    artifacts["metrics"] = atomic_write_json(paths.artifacts / "metrics.json", report_payload)
    artifacts["classification_json"] = atomic_write_json(
        paths.artifacts / "classification_report.json",
        {
            "test": report_payload["test"],
            "per_class": metrics["per_class"],
            "labels": metrics["labels"],
            "class_names": metrics["class_names"],
        },
    )
    artifacts["classification_csv"] = _write_csv(
        paths.artifacts / "classification_report.csv",
        metrics["per_class"],
        ("class_index", "label", "class_name", "support", "precision", "recall", "f1"),
    )

    confusion_rows = []
    for row_index, row in enumerate(metrics["confusion_matrix"]):
        confusion_rows.append(
            {
                "true_label": metrics["labels"][row_index],
                "true_class_name": metrics["class_names"][row_index],
                **{str(metrics["class_names"][column]): count for column, count in enumerate(row)},
            }
        )
    artifacts["confusion_csv"] = _write_csv(
        paths.artifacts / "confusion_matrix.csv",
        confusion_rows,
        ("true_label", "true_class_name", *[str(name) for name in metrics["class_names"]]),
    )
    artifacts["confusion_png"] = write_confusion_matrix_png(
        paths.artifacts / "confusion_matrix.png", metrics["confusion_matrix"], metrics["class_names"]
    )

    if history_rows:
        history_fields = ["epoch"] + sorted({key for row in history_rows for key in row if key != "epoch"})
        artifacts["history_csv"] = _write_csv(paths.artifacts / "history.csv", history_rows, history_fields)
        history_png = _write_history_png(paths.artifacts / "history.png", history_rows)
        if history_png is not None:
            artifacts["history_png"] = history_png
    artifacts["html"] = atomic_write_text(
        paths.artifacts / "report.html",
        _render_run_html(
            title=f"Benchmark — {metrics['run_id']}",
            metrics=metrics,
            per_class=metrics["per_class"],
            config=config,
            training_summary=effective_training,
            telemetry_summary=telemetry_summary,
            has_history_png="history_png" in artifacts,
        ),
    )
    return artifacts


def _get_nested(payload: Mapping[str, Any], *candidates: Sequence[str]) -> Any:
    for candidate in candidates:
        current: Any = payload
        found = True
        for part in candidate:
            if not isinstance(current, Mapping) or part not in current:
                found = False
                break
            current = current[part]
        if found:
            return current
    return None


def _dataset_run_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    run_root = dataset_dir / "runs"
    # ``runs/<run-id>`` is the current layout, while ``<norm>/<balance>/seed``
    # is accepted too so a report can consolidate an older/interrupted layout.
    manifests = sorted((run_root if run_root.exists() else dataset_dir).rglob("manifest.json"))
    rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        run_dir = manifest_path.parent
        manifest = read_json(manifest_path, default={}) or {}
        payload = read_json(run_dir / "artifacts" / "metrics.json", default={}) or {}
        test = payload.get("test", {}) if isinstance(payload, Mapping) else {}
        training = payload.get("training", {}) if isinstance(payload, Mapping) else {}
        telemetry = payload.get("telemetry", {}) if isinstance(payload, Mapping) else {}
        if not telemetry:
            telemetry = read_json(run_dir / "telemetry" / "summary.json", default={}) or {}
        identity = manifest.get("identity", {}) if isinstance(manifest.get("identity"), Mapping) else {}
        data_metadata: Mapping[str, Any] = {}
        for key in ("data_metadata", "prepared_metadata", "dataset_metadata", "metadata"):
            candidate = manifest.get(key)
            if isinstance(candidate, Mapping) and isinstance(candidate.get("balancing"), Mapping):
                data_metadata = candidate
                break
        balancing = data_metadata.get("balancing", {}) if isinstance(data_metadata.get("balancing"), Mapping) else {}
        manifest_config = manifest.get("config", {}) if isinstance(manifest.get("config"), Mapping) else {}
        row = {
            "run_id": manifest.get("run_id", run_dir.name),
            "status": manifest.get("status", "unknown"),
            "normalization": identity.get("normalization", manifest_config.get("normalization")),
            "balance_mode": identity.get("balance_mode", manifest_config.get("balance_mode")),
            "balance_no_op": balancing.get("is_noop"),
            "raw_train_class_counts": balancing.get("raw_class_counts"),
            "output_train_class_counts": balancing.get("output_class_counts"),
            "seed": identity.get("seed", manifest_config.get("seed")),
            "attempt": manifest.get("attempt"),
            "config_fingerprint": manifest.get("config_fingerprint"),
            "split_fingerprint": manifest.get("split_fingerprint"),
            "test_loss": _get_nested(test, ("loss",)),
            "test_accuracy": _get_nested(test, ("accuracy",)),
            "test_balanced_accuracy": _get_nested(test, ("balanced_accuracy",)),
            "test_macro_f1": _get_nested(test, ("macro_f1",)),
            "test_samples": _get_nested(test, ("n_samples",)),
            "training_total_seconds": _get_nested(
                training, ("total_seconds",), ("training_seconds",), ("duration_seconds",)
            ),
            "mean_epoch_seconds": _get_nested(training, ("mean_epoch_seconds",), ("avg_epoch_seconds",)),
            "telemetry_samples": _get_nested(telemetry, ("sample_count",)),
            "gpu_backend": telemetry.get("gpu_backend"),
            "gpu_memory_kind": _get_nested(telemetry, ("categorical", "gpu_memory_kind", "current")),
            "thermal_pressure": _get_nested(telemetry, ("categorical", "thermal_pressure", "current")),
            "peak_cpu_percent": _get_nested(telemetry, ("metrics", "cpu_percent", "max")),
            "peak_ram_used_bytes": _get_nested(telemetry, ("metrics", "ram_used_bytes", "max")),
            "peak_process_rss_bytes": _get_nested(telemetry, ("metrics", "process_rss_bytes", "max")),
            "peak_gpu_memory_used_bytes": _get_nested(telemetry, ("metrics", "gpu_memory_used_bytes", "max")),
            "peak_gpu_utilization_percent": _get_nested(telemetry, ("metrics", "gpu_utilization_percent", "max")),
            "peak_gpu_renderer_utilization_percent": _get_nested(
                telemetry, ("metrics", "gpu_renderer_utilization_percent", "max")
            ),
            "peak_gpu_tiler_utilization_percent": _get_nested(
                telemetry, ("metrics", "gpu_tiler_utilization_percent", "max")
            ),
            "peak_gpu_driver_allocated_memory_bytes": _get_nested(
                telemetry, ("metrics", "gpu_driver_allocated_memory_bytes", "max")
            ),
            "peak_gpu_system_memory_in_use_bytes": _get_nested(
                telemetry, ("metrics", "gpu_system_memory_in_use_bytes", "max")
            ),
            "report_path": (run_dir / "artifacts" / "report.html").relative_to(dataset_dir).as_posix(),
        }
        rows.append(row)
    return rows


def _write_dataset_chart(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path | None:
    usable = [row for row in rows if isinstance(_safe_number(row.get("test_macro_f1")), (int, float))]
    if not usable:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        width = max(10.0, min(28.0, 0.38 * len(usable) + 4.0))
        figure, axis = plt.subplots(figsize=(width, 5.5))
        labels = [str(row["run_id"]) for row in usable]
        values = [float(_safe_number(row["test_macro_f1"])) for row in usable]
        axis.bar(range(len(values)), values, color="#3572a5")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Macro-F1 no teste")
        axis.set_title("Macro-F1 por configuração (ordem estável das runs)")
        axis.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=7)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path
    except Exception:
        return None


def build_dataset_report(dataset_dir: str | Path, *, dataset_name: str | None = None) -> dict[str, Path]:
    """Aggregate a dataset's runs into CSV, JSON, HTML and PNG."""

    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = _dataset_run_rows(root)
    rows.sort(key=lambda row: str(row.get("run_id", "")))
    fields = (
        "run_id",
        "status",
        "normalization",
        "balance_mode",
        "balance_no_op",
        "raw_train_class_counts",
        "output_train_class_counts",
        "seed",
        "attempt",
        "config_fingerprint",
        "split_fingerprint",
        "test_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_samples",
        "training_total_seconds",
        "mean_epoch_seconds",
        "telemetry_samples",
        "gpu_backend",
        "gpu_memory_kind",
        "thermal_pressure",
        "peak_cpu_percent",
        "peak_ram_used_bytes",
        "peak_process_rss_bytes",
        "peak_gpu_memory_used_bytes",
        "peak_gpu_utilization_percent",
        "peak_gpu_renderer_utilization_percent",
        "peak_gpu_tiler_utilization_percent",
        "peak_gpu_driver_allocated_memory_bytes",
        "peak_gpu_system_memory_in_use_bytes",
        "report_path",
    )
    statuses = Counter(str(row.get("status", "unknown")) for row in rows)
    name = dataset_name or root.name
    payload = {
        "schema_version": 1,
        "dataset": name,
        "generated_at": utc_now(),
        "run_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "runs": rows,
    }
    artifacts: dict[str, Path] = {
        "summary_csv": _write_csv(root / "summary.csv", rows, fields),
        "summary_json": atomic_write_json(root / "summary.json", payload),
    }
    chart = _write_dataset_chart(root / "comparison.png", rows)
    if chart is not None:
        artifacts["comparison_png"] = chart

    table = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('run_id', '')))}</td>"
        f"<td>{html.escape(str(row.get('status', '')))}</td>"
        f"<td>{html.escape(str(row.get('normalization', '')))}</td>"
        f"<td>{html.escape(str(row.get('balance_mode', '')))}</td>"
        f"<td>{html.escape(str(row.get('balance_no_op', '')))}</td>"
        f"<td>{html.escape(str(row.get('seed', '')))}</td>"
        f"<td>{html.escape(str(row.get('test_accuracy', '')))}</td>"
        f"<td>{html.escape(str(row.get('test_balanced_accuracy', '')))}</td>"
        f"<td>{html.escape(str(row.get('test_macro_f1', '')))}</td>"
        f"<td><a href=\"{html.escape(str(row.get('report_path', '')))}\">relatório</a></td>"
        "</tr>"
        for row in rows
    )
    chart_html = '<img src="comparison.png" alt="Comparação de Macro-F1">' if chart is not None else ""
    artifacts["html"] = atomic_write_text(
        root / "report.html",
        f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>{html.escape(name)}</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:.35rem .55rem}}th{{background:#f3f5f7}}img{{max-width:100%;height:auto}}</style>
</head><body><h1>Benchmark — {html.escape(name)}</h1><p>Runs: {len(rows)} · Status: {html.escape(str(dict(statuses)))}</p>{chart_html}
<table><thead><tr><th>Run</th><th>Status</th><th>Normalização</th><th>Balanceamento</th><th>No-op</th><th>Seed</th><th>Accuracy</th><th>Balanced accuracy</th><th>Macro-F1</th><th>Detalhes</th></tr></thead><tbody>{table}</tbody></table>
</body></html>""",
    )
    return artifacts


def build_global_index(output_root: str | Path) -> dict[str, Path]:
    """Build a status-only global index; datasets are deliberately not ranked."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    datasets: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = read_json(summary_path, default={}) or {}
        datasets.append(
            {
                "dataset": summary.get("dataset", summary_path.parent.name),
                "run_count": summary.get("run_count", 0),
                "status_counts": summary.get("status_counts", {}),
                "report_path": str((summary_path.parent / "report.html").relative_to(root)),
            }
        )
    payload = {"schema_version": 1, "generated_at": utc_now(), "datasets": datasets}
    json_path = atomic_write_json(root / "index.json", payload)
    table = "".join(
        f"<tr><td>{html.escape(str(row['dataset']))}</td><td>{row['run_count']}</td><td>{html.escape(str(row['status_counts']))}</td><td><a href=\"{html.escape(row['report_path'])}\">abrir</a></td></tr>"
        for row in datasets
    )
    html_path = atomic_write_text(
        root / "index.html",
        f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Índice dos benchmarks</title>
<style>body{{font-family:Arial,sans-serif;margin:2rem}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:.35rem .55rem}}th{{background:#f3f5f7}}</style>
</head><body><h1>Índice dos benchmarks</h1><p>Índice de status e links; não há ranking entre datasets.</p><table><thead><tr><th>Dataset</th><th>Runs</th><th>Status</th><th>Relatório</th></tr></thead><tbody>{table}</tbody></table></body></html>""",
    )
    return {"json": json_path, "html": html_path}


__all__ = [
    "build_dataset_report",
    "build_global_index",
    "compute_classification_metrics",
    "summarize_training_history",
    "write_confusion_matrix_png",
    "write_run_report",
]
