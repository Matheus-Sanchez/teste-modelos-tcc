"""Consolida status, métricas e telemetria do teste curto em CSV/JSON/HTML.

O script é seguro para execução durante o treino: runs pendentes ou em curso
aparecem sem métricas finais, e o mesmo relatório é sobrescrito atomicamente
quando chamado novamente ao término da bateria.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _metric(summary: dict[str, Any], name: str, statistic: str = "max") -> float | None:
    metrics = summary.get("metrics", {})
    value = metrics.get(name, {}) if isinstance(metrics, dict) else {}
    raw = value.get(statistic) if isinstance(value, dict) else None
    return float(raw) if isinstance(raw, (int, float)) else None


def _last_epoch(path: Path) -> dict[str, str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return dict(list(csv.DictReader(handle))[-1])
    except (OSError, IndexError):
        return {}


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status_path in sorted(output_root.glob("*/runs/*/status.json")):
        run_dir = status_path.parent
        status = _read_json(status_path)
        manifest = _read_json(run_dir / "manifest.json")
        telemetry = _read_json(run_dir / "telemetry" / "summary.json")
        epoch = _last_epoch(run_dir / "checkpoints" / "epoch_metrics.csv")
        test = _read_json(run_dir / "artifacts" / "test_metrics.json")
        classification = test.get("classification", {}) if isinstance(test.get("classification"), dict) else {}
        config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
        protocol = config.get("protocol", {}) if isinstance(config.get("protocol"), dict) else {}
        # ``runner`` records durable elapsed-time values separately from the
        # lightweight status heartbeat.  Read that final summary so completed
        # rows expose the total, mean epoch and evaluation time requested by
        # the benchmark protocol.
        training = _read_json(run_dir / "logs" / "training_summary.json")
        if not training:
            training = status.get("training", {}) if isinstance(status.get("training"), dict) else {}
        rows.append(
            {
                "dataset": run_dir.parents[1].name,
                "run_id": str(status.get("run_id", run_dir.name)),
                "status": str(status.get("status", "unknown")),
                "detail": "" if status.get("detail") is None else str(status.get("detail")),
                "epochs_logged": int(epoch.get("epoch", 0) or 0),
                "last_loss": _number(epoch.get("loss")),
                "last_accuracy": _number(epoch.get("accuracy")),
                "last_val_macro_f1": _number(epoch.get("val_macro_f1")),
                "last_epoch_seconds": _number(epoch.get("epoch_seconds")),
                "last_train_examples_per_second": _number(epoch.get("train_examples_per_second")),
                "test_macro_f1": _number(classification.get("macro_f1")),
                "test_accuracy": _number(classification.get("accuracy")),
                "training_seconds": _number(training.get("training_seconds")),
                "mean_epoch_seconds": _number(training.get("mean_epoch_seconds")),
                "mean_train_examples_per_second": _number(training.get("mean_train_examples_per_second")),
                "evaluation_seconds": _number(training.get("evaluation_seconds")),
                "vram_peak_gib": _metric(telemetry, "gpu_memory_used_bytes") / (1024**3)
                if _metric(telemetry, "gpu_memory_used_bytes") is not None
                else None,
                "gpu_utilization_peak_pct": _metric(telemetry, "gpu_utilization_percent"),
                "gpu_temperature_peak_c": _metric(telemetry, "gpu_temperature_c"),
                "ram_peak_gib": _metric(telemetry, "ram_used_bytes") / (1024**3)
                if _metric(telemetry, "ram_used_bytes") is not None
                else None,
                "dtype_policy": str(protocol.get("dtype_policy", "")),
                "batch_size": config.get("batch_size"),
                "max_epochs": config.get("max_epochs"),
                "path": str(run_dir),
            }
        )
    return rows


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def write_report(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["dataset", "run_id", "status"]
    csv_path = output_root / "short-test-summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(rows),
        "status_counts": statuses,
        "rows": rows,
    }
    (output_root / "short-test-summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    headers = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_format(row.get(field)))}</td>" for field in fields) + "</tr>"
        for row in rows
    )
    page = f"""<!doctype html><html lang=\"pt-BR\"><meta charset=\"utf-8\">
<title>Teste curto — resumo</title>
<style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse;font-size:.85rem}}th,td{{border:1px solid #ccc;padding:.35rem;text-align:left}}th{{background:#eee}}code{{background:#f4f4f4;padding:.15rem .3rem}}</style>
<h1>Teste curto — resumo operacional</h1>
<p>Gerado em <code>{html.escape(payload['generated_at'])}</code>. Runs: {len(rows)}. Status: {html.escape(json.dumps(statuses, ensure_ascii=False))}.</p>
<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></html>"""
    (output_root / "short-test-summary.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/gpu-memory-check"))
    args = parser.parse_args()
    rows = collect(args.output_root.resolve())
    write_report(args.output_root.resolve(), rows)
    print(f"Resumo atualizado: {len(rows)} run(s) em {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
