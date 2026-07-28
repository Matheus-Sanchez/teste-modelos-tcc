"""Atualiza um estado operacional durante benchmarks executados em WSL.

Não toca em checkpoints, dados ou processos de treino.  O monitor apenas lê os
artefatos duráveis, reescreve o resumo curto e registra mudanças de status em
``artifacts/monitoring``.  Ele é apropriado para permanecer em segundo plano.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _collect(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for status_path in sorted(root.glob("*/runs/*/status.json")):
        status = _read_json(status_path)
        rows.append(
            {
                "dataset": status_path.parents[2].name,
                "run_id": str(status.get("run_id", status_path.parent.name)),
                "status": str(status.get("status", "unknown")),
                "detail": "" if status.get("detail") is None else str(status["detail"]),
                "updated_at": str(status.get("updated_at", "")),
            }
        )
    return rows


def _gpu_snapshot() -> dict[str, str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip().splitlines()[0]
        name, used, total, utilization, temperature = [part.strip() for part in line.split(",", maxsplit=4)]
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None
    return {
        "name": name,
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_percent": utilization,
        "temperature_c": temperature,
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    project = args.project_root.resolve()
    monitoring = project / "artifacts" / "monitoring"
    monitoring.mkdir(parents=True, exist_ok=True)
    latest = monitoring / "latest.json"
    events = monitoring / "events.jsonl"
    previous: dict[tuple[str, str], str] = {}

    while True:
        short_root = project / "artifacts" / "gpu-memory-check"
        full_root = project / "artifacts" / "full-100-epochs"
        rows = _collect(short_root) + _collect(full_root)
        now = datetime.now(timezone.utc).isoformat()
        changes = []
        current = {(row["dataset"], row["run_id"]): row["status"] for row in rows}
        for key, state in current.items():
            if previous.get(key) != state:
                changes.append({"timestamp": now, "dataset": key[0], "run_id": key[1], "status": state})
        if changes:
            with events.open("a", encoding="utf-8") as handle:
                for event in changes:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        previous = current
        payload = {
            "updated_at": now,
            "status_counts": {state: sum(row["status"] == state for row in rows) for state in sorted({row["status"] for row in rows})},
            "runs": rows,
            "gpu": _gpu_snapshot(),
        }
        _write(latest, payload)
        subprocess.run(
            [sys.executable, "scripts/summarize_short_test.py", "--output-root", str(short_root)],
            cwd=project,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(max(5.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
