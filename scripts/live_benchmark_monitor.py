"""Monitor read-only do estado do benchmark e do hardware.

Funciona no WSL/NVIDIA e no macOS/Apple Metal. O monitor não toca em
checkpoints nem em processos de treinamento.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - monitor remains usable in a minimal environment
    psutil = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _collect(root: Path, phase: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for status_path in sorted(root.glob("*/runs/*/status.json")):
        status = _read_json(status_path)
        rows.append(
            {
                "dataset": status_path.parents[2].name,
                "phase": phase,
                "run_id": str(status.get("run_id", status_path.parent.name)),
                "status": str(status.get("status", "unknown")),
                "detail": "" if status.get("detail") is None else str(status["detail"]),
                "updated_at": str(status.get("updated_at", "")),
            }
        )
    return rows


def _host_snapshot() -> dict[str, Any]:
    if psutil is None:
        return {"cpu_percent": None, "ram_used_bytes": None, "ram_total_bytes": None, "ram_percent": None}
    try:
        memory = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_bytes": int(memory.used),
            "ram_total_bytes": int(memory.total),
            "ram_percent": float(memory.percent),
        }
    except Exception:
        return {"cpu_percent": None, "ram_used_bytes": None, "ram_total_bytes": None, "ram_percent": None}


def _apple_metal_snapshot() -> dict[str, str] | None:
    if platform.system() != "Darwin" or not shutil.which("ioreg"):
        return None
    try:
        output = subprocess.check_output(
            ["ioreg", "-l", "-w", "0", "-r", "-c", "IOAccelerator"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'"PerformanceStatistics"\s*=\s*\{(?P<body>[^}]*)\}', output)
    if not match:
        return None
    body = match.group("body")

    def value(key: str) -> str:
        found = re.search(rf'"{re.escape(key)}"\s*=\s*([0-9]+(?:\.[0-9]+)?)', body)
        return found.group(1) if found else "n/a"

    return {
        "backend": "apple-metal-ioreg",
        "device_utilization_percent": value("Device Utilization %"),
        "renderer_utilization_percent": value("Renderer Utilization %"),
        "tiler_utilization_percent": value("Tiler Utilization %"),
        "alloc_system_memory_bytes": value("Alloc system memory"),
        "in_use_system_memory_bytes": value("In use system memory"),
    }


def _nvidia_snapshot() -> dict[str, str] | None:
    if platform.system() == "Darwin" or not shutil.which("nvidia-smi"):
        return None
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
        "backend": "nvidia-smi",
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
    parser.add_argument("--output-root", type=Path, default=Path("outputs/controlled-augmentation2-mac-m4"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else project / args.output_root
    output_root = output_root.resolve()
    monitoring = output_root / "monitoring"
    monitoring.mkdir(parents=True, exist_ok=True)
    latest = monitoring / "latest.json"
    events = monitoring / "events.jsonl"
    previous: dict[tuple[str, str], str] = {}

    while True:
        rows = (
            _collect(output_root / "batch", "batch")
            + _collect(output_root / "quantization", "quantization")
            + _collect(output_root / "activations", "activation")
        )
        now = datetime.now(timezone.utc).isoformat()
        current = {(row["dataset"], row["run_id"]): row["status"] for row in rows}
        changes = [
            {"timestamp": now, "dataset": key[0], "run_id": key[1], "status": status}
            for key, status in current.items()
            if previous.get(key) != status
        ]
        if changes:
            with events.open("a", encoding="utf-8") as handle:
                for event in changes:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        previous = current
        pipeline = _read_json(output_root / "pipeline-status.json")
        active = next((row for row in rows if row["status"] in {"running", "started"}), None)
        host = _host_snapshot()
        payload = {
            "updated_at": now,
            "platform": platform.platform(),
            "pipeline_status": pipeline.get("status", "not_started"),
            "status_counts": {state: sum(row["status"] == state for row in rows) for state in sorted({row["status"] for row in rows})},
            "runs": rows,
            "gpu": _apple_metal_snapshot() or _nvidia_snapshot(),
            "host": host,
            "active_run": active,
        }
        _write(latest, payload)
        counts = payload["status_counts"]
        print(
            f"[{now}] phase={(active or {}).get('phase', 'queue')} "
            f"dataset={(active or {}).get('dataset', 'none')} "
            f"run={(active or {}).get('run_id', 'none')} "
            f"pipeline={payload['pipeline_status']} runs={len(rows)} status={counts} "
            f"metal={payload['gpu']} ram={host.get('ram_percent')}% cpu={host.get('cpu_percent')}%",
            flush=True,
        )
        time.sleep(max(5.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
