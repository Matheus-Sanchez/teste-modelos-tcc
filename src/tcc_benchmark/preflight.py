"""Checagens portáveis de ambiente antes de auditoria ou treinamento."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi não encontrado"}
    fields = "name,driver_version,memory.total,memory.used,utilization.gpu"
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc)}
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mib": _maybe_number(parts[2]),
                    "memory_used_mib": _maybe_number(parts[3]),
                    "utilization_percent": _maybe_number(parts[4]),
                }
            )
    return {"available": True, "gpus": rows}


def _maybe_number(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _tensorflow_probe() -> dict[str, Any]:
    version = _package_version("tensorflow")
    if version is None:
        return {"installed": False, "version": None, "gpus": [], "error": "tensorflow não instalado"}
    try:
        import tensorflow as tf  # imported only during preflight/training

        return {
            "installed": True,
            "version": getattr(tf, "__version__", version),
            "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "build_info": getattr(tf.sysconfig, "get_build_info", lambda: {})(),
        }
    except Exception as exc:  # TensorFlow can be installed but fail to load CUDA
        return {"installed": True, "version": version, "gpus": [], "error": repr(exc)}


def run_preflight(
    *,
    output_root: str | Path | None = None,
    data_paths: Iterable[str | Path] = (),
    require_tensorflow: bool = False,
) -> dict[str, Any]:
    """Return a serializable preflight report; never changes machine state."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        cpu = {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "memory_total_bytes": memory.total,
            "memory_available_bytes": memory.available,
        }
    except Exception as exc:  # pragma: no cover - psutil is declared dependency
        cpu = {"error": repr(exc)}

    paths = [Path(path).expanduser().resolve() for path in data_paths]
    if output_root is not None:
        paths.append(Path(output_root).expanduser().resolve())
    disks: list[dict[str, Any]] = []
    for path in dict.fromkeys(paths):
        try:
            candidate = path if path.exists() else path.parent
            usage = shutil.disk_usage(candidate)
            disks.append({"path": str(path), "exists": path.exists(), "total_bytes": usage.total, "free_bytes": usage.free})
        except OSError as exc:
            disks.append({"path": str(path), "exists": path.exists(), "error": repr(exc)})

    tensorflow = _tensorflow_probe()
    errors: list[str] = []
    if require_tensorflow and not tensorflow.get("installed"):
        errors.append("TensorFlow é obrigatório para treinamento, mas não está instalado.")
    if require_tensorflow and tensorflow.get("installed") and tensorflow.get("error"):
        errors.append(f"TensorFlow não carregou corretamente: {tensorflow['error']}")

    return {
        "ok": not errors,
        "errors": errors,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
            "cwd": os.getcwd(),
        },
        "cpu": cpu,
        "disks": disks,
        "tensorflow": tensorflow,
        "nvidia_smi": _nvidia_smi(),
        "packages": {name: _package_version(name) for name in ("tensorflow", "numpy", "pandas", "scikit-learn", "psutil", "pynvml")},
    }


def format_preflight(report: dict[str, Any]) -> str:
    """Stable human-readable JSON used by the CLI."""
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)
