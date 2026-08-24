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
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "build_info": getattr(tf.sysconfig, "get_build_info", lambda: {})(),
        }
    except Exception as exc:  # TensorFlow can be installed but fail to load CUDA
        return {"installed": True, "version": version, "gpus": [], "error": repr(exc)}


def _apple_metal_probe() -> dict[str, Any]:
    """Capture read-only Metal facts available on Apple Silicon macOS."""

    if platform.system() != "Darwin":
        return {"available": False, "backend": None, "reason": "não é macOS"}
    result: dict[str, Any] = {"available": False, "backend": "apple-metal-ioreg"}
    ioreg = shutil.which("ioreg")
    if ioreg:
        try:
            probe = subprocess.run(
                [ioreg, "-l", "-w", "0", "-r", "-c", "IOAccelerator"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            result["ioreg_returncode"] = probe.returncode
            result["available"] = probe.returncode == 0 and "PerformanceStatistics" in probe.stdout
        except (OSError, subprocess.SubprocessError) as exc:
            result["error"] = repr(exc)
    system_profiler = shutil.which("system_profiler")
    if system_profiler:
        try:
            display = subprocess.run(
                [system_profiler, "SPDisplaysDataType"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            result["display_summary"] = display.stdout[:4000]
            result["metal_support"] = "Metal Support:" in display.stdout
        except (OSError, subprocess.SubprocessError):
            result["metal_support"] = False
    return result


def run_preflight(
    *,
    output_root: str | Path | None = None,
    data_paths: Iterable[str | Path] = (),
    require_tensorflow: bool = False,
    require_gpu: bool = False,
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
    apple_metal = _apple_metal_probe()
    errors: list[str] = []
    if require_tensorflow and not tensorflow.get("installed"):
        errors.append("TensorFlow é obrigatório para treinamento, mas não está instalado.")
    if require_tensorflow and tensorflow.get("installed") and tensorflow.get("error"):
        errors.append(f"TensorFlow não carregou corretamente: {tensorflow['error']}")
    if require_gpu:
        gpu_devices = tensorflow.get("gpus", []) if isinstance(tensorflow, dict) else []
        if not gpu_devices:
            errors.append("Nenhum dispositivo GPU TensorFlow foi detectado.")
        if platform.system() == "Darwin" and not apple_metal.get("available", False):
            errors.append("O dispositivo Metal não foi detectado via IOAccelerator/ioreg.")
        if platform.system() == "Darwin" and _package_version("tensorflow-metal") is None:
            errors.append("tensorflow-metal não está instalado no ambiente ativo.")

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
        "apple_metal": apple_metal,
        "nvidia_smi": _nvidia_smi(),
        "packages": {
            name: _package_version(name)
            for name in ("tensorflow", "tensorflow-metal", "numpy", "pandas", "scikit-learn", "psutil", "pynvml")
        },
    }


def format_preflight(report: dict[str, Any]) -> str:
    """Stable human-readable JSON used by the CLI."""
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)
