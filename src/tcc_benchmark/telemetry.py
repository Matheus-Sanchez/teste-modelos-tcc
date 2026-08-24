"""Coleta leve de uso de hardware durante uma execução de treinamento.

O coletor funciona sem GPU, NVML, TensorFlow ou até mesmo ``psutil``.  Dados
indisponíveis são persistidos como valores nulos, em vez de interromper uma run
longa por um detalhe de observabilidade.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
import platform
import re
import statistics
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .state import append_csv, atomic_write_json, canonical_json, utc_now

try:  # Optional in minimal installations / static tests.
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - depends on selected requirements profile
    psutil = None


SAMPLE_FIELDS: tuple[str, ...] = (
    "timestamp",
    "elapsed_seconds",
    "event",
    "metadata_json",
    "cpu_percent",
    "process_cpu_percent",
    "cpu_count_logical",
    "ram_total_bytes",
    "ram_available_bytes",
    "ram_used_bytes",
    "ram_percent",
    "process_rss_bytes",
    "process_vms_bytes",
    "process_read_bytes",
    "process_write_bytes",
    "disk_data_total_bytes",
    "disk_data_free_bytes",
    "disk_data_used_bytes",
    "disk_data_percent",
    "disk_output_total_bytes",
    "disk_output_free_bytes",
    "disk_output_used_bytes",
    "disk_output_percent",
    "disks_json",
    "gpu_backend",
    "gpu_count",
    "gpu_utilization_percent",
    "gpu_renderer_utilization_percent",
    "gpu_tiler_utilization_percent",
    "gpu_memory_used_bytes",
    "gpu_memory_total_bytes",
    "gpu_memory_percent",
    "gpu_memory_kind",
    "gpu_driver_allocated_memory_bytes",
    "gpu_system_memory_in_use_bytes",
    "gpu_temperature_c",
    "gpu_power_w",
    "thermal_pressure",
    "gpus_json",
)

_NUMERIC_SAMPLE_FIELDS = tuple(
    field
    for field in SAMPLE_FIELDS
    if field.endswith(("_percent", "_bytes", "_c", "_w", "_seconds")) or field == "gpu_count"
)


def _apple_runtime_metadata() -> dict[str, str | None]:
    """Return non-privileged macOS/Metal version facts for environment.json."""

    if platform.system() != "Darwin":
        return {"macos_version": None, "metal_version": None}
    macos_version = platform.mac_ver()[0] or None
    metal_version: str | None = None
    executable = shutil.which("system_profiler")
    if executable:
        try:
            result = subprocess.run(
                [executable, "SPDisplaysDataType"], capture_output=True, text=True, timeout=10, check=False
            )
            match = re.search(r"Metal Support:\s*([^\n]+)", result.stdout)
            metal_version = match.group(1).strip() if match else None
        except (OSError, subprocess.SubprocessError):
            pass
    return {"macos_version": macos_version, "metal_version": metal_version}


def _optional_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(values: Sequence[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if _safe_float(value) is not None]
    return statistics.fmean(numeric) if numeric else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


class _GpuProbe:
    """Probe NVIDIA GPUs or the Apple Metal device without failing training."""

    def __init__(self) -> None:
        self.backend = "none"
        self._nvml: Any | None = None
        self._nvml_initialized = False
        self._apple_metal = _AppleMetalProbe() if platform.system() == "Darwin" else None
        if self._apple_metal is not None:
            # macOS must never probe the WSL/NVIDIA stack.  A missing or
            # inaccessible IOAccelerator simply becomes an explicit fallback.
            self.backend = "apple-metal-ioreg" if self._apple_metal.available else "none"
            return
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_initialized = True
            self.backend = "pynvml"
        except Exception:
            self._nvml = None
            self._nvml_initialized = False
            if self._nvidia_smi_available():
                self.backend = "nvidia-smi"

    @staticmethod
    def _nvidia_smi_available() -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    @staticmethod
    def _decode_name(value: Any) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)

    def sample(self) -> list[dict[str, Any]]:
        if self.backend == "apple-metal-ioreg" and self._apple_metal is not None:
            return self._apple_metal.sample()
        if self.backend == "pynvml" and self._nvml is not None:
            try:
                return self._sample_nvml()
            except Exception:
                # NVML can disappear after driver resets. Continue with the
                # command-line fallback rather than aborting training.
                self.backend = "nvidia-smi" if self._nvidia_smi_available() else "none"
        if self.backend == "nvidia-smi":
            return self._sample_nvidia_smi()
        return []

    def _sample_nvml(self) -> list[dict[str, Any]]:
        assert self._nvml is not None
        devices: list[dict[str, Any]] = []
        count = int(self._nvml.nvmlDeviceGetCount())
        for index in range(count):
            handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
            memory = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            temperature: float | None = None
            power_w: float | None = None
            try:
                temperature = float(
                    self._nvml.nvmlDeviceGetTemperature(handle, self._nvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass
            try:
                power_w = float(self._nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            except Exception:
                pass
            devices.append(
                {
                    "index": index,
                    "name": self._decode_name(self._nvml.nvmlDeviceGetName(handle)),
                    "utilization_percent": _safe_float(getattr(utilization, "gpu", None)),
                    "memory_used_bytes": int(getattr(memory, "used", 0)),
                    "memory_total_bytes": int(getattr(memory, "total", 0)),
                    "temperature_c": temperature,
                    "power_w": power_w,
                }
            )
        return devices

    def _sample_nvidia_smi(self) -> list[dict[str, Any]]:
        fields = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            self.backend = "none"
            return []
        if result.returncode != 0:
            return []
        devices: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 7:
                continue
            try:
                index = int(parts[0])
            except ValueError:
                continue
            used_mib = _safe_float(parts[3])
            total_mib = _safe_float(parts[4])
            devices.append(
                {
                    "index": index,
                    "name": parts[1],
                    "utilization_percent": _safe_float(parts[2]),
                    "memory_used_bytes": None if used_mib is None else int(used_mib * 1024**2),
                    "memory_total_bytes": None if total_mib is None else int(total_mib * 1024**2),
                    "temperature_c": _safe_float(parts[5]),
                    "power_w": _safe_float(parts[6]),
                }
            )
        return devices

    def close(self) -> None:
        if self._nvml_initialized and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml_initialized = False


class _AppleMetalProbe:
    """Read public IOAccelerator counters exposed by Apple Silicon macOS.

    Apple Silicon uses unified memory, so the probe deliberately reports driver
    and system-memory counters instead of pretending that a discrete VRAM
    capacity exists.  ``ioreg`` is read-only and does not require sudo.
    """

    _COMMAND = ("ioreg", "-l", "-w", "0", "-r", "-c", "IOAccelerator")

    def __init__(self) -> None:
        self.available = bool(shutil.which("ioreg"))

    @staticmethod
    def _number(text: str, key: str) -> float | None:
        match = re.search(rf'"{re.escape(key)}"\s*=\s*([0-9]+(?:\.[0-9]+)?)', text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def sample(self) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            result = subprocess.run(
                list(self._COMMAND), capture_output=True, text=True, timeout=4, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if result.returncode != 0:
            return []
        statistics_block = re.search(r'"PerformanceStatistics"\s*=\s*\{(?P<body>[^}]*)\}', result.stdout)
        if statistics_block is None:
            return []
        body = statistics_block.group("body")
        device = self._number(body, "Device Utilization %")
        renderer = self._number(body, "Renderer Utilization %")
        tiler = self._number(body, "Tiler Utilization %")
        allocated = self._number(body, "Alloc system memory")
        in_use = self._number(body, "In use system memory")
        driver_in_use = self._number(body, "In use system memory (driver)")
        if all(value is None for value in (device, renderer, tiler, allocated, in_use, driver_in_use)):
            return []
        model_match = re.search(r'"model"\s*=\s*"([^"]+)"', result.stdout)
        core_count = self._number(result.stdout, "gpu-core-count")
        return [
            {
                "index": 0,
                "name": model_match.group(1) if model_match else "Apple Metal GPU",
                "utilization_percent": device,
                "renderer_utilization_percent": renderer,
                "tiler_utilization_percent": tiler,
                "memory_used_bytes": int(in_use) if in_use is not None else None,
                "memory_total_bytes": None,
                "memory_kind": "shared_unified_driver",
                "driver_allocated_memory_bytes": int(allocated) if allocated is not None else None,
                "system_memory_in_use_bytes": int(in_use) if in_use is not None else None,
                "driver_memory_in_use_bytes": int(driver_in_use) if driver_in_use is not None else None,
                "gpu_core_count": int(core_count) if core_count is not None else None,
                "temperature_c": None,
                "power_w": None,
            }
        ]


def _disk_usage(path: Path) -> dict[str, float | int | None]:
    if psutil is None:
        return {"total_bytes": None, "free_bytes": None, "used_bytes": None, "percent": None}
    try:
        usage = psutil.disk_usage(str(path))
        return {
            "total_bytes": int(usage.total),
            "free_bytes": int(usage.free),
            "used_bytes": int(usage.used),
            "percent": _safe_float(usage.percent),
        }
    except (OSError, ValueError):
        return {"total_bytes": None, "free_bytes": None, "used_bytes": None, "percent": None}


def _thermal_pressure() -> str | None:
    """Return macOS thermal warnings without requiring privileged samplers."""

    if platform.system() != "Darwin" or not shutil.which("pmset"):
        return None
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=3, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = " ".join(result.stdout.split())
    if not text:
        return None
    if "No thermal warning level has been recorded" in text:
        return "nominal"
    if "critical" in text.lower():
        return "critical"
    if "warning" in text.lower():
        return "warning"
    return text[:240]


def capture_environment(*, disk_paths: Mapping[str, str | Path] | None = None) -> dict[str, Any]:
    """Capture reproducibility context without leaking user environment variables."""

    probe = _GpuProbe()
    try:
        gpu_devices = probe.sample()
        hardware: dict[str, Any] = {
            "cpu_count_logical": os.cpu_count(),
            "gpu_backend": probe.backend,
            "gpus": gpu_devices,
            "disks": {str(name): _disk_usage(Path(path)) for name, path in (disk_paths or {}).items()},
        }
        if psutil is not None:
            try:
                memory = psutil.virtual_memory()
                hardware["ram_total_bytes"] = int(memory.total)
            except Exception:
                hardware["ram_total_bytes"] = None
        hardware.update(_apple_runtime_metadata())
        return {
            "captured_at": utc_now(),
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "packages": {
                name: _optional_version(distribution)
                for name, distribution in {
                    "tensorflow": "tensorflow",
                    "tensorflow_metal": "tensorflow-metal",
                    "numpy": "numpy",
                    "scikit_learn": "scikit-learn",
                    "psutil": "psutil",
                    "pynvml": "nvidia-ml-py",
                }.items()
            },
            "hardware": hardware,
        }
    finally:
        probe.close()


class TelemetrySampler:
    """Sample CPU/RAM/GPU/disk/process usage in a durable background stream.

    Parameters
    ----------
    output_dir:
        Usually ``RunPaths.telemetry``.  The sampler writes ``samples.csv`` and
        ``summary.json`` there.
    disk_paths:
        Optional explicit paths to account as ``data`` and ``output``.  The
        output directory is always accounted for when no output path is given.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        interval_seconds: float = 5.0,
        disk_paths: Mapping[str, str | Path] | None = None,
        data_path: str | Path | None = None,
        process_id: int | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds deve ser positivo")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.output_dir / "samples.csv"
        self.summary_path = self.output_dir / "summary.json"
        self.environment_path = self.output_dir / "environment.json"
        self.interval_seconds = float(interval_seconds)
        self.process_id = int(process_id or os.getpid())
        supplied_paths = {str(name): Path(path) for name, path in (disk_paths or {}).items()}
        supplied_paths.setdefault("output", self.output_dir)
        if data_path is not None:
            supplied_paths.setdefault("data", Path(data_path))
        self.disk_paths = supplied_paths
        self._gpu = _GpuProbe()
        self._process = None
        if psutil is not None:
            try:
                self._process = psutil.Process(self.process_id)
                self._process.cpu_percent(interval=None)  # Prime non-blocking process CPU sampling.
                psutil.cpu_percent(interval=None)
            except Exception:
                self._process = None
        self._start_monotonic: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []
        self._samples_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def start(self) -> "TelemetrySampler":
        """Start background sampling and capture a first ``train_start`` sample."""

        with self._lifecycle_lock:
            if self.running:
                return self
            self._start_monotonic = time.monotonic()
            self._stop_event.clear()
            atomic_write_json(self.environment_path, capture_environment(disk_paths=self.disk_paths))
            self.capture("train_start")
            self._thread = threading.Thread(target=self._loop, name="tcc-telemetry", daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.capture("periodic")
            except Exception:
                # Telemetry must never terminate model training.  The next
                # interval may succeed (for example after a GPU driver reset).
                continue

    def _base_process_sample(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cpu_percent": None,
            "process_cpu_percent": None,
            "cpu_count_logical": os.cpu_count(),
            "ram_total_bytes": None,
            "ram_available_bytes": None,
            "ram_used_bytes": None,
            "ram_percent": None,
            "process_rss_bytes": None,
            "process_vms_bytes": None,
            "process_read_bytes": None,
            "process_write_bytes": None,
        }
        if psutil is None:
            return result
        try:
            result["cpu_percent"] = _safe_float(psutil.cpu_percent(interval=None))
        except Exception:
            pass
        try:
            memory = psutil.virtual_memory()
            result.update(
                {
                    "ram_total_bytes": int(memory.total),
                    "ram_available_bytes": int(memory.available),
                    "ram_used_bytes": int(memory.used),
                    "ram_percent": _safe_float(memory.percent),
                }
            )
        except Exception:
            pass
        if self._process is not None:
            try:
                result["process_cpu_percent"] = _safe_float(self._process.cpu_percent(interval=None))
            except Exception:
                pass
            try:
                memory_info = self._process.memory_info()
                result["process_rss_bytes"] = int(memory_info.rss)
                result["process_vms_bytes"] = int(memory_info.vms)
            except Exception:
                pass
            try:
                io = self._process.io_counters()
                result["process_read_bytes"] = int(io.read_bytes)
                result["process_write_bytes"] = int(io.write_bytes)
            except Exception:
                pass
        return result

    def _sample(self, event: str, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        start = self._start_monotonic or time.monotonic()
        sample: dict[str, Any] = {
            "timestamp": utc_now(),
            "elapsed_seconds": round(time.monotonic() - start, 6),
            "event": str(event),
            "metadata_json": canonical_json(metadata or {}),
        }
        sample.update(self._base_process_sample())

        disk_values = {name: _disk_usage(path) for name, path in self.disk_paths.items()}
        sample["disks_json"] = canonical_json(disk_values)
        for name in ("data", "output"):
            disk = disk_values.get(name, {})
            sample[f"disk_{name}_total_bytes"] = disk.get("total_bytes")
            sample[f"disk_{name}_free_bytes"] = disk.get("free_bytes")
            sample[f"disk_{name}_used_bytes"] = disk.get("used_bytes")
            sample[f"disk_{name}_percent"] = disk.get("percent")

        gpus = self._gpu.sample()
        sample["gpu_backend"] = self._gpu.backend
        sample["gpu_count"] = len(gpus)
        sample["gpu_utilization_percent"] = _mean([gpu.get("utilization_percent") for gpu in gpus])
        sample["gpu_renderer_utilization_percent"] = _mean(
            [gpu.get("renderer_utilization_percent") for gpu in gpus]
        )
        sample["gpu_tiler_utilization_percent"] = _mean([gpu.get("tiler_utilization_percent") for gpu in gpus])
        used_values = [int(gpu["memory_used_bytes"]) for gpu in gpus if gpu.get("memory_used_bytes") is not None]
        total_values = [int(gpu["memory_total_bytes"]) for gpu in gpus if gpu.get("memory_total_bytes") is not None]
        sample["gpu_memory_used_bytes"] = sum(used_values) if used_values else None
        sample["gpu_memory_total_bytes"] = sum(total_values) if total_values else None
        total = sample["gpu_memory_total_bytes"]
        used = sample["gpu_memory_used_bytes"]
        sample["gpu_memory_percent"] = None if not total or used is None else 100.0 * float(used) / float(total)
        sample["gpu_memory_kind"] = next(
            (str(gpu.get("memory_kind")) for gpu in gpus if gpu.get("memory_kind")), None
        )
        driver_values = [
            int(gpu["driver_allocated_memory_bytes"])
            for gpu in gpus
            if gpu.get("driver_allocated_memory_bytes") is not None
        ]
        system_values = [
            int(gpu["system_memory_in_use_bytes"])
            for gpu in gpus
            if gpu.get("system_memory_in_use_bytes") is not None
        ]
        sample["gpu_driver_allocated_memory_bytes"] = sum(driver_values) if driver_values else None
        sample["gpu_system_memory_in_use_bytes"] = sum(system_values) if system_values else None
        sample["gpu_temperature_c"] = _mean([gpu.get("temperature_c") for gpu in gpus])
        sample["gpu_power_w"] = _mean([gpu.get("power_w") for gpu in gpus])
        sample["thermal_pressure"] = _thermal_pressure()
        sample["gpus_json"] = canonical_json(gpus)
        return sample

    def capture(self, event: str = "manual", metadata: Mapping[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
        """Capture a sample now; use for epoch boundaries in addition to 5s ticks."""

        merged_metadata = dict(metadata or {})
        merged_metadata.update(extra)
        sample = self._sample(event, merged_metadata)
        append_csv(self.samples_path, [sample], fieldnames=SAMPLE_FIELDS)
        with self._samples_lock:
            self._samples.append(sample)
        return sample

    def summary(self) -> dict[str, Any]:
        """Return aggregate means, peaks and p50/p95 for this process lifetime."""

        with self._samples_lock:
            samples = list(self._samples)
        metrics: dict[str, dict[str, float | int | None]] = {}
        for field in _NUMERIC_SAMPLE_FIELDS:
            numeric = [_safe_float(sample.get(field)) for sample in samples]
            values = [value for value in numeric if value is not None]
            if not values:
                continue
            metrics[field] = {
                "count": len(values),
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "p50": _percentile(values, 50),
                "p95": _percentile(values, 95),
            }
        event_counts: dict[str, int] = {}
        for sample in samples:
            event = str(sample.get("event", "unknown"))
            event_counts[event] = event_counts.get(event, 0) + 1
        categorical: dict[str, dict[str, Any]] = {}
        for field in ("gpu_memory_kind", "thermal_pressure"):
            values = [str(sample[field]) for sample in samples if sample.get(field) not in {None, ""}]
            if values:
                categorical[field] = {
                    "counts": dict(sorted(Counter(values).items())),
                    "current": values[-1],
                }
        elapsed = None
        if self._start_monotonic is not None:
            elapsed = time.monotonic() - self._start_monotonic
        return {
            "schema_version": 1,
            "generated_at": utc_now(),
            "sample_count": len(samples),
            "interval_seconds": self.interval_seconds,
            "elapsed_seconds": elapsed,
            "gpu_backend": self._gpu.backend,
            "event_counts": event_counts,
            "categorical": categorical,
            "metrics": metrics,
        }

    def stop(self, *, final_event: str = "train_end") -> dict[str, Any]:
        """Stop sampling, persist final aggregate JSON, and release NVML cleanly."""

        with self._lifecycle_lock:
            was_running = self.running
            self._stop_event.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self.interval_seconds + 1.0))
        if was_running:
            try:
                self.capture(final_event)
            except Exception:
                pass
        summary = self.summary()
        atomic_write_json(self.summary_path, summary)
        self._gpu.close()
        return summary


def make_keras_telemetry_callback(sampler: TelemetrySampler, *, manage_lifecycle: bool = True) -> Any:
    """Return a Keras callback when TensorFlow is installed, otherwise a duck type.

    The callback records explicit epoch samples containing duration and the
    scalar logs provided by Keras.  By default it owns sampler start/stop so
    interrupted ``fit`` calls still get a persisted telemetry summary through
    ``on_train_end``.  With ``manage_lifecycle=False`` the runner owns those
    boundaries; this records a ``fit_end`` marker without stopping sampling so
    final evaluation remains in the same hardware trace.
    """

    try:
        import tensorflow as tf  # type: ignore

        base: type[Any] = tf.keras.callbacks.Callback
    except Exception:  # pragma: no cover - exercised on minimal Windows tooling
        base = object

    class _TelemetryCallback(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self._epoch_start: float | None = None

        def on_train_begin(self, logs: Mapping[str, Any] | None = None) -> None:
            if manage_lifecycle:
                sampler.start()
            else:
                sampler.capture("fit_start")

        def on_epoch_begin(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
            self._epoch_start = time.perf_counter()

        def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
            safe_logs = {
                str(key): _safe_float(value) if _safe_float(value) is not None else str(value)
                for key, value in (logs or {}).items()
            }
            duration = None if self._epoch_start is None else time.perf_counter() - self._epoch_start
            sampler.capture("epoch_end", epoch=int(epoch) + 1, epoch_seconds=duration, logs=safe_logs)

        def on_train_end(self, logs: Mapping[str, Any] | None = None) -> None:
            if manage_lifecycle:
                sampler.stop()
            else:
                safe_logs = {
                    str(key): _safe_float(value) if _safe_float(value) is not None else str(value)
                    for key, value in (logs or {}).items()
                }
                sampler.capture("fit_end", logs=safe_logs)

    return _TelemetryCallback()


__all__ = [
    "SAMPLE_FIELDS",
    "TelemetrySampler",
    "capture_environment",
    "make_keras_telemetry_callback",
]
