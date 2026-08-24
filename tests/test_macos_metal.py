from __future__ import annotations

from types import SimpleNamespace

import pytest

from tcc_benchmark import preflight, telemetry


def test_ioreg_parser_extracts_metal_utilization_and_unified_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    output = '''
      "PerformanceStatistics" = {"In use system memory (driver)"=1234,"Alloc system memory"=5678,"Tiler Utilization %"=12,"Renderer Utilization %"=34,"Device Utilization %"=45,"In use system memory"=6789}
      "model" = "Apple M4"
      "gpu-core-count" = 10
    '''
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "/usr/bin/ioreg" if name == "ioreg" else None)
    monkeypatch.setattr(
        telemetry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=output, stderr=""),
    )

    probe = telemetry._AppleMetalProbe()
    devices = probe.sample()

    assert len(devices) == 1
    assert devices[0]["name"] == "Apple M4"
    assert devices[0]["utilization_percent"] == 45
    assert devices[0]["renderer_utilization_percent"] == 34
    assert devices[0]["tiler_utilization_percent"] == 12
    assert devices[0]["memory_kind"] == "shared_unified_driver"
    assert devices[0]["driver_allocated_memory_bytes"] == 5678
    assert devices[0]["system_memory_in_use_bytes"] == 6789
    assert devices[0]["memory_total_bytes"] is None


def test_darwin_gpu_probe_never_falls_back_to_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnavailableMetal:
        available = False

        def sample(self) -> list[dict[str, object]]:
            return []

    monkeypatch.setattr(telemetry.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(telemetry, "_AppleMetalProbe", UnavailableMetal)
    monkeypatch.setattr(telemetry._GpuProbe, "_nvidia_smi_available", staticmethod(lambda: (_ for _ in ()).throw(AssertionError("nvidia-smi"))))

    probe = telemetry._GpuProbe()
    assert probe.backend == "none"
    assert probe.sample() == []


def test_preflight_reports_apple_metal_fields() -> None:
    report = preflight.run_preflight()
    assert "apple_metal" in report
    assert "packages" in report
    if report["platform"]["system"] == "Darwin":
        assert report["platform"]["machine"] == "arm64"
