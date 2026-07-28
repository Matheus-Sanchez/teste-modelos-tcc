from __future__ import annotations

from tcc_benchmark.preflight import run_preflight


def test_preflight_is_serializable(tmp_path) -> None:
    report = run_preflight(output_root=tmp_path)
    assert "platform" in report
    assert "tensorflow" in report
    assert report["disks"]
