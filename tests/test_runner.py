from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tcc_benchmark.config import DatasetEntry, SuiteSettings
from tcc_benchmark.runner import DatasetMaterials, _completed_artifacts_valid, _execution_action, _run_cell, _subset_for_smoke
from tcc_benchmark.state import RunPaths, set_run_status


def _materials() -> DatasetMaterials:
    return DatasetMaterials(
        samples=[np.zeros((8, 8, 1), dtype=np.uint8) for _ in range(30)],
        labels=np.repeat(np.arange(3), 10),
        channels=1,
        class_names=("zero", "one", "two"),
        source_manifest={"name": "mnist", "splits": {"all": 30}},
    )


def test_dry_run_creates_stable_manifest_without_tensorflow(tmp_path: Path) -> None:
    settings = SuiteSettings(output_root=tmp_path / "artifacts", seeds=(42,))
    entry = DatasetEntry(name="mnist", adapter="mnist", root=tmp_path / "data")
    outcome = _run_cell(
        settings=settings,
        entry=entry,
        materials=_materials(),
        normalization="unit_interval",
        balance_mode="all_raw",
        seed=42,
        resume=False,
        rerun_failed=False,
        dry_run=True,
    )

    assert outcome.action == "planned"
    run_dir = tmp_path / "artifacts" / "mnist" / "runs" / outcome.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "pending"
    assert manifest["split_fingerprint"]
    assert manifest["config"]["protocol"]["training_from_scratch"] is True
    assert (run_dir / "status.json").exists()


def test_resume_policy_requires_an_explicit_resume_after_interruption(tmp_path: Path) -> None:
    settings = SuiteSettings(output_root=tmp_path / "artifacts", seeds=(42,))
    entry = DatasetEntry(name="mnist", adapter="mnist", root=tmp_path / "data")
    outcome = _run_cell(
        settings=settings,
        entry=entry,
        materials=_materials(),
        normalization="unit_interval",
        balance_mode="all_raw",
        seed=42,
        resume=False,
        rerun_failed=False,
        dry_run=True,
    )
    run_dir = tmp_path / "artifacts" / "mnist" / "runs" / outcome.run_id
    set_run_status(run_dir, "running")
    set_run_status(run_dir, "interrupted")

    assert _execution_action("interrupted", resume=False, rerun_failed=False, dry_run=False) == "skip_needs_resume"
    assert _execution_action("interrupted", resume=True, rerun_failed=False, dry_run=False) == "execute"


def test_completed_run_requires_checkpoint_metrics_and_telemetry(tmp_path: Path) -> None:
    paths = RunPaths.from_root(tmp_path / "run").ensure()
    for path in (
        paths.manifest,
        paths.status,
        paths.artifacts / "test_metrics.json",
        paths.logs / "training_summary.json",
        paths.telemetry / "environment.json",
        paths.telemetry / "samples.csv",
        paths.telemetry / "summary.json",
    ):
        path.write_text("{}", encoding="utf-8")
    (paths.checkpoints / "best.keras").write_bytes(b"checkpoint")

    assert _completed_artifacts_valid(paths)[0] is True
    (paths.telemetry / "summary.json").unlink()
    valid, detail = _completed_artifacts_valid(paths)
    assert valid is False
    assert "telemetry/summary.json" in detail


def test_smoke_subset_is_balanced_and_deterministic() -> None:
    first = _subset_for_smoke(_materials(), examples_per_class=3, seed=42)
    second = _subset_for_smoke(_materials(), examples_per_class=3, seed=42)

    assert len(first.labels) == 9
    assert np.array_equal(first.labels, second.labels)
    assert np.bincount(first.labels, minlength=3).tolist() == [3, 3, 3]
    assert first.source_manifest["smoke_subset"]["selection_fingerprint"] == second.source_manifest["smoke_subset"]["selection_fingerprint"]
