"""Chain the short GPU check, its summary, and the final 100-epoch matrix.

Run through ``scripts/wsl-gpu-env.sh``.  The supervisor never starts the final
matrix if any short-test run is failed, interrupted or missing; this keeps a
data/VRAM failure visible instead of silently promoting an unsafe setup.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DATASETS = (
    "mnist",
    "fashion_mnist",
    "kmnist",
    "emnist_balanced",
    "cifar10",
    "cifar100_coarse",
    "svhn",
    "gtsrb",
    "fer2013",
)


def _statuses(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for dataset in DATASETS:
        status_path = root / dataset / "runs" / f"{dataset}__unit_interval__all_raw__seed-42" / "status.json"
        try:
            result[dataset] = str(json.loads(status_path.read_text(encoding="utf-8")).get("status", "unknown"))
        except (OSError, json.JSONDecodeError):
            result[dataset] = "missing"
    return result


def _run(command: list[str], *, cwd: Path) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    project = args.project_root.resolve()
    short_root = project / "artifacts" / "gpu-memory-check"

    # The MNIST resume is already active when this supervisor is launched.
    # Wait for it so that no second process touches the same run directory.
    while True:
        mnist = _statuses(short_root)["mnist"]
        print(f"MNIST curto: {mnist}", flush=True)
        subprocess.run(
            [sys.executable, "scripts/summarize_short_test.py", "--output-root", str(short_root)],
            cwd=project,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if mnist != "running":
            break
        time.sleep(max(float(args.poll_seconds), 5.0))
    if mnist != "completed":
        _run([sys.executable, "scripts/summarize_short_test.py", "--output-root", str(short_root)], cwd=project)
        print("Teste curto não foi concluído; matriz de 100 épocas não será iniciada.", file=sys.stderr)
        return 2

    short_command = ["tcc-benchmark", "resume", "--all", "--suite", "configs/gpu-memory-check.yaml"]
    # If every durable short-run state is already complete, loading every
    # source dataset again merely to skip completed runs is expensive.
    # The statuses are the same source of truth used by the promotion gate
    # below, so skip that no-op resume pass and move directly to validation.
    statuses_before_resume = _statuses(short_root)
    if all(state == "completed" for state in statuses_before_resume.values()):
        print("Teste curto já concluído; pulando resume redundante.", flush=True)
        short_returncode = 0
    else:
        print("$", " ".join(short_command), flush=True)
        short_returncode = subprocess.run(short_command, cwd=project, check=False).returncode
    _run([sys.executable, "scripts/summarize_short_test.py", "--output-root", str(short_root)], cwd=project)
    statuses = _statuses(short_root)
    incomplete = {name: state for name, state in statuses.items() if state != "completed"}
    if short_returncode or incomplete:
        print(f"Teste curto incompleto; matriz de 100 épocas não será iniciada: {incomplete}", file=sys.stderr)
        return 3

    # Always resume the final queue: a controlled restart must restore an
    # interrupted cell instead of silently skipping it and advancing the grid.
    _run(["tcc-benchmark", "resume", "--all", "--suite", "configs/full-100-epochs.yaml"], cwd=project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
