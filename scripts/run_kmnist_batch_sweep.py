"""Executa uma varredura retomável de batch sizes do KMNIST, sequencialmente.

Cada batch recebe uma cópia da suíte e um ``output_root`` próprio porque o ID
durável da run do benchmark não inclui ``batch_size``. A cópia só difere da
configuração-base em ``training.batch_size`` e no local dos artefatos.

Execute dentro do ambiente WSL com CUDA, por exemplo:

    PYTHON_BIN=/home/msduda/.venvs/tcc-benchmark/bin/python \
      scripts/wsl-gpu-env.sh /home/msduda/.venvs/tcc-benchmark/bin/python \
      scripts/run_kmnist_batch_sweep.py --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_SIZES = tuple(range(16, 513, 16))
RUN_ID = "kmnist__unit_interval__all_raw__seed-42"
NEUTRAL_AUGMENTATION: dict[str, float | bool] = {
    "flip_lr": False,
    "brightness_delta": 0.0,
    "contrast_lower": 1.0,
    "contrast_upper": 1.0,
    "translate_frac": 0.0,
    "zoom_min": 1.0,
    "zoom_max": 1.0,
    "noise_std": 0.0,
    "cutout_prob": 0.0,
    "cutout_max_frac": 0.0,
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_from_project(value: Path) -> Path:
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return DEFAULT_BATCH_SIZES
    try:
        sizes = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batch-sizes deve ser uma lista de inteiros separados por vírgula.") from exc
    if not sizes:
        raise argparse.ArgumentTypeError("--batch-sizes não pode ser vazio.")
    invalid = [size for size in sizes if size < 16 or size > 512 or size % 16]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Cada batch deve ser múltiplo de 16 entre 16 e 512; inválidos: {invalid}."
        )
    return sizes


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict) or not isinstance(value.get("suite"), dict):
        raise ValueError(f"A configuração precisa conter o mapa 'suite': {path}")
    return value


def _resolve_suite_output_root(suite_path: Path, suite: dict[str, Any]) -> Path:
    raw_output = suite.get("output_root")
    if not raw_output:
        raise ValueError("suite.output_root é obrigatório.")
    candidate = Path(str(raw_output)).expanduser()
    return candidate if candidate.is_absolute() else (suite_path.parent / candidate).resolve()


def _validate_base_suite(suite: dict[str, Any], *, epochs: int) -> dict[str, Any]:
    training = suite.get("training")
    if not isinstance(training, dict):
        raise ValueError("suite.training precisa ser um mapa.")
    if tuple(suite.get("seeds", ())) != (42,):
        raise ValueError("O sweep requer suite.seeds: [42].")
    if tuple(suite.get("normalizations", ())) != ("unit_interval",):
        raise ValueError("O sweep requer suite.normalizations: [unit_interval].")
    if tuple(suite.get("balance_modes", ())) != ("all_raw",):
        raise ValueError("O sweep requer suite.balance_modes: [all_raw].")
    fractions = (suite.get("train_fraction"), suite.get("validation_fraction"), suite.get("test_fraction"))
    if fractions != (0.70, 0.15, 0.15):
        raise ValueError("O sweep requer o split fixo 70/15/15.")
    if int(training.get("max_epochs", 0)) != int(epochs):
        raise ValueError(f"A configuração-base precisa definir max_epochs: {epochs}.")
    if float(training.get("extra_fraction", -1.0)) != 0.0:
        raise ValueError("O sweep requer training.extra_fraction: 0.0 para não gerar exemplos extras.")
    augmentation = training.get("augmentation")
    if augmentation != NEUTRAL_AUGMENTATION:
        raise ValueError("O sweep requer todos os parâmetros de augmentation explicitamente neutros.")
    if int(training.get("early_stopping_patience", 0)) < int(epochs):
        raise ValueError("early_stopping_patience deve ser ao menos o número de épocas do sweep.")
    if int(training.get("reduce_lr_patience", 0)) < int(epochs):
        raise ValueError("reduce_lr_patience deve ser ao menos o número de épocas do sweep.")
    return training


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_status(batch_root: Path) -> str:
    status_path = batch_root / "kmnist" / "runs" / RUN_ID / "status.json"
    try:
        return str(json.loads(status_path.read_text(encoding="utf-8")).get("status", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "missing"


def _write_batch_suite(
    *, base_payload: dict[str, Any], destination: Path, batch_size: int, batch_root: Path
) -> None:
    payload = copy.deepcopy(base_payload)
    suite = payload["suite"]
    training = suite["training"]
    suite["output_root"] = str(batch_root)
    training["batch_size"] = int(batch_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/kmnist-batch-sweep.yaml"))
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets.wsl.yaml"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Retoma batches interrompidos e pula os concluídos.")
    parser.add_argument("--rerun-failed", action="store_true", help="Inclui batches duravelmente marcados como failed.")
    parser.add_argument("--dry-run", action="store_true", help="Valida todos os jobs sem iniciar TensorFlow.")
    parser.add_argument("--fail-fast", action="store_true", help="Interrompe a fila no primeiro batch que falhar.")
    args = parser.parse_args()

    if args.epochs != 50:
        parser.error("Este experimento foi fixado em 50 épocas; use --epochs 50.")
    suite_path = _resolve_from_project(args.suite)
    registry_path = _resolve_from_project(args.registry)
    if not suite_path.is_file():
        parser.error(f"Suite não encontrada: {suite_path}")
    if not registry_path.is_file():
        parser.error(f"Registro não encontrado: {registry_path}")

    base_payload = _read_yaml(suite_path)
    suite = base_payload["suite"]
    _validate_base_suite(suite, epochs=args.epochs)
    root = _resolve_from_project(args.output_root) if args.output_root else _resolve_suite_output_root(suite_path, suite)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "sweep-status.json"
    generated_suites = root / "generated-suites"
    batches = tuple(args.batch_sizes)

    status: dict[str, Any] = {
        "experiment": "kmnist-alldata-noaug-batch-sweep",
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "status": "running",
        "protocol": {
            "dataset": "kmnist",
            "source_data": "all_raw: 70,000 examples before the fixed 70/15/15 split",
            "normalization": "unit_interval",
            "seed": 42,
            "max_epochs": args.epochs,
            "extra_fraction": 0.0,
            "augmentation": NEUTRAL_AUGMENTATION,
            "variable": "training.batch_size",
        },
        "batches": [int(batch) for batch in batches],
        "runs": {},
    }
    _write_json(status_path, status)

    failures = 0
    for batch_size in batches:
        batch_root = root / f"batch-{batch_size:03d}"
        generated_suite = generated_suites / f"kmnist-batch-{batch_size:03d}.yaml"
        _write_batch_suite(
            base_payload=base_payload,
            destination=generated_suite,
            batch_size=batch_size,
            batch_root=batch_root.resolve(),
        )
        before = _run_status(batch_root)
        record: dict[str, Any] = {
            "batch_size": int(batch_size),
            "suite": str(generated_suite),
            "output_root": str(batch_root),
            "status_before": before,
            "started_at": _timestamp(),
        }
        status["runs"][str(batch_size)] = record
        status["updated_at"] = _timestamp()
        _write_json(status_path, status)
        command = [
            str(PROJECT_ROOT / "scripts" / "wsl-gpu-env.sh"),
            sys.executable,
            "-m",
            "tcc_benchmark",
            "run",
            "--dataset",
            "kmnist",
            "--suite",
            str(generated_suite),
            "--registry",
            str(registry_path),
        ]
        if args.resume:
            command.append("--resume")
        if args.rerun_failed:
            command.append("--rerun-failed")
        if args.dry_run:
            command.append("--dry-run")
        if args.fail_fast:
            command.append("--fail-fast")

        print(f"[{_timestamp()}] batch={batch_size}: iniciando", flush=True)
        started = time.perf_counter()
        try:
            environment = dict(os.environ)
            environment.setdefault("PYTHON_BIN", sys.executable)
            environment.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            return_code = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode
        except OSError as exc:
            return_code = 127
            record["launcher_error"] = repr(exc)
        record["return_code"] = int(return_code)
        record["duration_seconds"] = round(time.perf_counter() - started, 3)
        record["finished_at"] = _timestamp()
        record["status_after"] = _run_status(batch_root)
        status["updated_at"] = _timestamp()
        _write_json(status_path, status)
        print(
            f"[{_timestamp()}] batch={batch_size}: {record['status_after']} "
            f"(rc={return_code}, {record['duration_seconds']} s)",
            flush=True,
        )
        if return_code or record["status_after"] == "failed":
            failures += 1
            if args.fail_fast:
                status["status"] = "failed"
                status["updated_at"] = _timestamp()
                _write_json(status_path, status)
                return 1

    status["status"] = "completed" if failures == 0 else "completed_with_failures"
    status["failures"] = failures
    status["updated_at"] = _timestamp()
    _write_json(status_path, status)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
