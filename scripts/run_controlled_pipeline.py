"""Execute the user-defined batch, quantization and activation sequence.

Run in WSL after ``scripts/bootstrap_controlled_benchmark.sh``:

    scripts/wsl-gpu-env.sh python scripts/run_controlled_pipeline.py

The process is intentionally serial and self-supervising. It writes durable
stage status files under the output root and stops before the next phase on a
failure. Re-run with ``--resume`` after fixing an environment/data problem.
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

from tcc_benchmark.controlled import select_activation, select_batch, select_quantization, write_phase_report
from tcc_benchmark.state import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
BATCH_SIZES = (32, 64, 128, 256)
QUANTIZATION_VARIANTS = ("fp32", "fp16", "int8_ptq")
ACTIVATIONS = ("relu", "sigmoid", "softmax")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def status_of(run_root: Path) -> str:
    return str((read_json(run_root / "status.json", {}) or {}).get("status", "missing"))


def run_id(dataset: str) -> str:
    return f"{dataset}__unit_interval__all_raw__seed-42"


def run_root(output_root: Path, dataset: str) -> Path:
    return output_root / dataset / "runs" / run_id(dataset)


def read_suite(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict) or not isinstance(value.get("suite"), dict):
        raise ValueError(f"Suite inválida: {path}")
    return value


def write_variant_suite(base: dict[str, Any], path: Path, output_root: Path, **training_changes: Any) -> None:
    payload = copy.deepcopy(base)
    payload["suite"]["output_root"] = str(output_root)
    payload["suite"]["training"].update(training_changes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def execute_training(dataset: str, suite_path: Path, registry_path: Path, *, resume: bool, dry_run: bool) -> int:
    command = [
        str(PROJECT_ROOT / "scripts" / "wsl-gpu-env.sh"),
        sys.executable,
        "-m",
        "tcc_benchmark",
        "run",
        "--dataset",
        dataset,
        "--suite",
        str(suite_path),
        "--registry",
        str(registry_path),
        "--fail-fast",
    ]
    if resume:
        command.append("--resume")
        # A failed cell is intentionally not retried by the core runner unless
        # this is explicit. A pipeline resume is that explicit retry after the
        # user has addressed the reported environment/data failure.
        command.append("--rerun-failed")
    if dry_run:
        command.append("--dry-run")
    environment = dict(os.environ)
    environment.setdefault("PYTHON_BIN", sys.executable)
    environment.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


def training_row(dataset: str, candidate: str, root: Path, *, kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    current = run_root(root, dataset)
    test = read_json(current / "artifacts" / "test_metrics.json", {}) or {}
    summary = read_json(current / "logs" / "training_summary.json", {}) or {}
    classification = test.get("classification", {}) if isinstance(test, dict) else {}
    row: dict[str, Any] = {
        "dataset": dataset,
        "status": status_of(current),
        "macro_f1": classification.get("macro_f1"),
        "accuracy": classification.get("accuracy"),
        "mean_epoch_seconds": summary.get("mean_epoch_seconds"),
        "run_root": str(current),
    }
    row[{"batch": "batch_size", "quant": "variant", "activation": "hidden_activation"}[kind]] = candidate
    if extra:
        row.update(extra)
    return row


def prepare_data(suite_path: Path, registry_path: Path, dataset: str) -> tuple[Any, Any, Any]:
    from tcc_benchmark.config import load_dataset_registry, load_suite_settings
    from tcc_benchmark.data import prepare_run_datasets
    from tcc_benchmark.runner import _augmentation_from_training, _join_source_samples, _load_entry, _validate_fixed_resolution

    settings = load_suite_settings(suite_path)
    entry = load_dataset_registry(registry_path)[dataset]
    size = _validate_fixed_resolution(entry, settings.training)
    materials = _join_source_samples(_load_entry(entry), image_size=size)
    prepared = prepare_run_datasets(
        materials.samples, materials.labels, image_size=size, channels=materials.channels,
        batch_size=settings.training.batch_size, normalization_mode="unit_interval", balance_mode="all_raw", seed=42,
        train_fraction=settings.train_fraction, validation_fraction=settings.validation_fraction, test_fraction=settings.test_fraction,
        extra_fraction=settings.training.extra_fraction, augmentation=_augmentation_from_training(settings.training),
        preprocess_cache_max_mib=settings.training.preprocess_cache_max_mib,
        shuffle_buffer_max_mib=settings.training.shuffle_buffer_max_mib,
    )
    return materials, prepared, settings


def export_litert(dataset: str, root: Path, suite_path: Path, registry_path: Path, mode: str, *, resume: bool) -> dict[str, Any]:
    """Export completed training to LiteRT and measure the same fixed test split."""

    from tcc_benchmark.quantization import benchmark_litert, convert_litert_model, write_litert_results

    current = run_root(root, dataset)
    artifacts = current / "artifacts"
    marker = artifacts / f"{mode}_litert_postprocess.json"
    prior = read_json(marker)
    if resume and isinstance(prior, dict) and prior.get("status") == "completed":
        return prior
    checkpoint = current / "checkpoints" / "best.keras"
    if not checkpoint.exists():
        checkpoint = current / "checkpoints" / "last.keras"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {current}")
    materials, prepared, _settings = prepare_data(suite_path, registry_path, dataset)
    import tensorflow as tf

    model = tf.keras.models.load_model(checkpoint, compile=False)
    conversion = convert_litert_model(
        model, artifacts / f"{mode}.tflite", mode=mode, representative_dataset=prepared.train_ds if mode == "int8" else None
    )
    payload: dict[str, Any] = {"status": "completed", "mode": mode, "conversion": conversion, "finished_at": timestamp()}
    if conversion.get("status") != "completed":
        payload.update({"status": "failed", "error": conversion.get("error")})
    else:
        benchmark, labels, logits = benchmark_litert(artifacts / f"{mode}.tflite", prepared.test_ds, num_classes=materials.num_classes)
        benchmark["conversion"] = conversion
        write_litert_results(artifacts, benchmark, labels, logits)
        payload["benchmark"] = benchmark
    payload["serialized_model_bytes"] = checkpoint.stat().st_size
    payload["serialized_litert_bytes"] = conversion.get("bytes")
    atomic_write_json(marker, payload)
    tf.keras.backend.clear_session()
    return payload


def ptq_row(dataset: str, fp32_root: Path, ptq_root: Path, suite_path: Path, registry_path: Path, *, resume: bool) -> dict[str, Any]:
    """Create a durable INT8-PTQ artefact from the completed FP32 checkpoint."""

    target = run_root(ptq_root, dataset)
    artifacts = target / "artifacts"
    marker = artifacts / "int8_ptq_litert_postprocess.json"
    prior = read_json(marker)
    if not (resume and isinstance(prior, dict) and prior.get("status") == "completed"):
        target.mkdir(parents=True, exist_ok=True)
        checkpoint = run_root(fp32_root, dataset) / "checkpoints" / "best.keras"
        if not checkpoint.exists():
            checkpoint = run_root(fp32_root, dataset) / "checkpoints" / "last.keras"
        if not checkpoint.exists():
            raise FileNotFoundError(f"INT8-PTQ exige FP32 concluído para {dataset}.")
        artifacts.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target / "status.json", {"status": "running", "updated_at": timestamp()})
        from tcc_benchmark.quantization import benchmark_litert, convert_litert_model, write_litert_results
        import tensorflow as tf

        materials, prepared, _settings = prepare_data(suite_path, registry_path, dataset)
        model = tf.keras.models.load_model(checkpoint, compile=False)
        conversion = convert_litert_model(model, artifacts / "int8_ptq.tflite", mode="int8", representative_dataset=prepared.train_ds)
        prior = {"status": "completed", "mode": "int8", "conversion": conversion, "finished_at": timestamp(), "serialized_model_bytes": checkpoint.stat().st_size, "serialized_litert_bytes": conversion.get("bytes")}
        if conversion.get("status") != "completed":
            prior.update({"status": "failed", "error": conversion.get("error")})
        else:
            benchmark, labels, logits = benchmark_litert(artifacts / "int8_ptq.tflite", prepared.test_ds, num_classes=materials.num_classes)
            benchmark["conversion"] = conversion
            write_litert_results(artifacts, benchmark, labels, logits)
            prior["benchmark"] = benchmark
        atomic_write_json(marker, prior)
        atomic_write_json(target / "status.json", {"status": prior["status"], "updated_at": timestamp(), "detail": "INT8 PTQ"})
        tf.keras.backend.clear_session()
    benchmark = prior.get("benchmark", {}) if isinstance(prior, dict) else {}
    classification = benchmark.get("classification", {}) if isinstance(benchmark, dict) else {}
    return {"dataset": dataset, "variant": "int8_ptq", "status": prior.get("status", "missing"), "macro_f1": classification.get("macro_f1"), "accuracy": classification.get("accuracy"), "median_batch_latency_ms": benchmark.get("median_batch_latency_ms"), "serialized_model_bytes": prior.get("serialized_model_bytes"), "serialized_litert_bytes": prior.get("serialized_litert_bytes"), "run_root": str(target)}


def quant_row(dataset: str, variant: str, root: Path, suite_path: Path, registry_path: Path, *, resume: bool) -> dict[str, Any]:
    post = export_litert(dataset, root, suite_path, registry_path, "fp32" if variant == "fp32" else "fp16", resume=resume)
    benchmark = post.get("benchmark", {}) if isinstance(post, dict) else {}
    classification = benchmark.get("classification", {}) if isinstance(benchmark, dict) else {}
    source = training_row(dataset, variant, root, kind="quant")
    source.update({"status": post.get("status", source["status"]), "macro_f1": classification.get("macro_f1"), "accuracy": classification.get("accuracy"), "median_batch_latency_ms": benchmark.get("median_batch_latency_ms"), "serialized_model_bytes": post.get("serialized_model_bytes"), "serialized_litert_bytes": post.get("serialized_litert_bytes")})
    return source


def run_stage_batch(base: dict[str, Any], root: Path, registry: Path, *, resume: bool, dry_run: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for batch in BATCH_SIZES:
            output = root / "batch" / f"batch-{batch:03d}"
            suite = root / "generated-suites" / "batch" / f"{dataset}-batch-{batch:03d}.yaml"
            write_variant_suite(base, suite, output, batch_size=batch, dtype_policy="mixed_float16", hidden_activation="swish")
            rc = execute_training(dataset, suite, registry, resume=resume, dry_run=dry_run)
            row = training_row(dataset, str(batch), output, kind="batch", extra={"return_code": rc})
            rows.append(row)
            if rc or row["status"] == "failed":
                raise RuntimeError(f"Batch falhou: {dataset}, batch={batch} (rc={rc}, status={row['status']}).")
    if dry_run:
        write_phase_report(root / "reports", phase="batch", rows=rows, winners={})
        return {"rows": rows, "winners": {}}
    winners = {dataset: select_batch([row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS}
    write_phase_report(root / "reports", phase="batch", rows=rows, winners=winners)
    return {"rows": rows, "winners": winners}


def run_stage_quant(base: dict[str, Any], root: Path, registry: Path, batches: dict[str, Any], *, resume: bool, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"rows": [], "winners": {}}
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        batch = int(batches[dataset]["batch_size"])
        fp_roots: dict[str, Path] = {}
        for variant, policy in (("fp32", "float32"), ("fp16", "mixed_float16")):
            output = root / "quantization" / dataset / variant
            fp_roots[variant] = output
            suite = root / "generated-suites" / "quantization" / f"{dataset}-{variant}.yaml"
            write_variant_suite(base, suite, output, batch_size=batch, dtype_policy=policy, hidden_activation="swish")
            rc = execute_training(dataset, suite, registry, resume=resume, dry_run=False)
            if rc or status_of(run_root(output, dataset)) == "failed":
                raise RuntimeError(f"Quantização {variant} falhou em {dataset}.")
            rows.append(quant_row(dataset, variant, output, suite, registry, resume=resume))
        ptq_output = root / "quantization" / dataset / "int8_ptq"
        rows.append(ptq_row(dataset, fp_roots["fp32"], ptq_output, root / "generated-suites" / "quantization" / f"{dataset}-fp32.yaml", registry, resume=resume))
        if rows[-1]["status"] != "completed":
            raise RuntimeError(f"INT8-PTQ falhou em {dataset}.")
    winners = {dataset: select_quantization([row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS}
    write_phase_report(root / "reports", phase="quantization", rows=rows, winners=winners)
    return {"rows": rows, "winners": winners}


def run_stage_activation(base: dict[str, Any], root: Path, registry: Path, batches: dict[str, Any], quant: dict[str, Any], *, resume: bool, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"rows": [], "winners": {}}
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        batch = int(batches[dataset]["batch_size"])
        selected = str(quant[dataset]["variant"])
        policy = "mixed_float16" if selected == "fp16" else "float32"
        export_mode = "fp16" if selected == "fp16" else "int8" if selected == "int8_ptq" else "fp32"
        for activation in ACTIVATIONS:
            output = root / "activations" / dataset / activation
            suite = root / "generated-suites" / "activations" / f"{dataset}-{activation}.yaml"
            write_variant_suite(base, suite, output, batch_size=batch, dtype_policy=policy, hidden_activation=activation)
            rc = execute_training(dataset, suite, registry, resume=resume, dry_run=False)
            if rc or status_of(run_root(output, dataset)) == "failed":
                raise RuntimeError(f"Ativação {activation} falhou em {dataset}.")
            post = export_litert(dataset, output, suite, registry, export_mode, resume=resume)
            benchmark = post.get("benchmark", {}) if isinstance(post, dict) else {}
            classification = benchmark.get("classification", {}) if isinstance(benchmark, dict) else {}
            row = training_row(dataset, activation, output, kind="activation", extra={"selected_quantization": selected, "median_batch_latency_ms": benchmark.get("median_batch_latency_ms"), "serialized_litert_bytes": post.get("serialized_litert_bytes")})
            row.update({"status": post.get("status", row["status"]), "macro_f1": classification.get("macro_f1"), "accuracy": classification.get("accuracy")})
            rows.append(row)
            if row["status"] != "completed":
                raise RuntimeError(f"Exportação {export_mode} falhou para {dataset}/{activation}.")
    winners = {dataset: select_activation([row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS}
    write_phase_report(root / "reports", phase="activation", rows=rows, winners=winners)
    return {"rows": rows, "winners": winners}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/controlled-augmentation2.yaml"))
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/controlled-augmentation2"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = (PROJECT_ROOT / args.suite).resolve() if not args.suite.is_absolute() else args.suite.resolve()
    registry = (PROJECT_ROOT / args.registry).resolve() if not args.registry.is_absolute() else args.registry.resolve()
    root = (PROJECT_ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    if not suite.is_file() or not registry.is_file():
        parser.error("Suite ou registro de datasets não encontrado.")
    base = read_suite(suite)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline-status.json"
    status: dict[str, Any] = read_json(state_path, {}) or {"experiment": "controlled-augmentation2", "created_at": timestamp(), "stages": {}}
    status.setdefault("stages", {})
    started = time.perf_counter()
    try:
        batch = run_stage_batch(base, root, registry, resume=args.resume, dry_run=args.dry_run)
        if args.dry_run:
            status.update({"status": "dry_run_completed", "finished_at": timestamp(), "duration_seconds": round(time.perf_counter() - started, 3)})
            status["stages"]["batch"] = {"status": "planned"}
            atomic_write_json(state_path, status)
            return 0
        status["stages"]["batch"] = {"status": "completed", "winners": batch["winners"]}
        atomic_write_json(state_path, status)
        quant = run_stage_quant(base, root, registry, batch["winners"], resume=args.resume, dry_run=args.dry_run)
        status["stages"]["quantization"] = {"status": "completed", "winners": quant["winners"]}
        atomic_write_json(state_path, status)
        activation = run_stage_activation(base, root, registry, batch["winners"], quant["winners"], resume=args.resume, dry_run=args.dry_run)
        status["stages"]["activation"] = {"status": "completed", "winners": activation["winners"]}
        status.update({"status": "completed", "finished_at": timestamp(), "duration_seconds": round(time.perf_counter() - started, 3)})
        atomic_write_json(state_path, status)
        return 0
    except Exception as exc:
        status.update({"status": "failed", "failed_at": timestamp(), "error": repr(exc), "duration_seconds": round(time.perf_counter() - started, 3)})
        atomic_write_json(state_path, status)
        print(f"Pipeline interrompido: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
