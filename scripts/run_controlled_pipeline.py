"""Supervisor serial, multiplataforma e retomável do benchmark controlado.

No macOS o processo usa diretamente o Python ativo e o dispositivo
tensorflow-metal. Em WSL/Linux ele mantém o wrapper CUDA existente. O
supervisor não executa jobs concorrentes, não ajusta parâmetros depois de uma
falha e grava estado após cada gate e fase.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
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
ACTIVATIONS = ("relu", "sigmoid", "softmax")
HIDDEN_ACTIVATIONS = ("swish", *ACTIVATIONS)
DTYPE_POLICIES = {"fp16": "mixed_float16", "fp32": "float32"}


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, non-empty subset of the controlled matrix."""

    try:
        batches = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batch-sizes deve conter inteiros separados por vírgula.") from exc
    if not batches or len(set(batches)) != len(batches) or any(batch not in BATCH_SIZES for batch in batches):
        allowed = ", ".join(str(batch) for batch in BATCH_SIZES)
        raise argparse.ArgumentTypeError(f"--batch-sizes deve ser um subconjunto sem repetição de: {allowed}.")
    return batches


def parse_activations(value: str) -> tuple[str, ...]:
    """Parse explicit activation candidates for a standalone activation phase."""

    values = tuple(item.strip() for item in str(value).split(",") if item.strip())
    unknown = [item for item in values if item not in HIDDEN_ACTIVATIONS]
    if not values or len(set(values)) != len(values) or unknown:
        allowed = ", ".join(HIDDEN_ACTIVATIONS)
        raise argparse.ArgumentTypeError(f"--activations deve ser uma lista sem repetição de: {allowed}.")
    return values


def forced_activation_inputs(*, batch_size: int, dtype_policy: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Create portable, explicit upstream selections for activation-only work."""

    if int(batch_size) < 1:
        raise ValueError("--activation-batch-size deve ser positivo.")
    variants = {policy: variant for variant, policy in DTYPE_POLICIES.items()}
    if dtype_policy not in variants:
        raise ValueError(f"Política de dtype não suportada: {dtype_policy!r}.")
    variant = variants[dtype_policy]
    batches = {
        dataset: {"dataset": dataset, "batch_size": int(batch_size), "status": "forced"}
        for dataset in DATASETS
    }
    quant = {
        dataset: {"dataset": dataset, "batch_size": int(batch_size), "variant": variant, "status": "forced"}
        for dataset in DATASETS
    }
    return batches, quant


def _is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def _runtime_command(arguments: list[str]) -> list[str]:
    command = [sys.executable, "-m", "tcc_benchmark", *arguments]
    if _is_wsl():
        return [str(PROJECT_ROOT / "scripts" / "wsl-gpu-env.sh"), *command]
    return command


def _runtime_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    if _is_wsl():
        environment.setdefault("PYTHON_BIN", sys.executable)
        environment.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    else:
        environment.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)
    return environment


def _run_command(command: list[str]) -> int:
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=_runtime_environment(), check=False).returncode


def execute_training(dataset: str, suite_path: Path, registry_path: Path, *, resume: bool, dry_run: bool) -> int:
    arguments = [
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
        arguments.extend(("--resume", "--rerun-failed"))
    if dry_run:
        arguments.append("--dry-run")
    return _run_command(_runtime_command(arguments))


def training_row(dataset: str, candidate: str, root: Path, *, kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    current = run_root(root, dataset)
    test = read_json(current / "artifacts" / "test_metrics.json", {}) or {}
    summary = read_json(current / "logs" / "training_summary.json", {}) or {}
    telemetry = read_json(current / "telemetry" / "summary.json", {}) or {}
    classification = test.get("classification", {}) if isinstance(test, dict) else {}
    hardware = telemetry.get("metrics", {}) if isinstance(telemetry, dict) else {}
    categorical = telemetry.get("categorical", {}) if isinstance(telemetry, dict) else {}

    def metric(name: str, statistic: str = "max") -> Any:
        value = hardware.get(name, {}) if isinstance(hardware, dict) else {}
        return value.get(statistic) if isinstance(value, dict) else None

    row: dict[str, Any] = {
        "dataset": dataset,
        "status": status_of(current),
        "macro_f1": classification.get("macro_f1"),
        "accuracy": classification.get("accuracy"),
        "mean_epoch_seconds": summary.get("mean_epoch_seconds"),
        "mean_train_examples_per_second": summary.get("mean_train_examples_per_second"),
        "telemetry_samples": telemetry.get("sample_count"),
        "gpu_backend": telemetry.get("gpu_backend"),
        "gpu_memory_kind": (categorical.get("gpu_memory_kind", {}) or {}).get("current") if isinstance(categorical, dict) else None,
        "thermal_pressure": (categorical.get("thermal_pressure", {}) or {}).get("current") if isinstance(categorical, dict) else None,
        "peak_cpu_percent": metric("cpu_percent"),
        "peak_ram_used_bytes": metric("ram_used_bytes"),
        "peak_process_rss_bytes": metric("process_rss_bytes"),
        "peak_gpu_utilization_percent": metric("gpu_utilization_percent"),
        "peak_gpu_renderer_utilization_percent": metric("gpu_renderer_utilization_percent"),
        "peak_gpu_tiler_utilization_percent": metric("gpu_tiler_utilization_percent"),
        "peak_gpu_driver_allocated_memory_bytes": metric("gpu_driver_allocated_memory_bytes"),
        "peak_gpu_system_memory_in_use_bytes": metric("gpu_system_memory_in_use_bytes"),
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
        materials.samples,
        materials.labels,
        image_size=size,
        channels=materials.channels,
        batch_size=settings.training.batch_size,
        normalization_mode="unit_interval",
        balance_mode="all_raw",
        seed=42,
        train_fraction=settings.train_fraction,
        validation_fraction=settings.validation_fraction,
        test_fraction=settings.test_fraction,
        extra_fraction=settings.training.extra_fraction,
        augmentation=_augmentation_from_training(settings),
        preprocess_cache_max_mib=settings.training.preprocess_cache_max_mib,
        shuffle_buffer_max_mib=settings.training.shuffle_buffer_max_mib,
    )
    return materials, prepared, settings


def export_litert(dataset: str, root: Path, suite_path: Path, registry_path: Path, mode: str, *, resume: bool) -> dict[str, Any]:
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
        model,
        artifacts / f"{mode}.tflite",
        mode=mode,
        representative_dataset=prepared.train_ds if mode == "int8" else None,
    )
    payload: dict[str, Any] = {"status": "completed", "mode": mode, "conversion": conversion, "finished_at": timestamp()}
    if conversion.get("status") != "completed":
        payload.update({"status": "failed", "error": conversion.get("error")})
    else:
        benchmark, labels, logits = benchmark_litert(
            artifacts / f"{mode}.tflite", prepared.test_ds, num_classes=materials.num_classes
        )
        benchmark["conversion"] = conversion
        write_litert_results(artifacts, benchmark, labels, logits)
        payload["benchmark"] = benchmark
    payload["serialized_model_bytes"] = checkpoint.stat().st_size
    payload["serialized_litert_bytes"] = conversion.get("bytes")
    atomic_write_json(marker, payload)
    tf.keras.backend.clear_session()
    return payload


def ptq_row(dataset: str, fp32_root: Path, ptq_root: Path, suite_path: Path, registry_path: Path, *, resume: bool) -> dict[str, Any]:
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
        conversion = convert_litert_model(
            model,
            artifacts / "int8_ptq.tflite",
            mode="int8",
            representative_dataset=prepared.train_ds,
        )
        prior = {
            "status": "completed",
            "mode": "int8",
            "conversion": conversion,
            "finished_at": timestamp(),
            "serialized_model_bytes": checkpoint.stat().st_size,
            "serialized_litert_bytes": conversion.get("bytes"),
        }
        if conversion.get("status") != "completed":
            prior.update({"status": "failed", "error": conversion.get("error")})
        else:
            benchmark, labels, logits = benchmark_litert(
                artifacts / "int8_ptq.tflite", prepared.test_ds, num_classes=materials.num_classes
            )
            benchmark["conversion"] = conversion
            write_litert_results(artifacts, benchmark, labels, logits)
            prior["benchmark"] = benchmark
        atomic_write_json(marker, prior)
        atomic_write_json(target / "status.json", {"status": prior["status"], "updated_at": timestamp(), "detail": "INT8 PTQ"})
        tf.keras.backend.clear_session()
    benchmark = prior.get("benchmark", {}) if isinstance(prior, dict) else {}
    classification = benchmark.get("classification", {}) if isinstance(benchmark, dict) else {}
    return {
        "dataset": dataset,
        "variant": "int8_ptq",
        "status": prior.get("status", "missing"),
        "macro_f1": classification.get("macro_f1"),
        "accuracy": classification.get("accuracy"),
        "median_batch_latency_ms": benchmark.get("median_batch_latency_ms"),
        "throughput_examples_per_second": benchmark.get("throughput_examples_per_second"),
        "serialized_model_bytes": prior.get("serialized_model_bytes"),
        "serialized_litert_bytes": prior.get("serialized_litert_bytes"),
        "run_root": str(target),
    }


def quant_row(dataset: str, variant: str, root: Path, suite_path: Path, registry_path: Path, *, resume: bool) -> dict[str, Any]:
    post = export_litert(dataset, root, suite_path, registry_path, "fp32" if variant == "fp32" else "fp16", resume=resume)
    benchmark = post.get("benchmark", {}) if isinstance(post, dict) else {}
    classification = benchmark.get("classification", {}) if isinstance(benchmark, dict) else {}
    source = training_row(dataset, variant, root, kind="quant")
    source.update(
        {
            "status": post.get("status", source["status"]),
            "macro_f1": classification.get("macro_f1"),
            "accuracy": classification.get("accuracy"),
            "median_batch_latency_ms": benchmark.get("median_batch_latency_ms"),
            "throughput_examples_per_second": benchmark.get("throughput_examples_per_second"),
            "serialized_model_bytes": post.get("serialized_model_bytes"),
            "serialized_litert_bytes": post.get("serialized_litert_bytes"),
        }
    )
    return source


def run_stage_batch(
    base: dict[str, Any],
    root: Path,
    registry: Path,
    *,
    resume: bool,
    dry_run: bool,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for batch in batch_sizes:
            output = root / "batch" / f"batch-{batch:03d}"
            suite = root / "generated-suites" / "batch" / f"{dataset}-batch-{batch:03d}.yaml"
            write_variant_suite(base, suite, output, batch_size=batch, dtype_policy="mixed_float16", hidden_activation="swish")
            rc = execute_training(dataset, suite, registry, resume=resume, dry_run=dry_run)
            row = training_row(dataset, str(batch), output, kind="batch", extra={"return_code": rc})
            if dry_run:
                row["status"] = "planned"
            rows.append(row)
            if rc or (not dry_run and row["status"] == "failed"):
                raise RuntimeError(f"Batch falhou: {dataset}, batch={batch} (rc={rc}, status={row['status']}).")
    winners = {} if dry_run else {dataset: select_batch([row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS}
    write_phase_report(root / "reports", phase="batch", rows=rows, winners=winners)
    return {"rows": rows, "winners": winners}


def run_stage_quant(base: dict[str, Any], root: Path, registry: Path, batches: dict[str, Any], *, resume: bool, dry_run: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if dry_run:
        # A dry-run must materialize the complete 18-run quantization plan,
        # including the PTQ conversion that follows each FP32 candidate.
        for dataset in DATASETS:
            batch = int(batches.get(dataset, {}).get("batch_size", BATCH_SIZES[0]))
            for variant, policy in (("fp32", "float32"), ("fp16", "mixed_float16")):
                output = root / "quantization" / dataset / variant
                suite = root / "generated-suites" / "quantization" / f"{dataset}-{variant}.yaml"
                write_variant_suite(base, suite, output, batch_size=batch, dtype_policy=policy, hidden_activation="swish")
                row = training_row(dataset, variant, output, kind="quant", extra={"return_code": 0})
                row["status"] = "planned"
                rows.append(row)
            ptq_output = root / "quantization" / dataset / "int8_ptq"
            rows.append(
                {
                    "dataset": dataset,
                    "variant": "int8_ptq",
                    "status": "planned",
                    "batch_size": batch,
                    "run_root": str(run_root(ptq_output, dataset)),
                    "return_code": 0,
                }
            )
        winners = {
            dataset: {
                "dataset": dataset,
                "variant": "fp32",
                "batch_size": int(batches.get(dataset, {}).get("batch_size", BATCH_SIZES[0])),
                "status": "planned",
            }
            for dataset in DATASETS
        }
        write_phase_report(root / "reports", phase="quantization", rows=rows, winners=winners)
        return {"rows": rows, "winners": winners}
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
        rows.append(
            ptq_row(
                dataset,
                fp_roots["fp32"],
                ptq_output,
                root / "generated-suites" / "quantization" / f"{dataset}-fp32.yaml",
                registry,
                resume=resume,
            )
        )
        if rows[-1]["status"] != "completed":
            raise RuntimeError(f"INT8-PTQ falhou em {dataset}.")
    winners = {dataset: select_quantization([row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS}
    write_phase_report(root / "reports", phase="quantization", rows=rows, winners=winners)
    return {"rows": rows, "winners": winners}


def run_stage_activation(
    base: dict[str, Any],
    root: Path,
    registry: Path,
    batches: dict[str, Any],
    quant: dict[str, Any],
    *,
    resume: bool,
    dry_run: bool,
    activations: tuple[str, ...] = ACTIVATIONS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if dry_run:
        # Keep planning serial and side-effect free with respect to TensorFlow:
        # generated suites and planned rows are enough to audit all 27 jobs.
        for dataset in DATASETS:
            batch = int(batches.get(dataset, {}).get("batch_size", BATCH_SIZES[0]))
            selected = str(quant.get(dataset, {}).get("variant", "fp32"))
            policy = "mixed_float16" if selected == "fp16" else "float32"
            for activation in activations:
                output = root / "activations" / dataset / activation
                suite = root / "generated-suites" / "activations" / f"{dataset}-{activation}.yaml"
                write_variant_suite(base, suite, output, batch_size=batch, dtype_policy=policy, hidden_activation=activation)
                row = training_row(
                    dataset,
                    activation,
                    output,
                    kind="activation",
                    extra={"selected_quantization": selected, "return_code": 0},
                )
                row["status"] = "planned"
                rows.append(row)
        write_phase_report(root / "reports", phase="activation", rows=rows, winners={})
        return {"rows": rows, "winners": {}}
    for dataset in DATASETS:
        batch = int(batches[dataset]["batch_size"])
        selected = str(quant[dataset]["variant"])
        policy = "mixed_float16" if selected == "fp16" else "float32"
        export_mode = "fp16" if selected == "fp16" else "int8" if selected == "int8_ptq" else "fp32"
        for activation in activations:
            output = root / "activations" / dataset / activation
            suite = root / "generated-suites" / "activations" / f"{dataset}-{activation}.yaml"
            write_variant_suite(base, suite, output, batch_size=batch, dtype_policy=policy, hidden_activation=activation)
            rc = execute_training(dataset, suite, registry, resume=resume, dry_run=False)
            if rc or status_of(run_root(output, dataset)) == "failed":
                raise RuntimeError(f"Ativação {activation} falhou em {dataset}.")
            post = export_litert(dataset, output, suite, registry, export_mode, resume=resume)
            benchmark = post.get("benchmark", {}) if isinstance(post, dict) else {}
            classification = benchmark.get("classification", {}) if isinstance(benchmark, dict) else {}
            row = training_row(
                dataset,
                activation,
                output,
                kind="activation",
                extra={
                    "selected_quantization": selected,
                    "median_batch_latency_ms": benchmark.get("median_batch_latency_ms"),
                    "throughput_examples_per_second": benchmark.get("throughput_examples_per_second"),
                    "serialized_litert_bytes": post.get("serialized_litert_bytes"),
                },
            )
            row.update({"status": post.get("status", row["status"]), "macro_f1": classification.get("macro_f1"), "accuracy": classification.get("accuracy")})
            rows.append(row)
            if row["status"] != "completed":
                raise RuntimeError(f"Exportação {export_mode} falhou para {dataset}/{activation}.")
    winners = {dataset: select_activation([row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS}
    write_phase_report(root / "reports", phase="activation", rows=rows, winners=winners)
    return {"rows": rows, "winners": winners}


def _write_dataset_phase_reports(root: Path, phase_rows: dict[str, list[dict[str, Any]]], winners: dict[str, dict[str, Any]]) -> None:
    for phase, rows in phase_rows.items():
        for dataset in DATASETS:
            selected = [row for row in rows if row.get("dataset") == dataset]
            winner = winners.get(phase, {}).get(dataset)
            write_phase_report(root / "reports" / dataset, phase=phase, rows=selected, winners={dataset: winner} if winner else {})


def _run_gates(root: Path, suite: Path, registry: Path, *, resume: bool, dry_run: bool, state: dict[str, Any]) -> None:
    state.setdefault("gates", {})
    if dry_run:
        print("Dry-run: preflight e smoke não serão executados; auditoria e planejamento serão validados.", flush=True)
    elif not (resume and state["gates"].get("preflight") == "completed"):
        rc = _run_command(
            _runtime_command(
                [
                    "preflight",
                    "--require-tensorflow",
                    "--require-gpu",
                    "--output-root",
                    str(root / "preflight"),
                    "--data-path",
                    str(PROJECT_ROOT / "datasets"),
                ]
            )
        )
        if rc:
            raise RuntimeError("Preflight falhou; nenhuma fase de treinamento foi iniciada.")
        state["gates"]["preflight"] = "completed"
        atomic_write_json(root / "pipeline-status.json", state)

    if not (resume and state["gates"].get("audit") == "completed"):
        rc = _run_command(
            _runtime_command(
                [
                    "audit",
                    "--all",
                    "--suite",
                    str(suite),
                    "--registry",
                    str(registry),
                    "--output-root",
                    str(root / "audit"),
                    "--allow-conflicting-duplicates",
                ]
            )
        )
        if rc:
            raise RuntimeError("Auditoria dos datasets falhou; o benchmark foi bloqueado.")
        state["gates"]["audit"] = "completed"
        atomic_write_json(root / "pipeline-status.json", state)

    if dry_run:
        return
    if not (resume and state["gates"].get("smoke") == "completed"):
        for dataset in DATASETS:
            rc = _run_command(
                _runtime_command(
                    [
                        "smoke",
                        "--dataset",
                        dataset,
                        "--suite",
                        str(suite),
                        "--registry",
                        str(registry),
                        "--output-root",
                        str(root),
                        "--examples-per-class",
                        "16",
                        "--epochs",
                        "2",
                    ]
                )
            )
            if rc:
                raise RuntimeError(f"Smoke test falhou para {dataset}; a matriz foi bloqueada.")
        state["gates"]["smoke"] = "completed"
        atomic_write_json(root / "pipeline-status.json", state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/controlled-augmentation2-mac-m4.yaml"))
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/controlled-augmentation2-mac-m4"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=BATCH_SIZES,
        help="Batches a executar na fase batch, separados por vírgula (padrão: 32,64,128,256).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--skip-activation",
        action="store_true",
        help="Executa gates, batch e quantização, mas encerra antes da fase de ativações.",
    )
    mode.add_argument(
        "--activation-only",
        action="store_true",
        help="Executa somente a fase de ativações com batch e precisão declarados explicitamente.",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Encerra após a fase batch; exige --skip-activation.",
    )
    parser.add_argument(
        "--activation-batch-size",
        type=int,
        help="Batch obrigatório para --activation-only; não depende de pipeline-status remoto.",
    )
    parser.add_argument(
        "--activation-dtype-policy",
        choices=tuple(DTYPE_POLICIES.values()),
        default="mixed_float16",
        help="Precisão para --activation-only (padrão: mixed_float16, equivalente a FP16).",
    )
    parser.add_argument(
        "--activations",
        type=parse_activations,
        default=ACTIVATIONS,
        help="Ativações para --activation-only, separadas por vírgula (padrão: relu,sigmoid,softmax).",
    )
    parser.add_argument(
        "--skip-gates",
        action="store_true",
        help="Não executa preflight, auditoria nem smoke; permitido apenas com --activation-only.",
    )
    args = parser.parse_args()
    if args.skip_gates and not args.activation_only:
        parser.error("--skip-gates só pode ser usado com --activation-only.")
    if args.skip_quantization and args.activation_only:
        parser.error("--skip-quantization não pode ser usado com --activation-only.")
    if args.skip_quantization and not args.skip_activation:
        parser.error("--skip-quantization exige --skip-activation, pois ativações dependem da quantização.")
    if args.activation_batch_size is not None and not args.activation_only:
        parser.error("--activation-batch-size exige --activation-only.")
    if args.activation_only and args.activation_batch_size is None:
        parser.error("--activation-only exige --activation-batch-size.")
    suite = (PROJECT_ROOT / args.suite).resolve() if not args.suite.is_absolute() else args.suite.resolve()
    registry = (PROJECT_ROOT / args.registry).resolve() if not args.registry.is_absolute() else args.registry.resolve()
    root = (PROJECT_ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    if not suite.is_file() or not registry.is_file():
        parser.error("Suite ou registro de datasets não encontrado.")
    base = read_suite(suite)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline-status.json"
    state: dict[str, Any] = read_json(state_path, {}) or {
        "experiment": "controlled-augmentation2",
        "platform_profile": "mac-metal" if platform.system() == "Darwin" else "wsl-or-cpu",
        "created_at": timestamp(),
        "stages": {},
        "gates": {},
    }
    state.setdefault("stages", {})
    state.setdefault("gates", {})
    started = time.perf_counter()
    try:
        if not args.skip_gates:
            _run_gates(root, suite, registry, resume=args.resume, dry_run=args.dry_run, state=state)
        stage_status = "planned" if args.dry_run else "completed"
        phase_rows: dict[str, list[dict[str, Any]]] = {}
        winners: dict[str, dict[str, dict[str, Any]]] = {}

        if args.activation_only:
            batches, quant = forced_activation_inputs(
                batch_size=args.activation_batch_size,
                dtype_policy=args.activation_dtype_policy,
            )
            activation = run_stage_activation(
                base,
                root,
                registry,
                batches,
                quant,
                resume=args.resume,
                dry_run=args.dry_run,
                activations=args.activations,
            )
            state["stages"]["activation"] = {
                "status": stage_status,
                "winners": activation["winners"],
                "run_count": len(activation["rows"]),
                "mode": "activation_only",
                "forced_batch_size": int(args.activation_batch_size),
                "forced_dtype_policy": args.activation_dtype_policy,
                "activations": list(args.activations),
            }
            phase_rows["activation"] = activation["rows"]
            winners["activation"] = activation["winners"]
            final_status = "activation_only_planned" if args.dry_run else "activation_only_completed"
            counts = {
                "batch_training_runs": 0,
                "quantization_training_runs": 0,
                "activation_training_runs": len(activation["rows"]),
                "total_training_runs": len(activation["rows"]),
            }
        else:
            batch = run_stage_batch(
                base,
                root,
                registry,
                resume=args.resume,
                dry_run=args.dry_run,
                batch_sizes=args.batch_sizes,
            )
            state["stages"]["batch"] = {
                "status": stage_status,
                "winners": batch["winners"],
                "run_count": len(batch["rows"]),
                "batch_sizes": list(args.batch_sizes),
            }
            atomic_write_json(state_path, state)
            phase_rows["batch"] = batch["rows"]
            winners["batch"] = batch["winners"]
            if args.skip_quantization:
                state["stages"]["quantization"] = {
                    "status": "skipped",
                    "reason": "Execução limitada à fase batch.",
                }
                state["stages"]["activation"] = {
                    "status": "skipped",
                    "reason": "Ativações exigem resultados da quantização e foram transferidas para outro ambiente.",
                }
                final_status = "completed_batch_only" if not args.dry_run else "planned_batch_only"
                activation_count = 0
                quantization_training_count = 0
            else:
                quant = run_stage_quant(base, root, registry, batch["winners"], resume=args.resume, dry_run=args.dry_run)
                state["stages"]["quantization"] = {
                    "status": stage_status,
                    "winners": quant["winners"],
                    "run_count": len(quant["rows"]),
                }
                atomic_write_json(state_path, state)
                phase_rows["quantization"] = quant["rows"]
                winners["quantization"] = quant["winners"]
                quantization_training_count = len(DATASETS) * 2
                if args.skip_activation:
                    state["stages"]["activation"] = {
                        "status": "skipped",
                        "reason": "Execução transferida para outro ambiente.",
                    }
                    final_status = "completed_without_activation" if not args.dry_run else "planned_without_activation"
                    activation_count = 0
                else:
                    activation = run_stage_activation(
                        base,
                        root,
                        registry,
                        batch["winners"],
                        quant["winners"],
                        resume=args.resume,
                        dry_run=args.dry_run,
                        activations=args.activations,
                    )
                    state["stages"]["activation"] = {
                        "status": stage_status,
                        "winners": activation["winners"],
                        "run_count": len(activation["rows"]),
                    }
                    phase_rows["activation"] = activation["rows"]
                    winners["activation"] = activation["winners"]
                    final_status = "dry_run_completed" if args.dry_run else "completed"
                    activation_count = len(activation["rows"])
            counts = {
                "batch_training_runs": len(batch["rows"]),
                "quantization_training_runs": quantization_training_count,
                "activation_training_runs": activation_count,
                "total_training_runs": len(batch["rows"]) + quantization_training_count + activation_count,
            }
        atomic_write_json(state_path, state)
        _write_dataset_phase_reports(root, phase_rows, winners)
        write_phase_report(root / "reports", phase="consolidated", rows=[row for rows in phase_rows.values() for row in rows], winners={})
        state.update(
            {
                "status": final_status,
                "finished_at": timestamp(),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "counts": counts,
            }
        )
        atomic_write_json(state_path, state)
        print(f"Pipeline concluído: {state['status']}. Saída: {root}", flush=True)
        return 0
    except KeyboardInterrupt:
        state.update({"status": "interrupted", "interrupted_at": timestamp()})
        atomic_write_json(state_path, state)
        print("Pipeline interrompido. Execute novamente com --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        state.update({"status": "failed", "failed_at": timestamp(), "error": repr(exc), "duration_seconds": round(time.perf_counter() - started, 3)})
        atomic_write_json(state_path, state)
        print(f"Pipeline interrompido: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
