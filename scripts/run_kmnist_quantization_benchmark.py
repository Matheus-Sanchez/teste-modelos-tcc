"""Run the fixed KMNIST FP32/FP16/QAT/PTQ quantization benchmark sequentially.

Run inside WSL through the project wrapper:

    PYTHON_BIN=/home/msduda/.venvs/tcc-benchmark/bin/python \
      scripts/wsl-gpu-env.sh /home/msduda/.venvs/tcc-benchmark/bin/python \
      scripts/run_kmnist_quantization_benchmark.py --resume
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

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "kmnist__unit_interval__all_raw__seed-42"
EXPECTED_EPOCHS = 50
CALIBRATION_EXAMPLES = 1024
NEUTRAL_AUGMENTATION = {
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


def _project_path(value: Path) -> Path:
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _status(run_root: Path) -> str:
    try:
        return str(json.loads((run_root / "status.json").read_text(encoding="utf-8")).get("status", "missing"))
    except (OSError, json.JSONDecodeError):
        return "missing"


def _read_base_suite(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict) or not isinstance(value.get("suite"), dict):
        raise ValueError(f"Suite inválida: {path}")
    suite = value["suite"]
    training = suite.get("training")
    if not isinstance(training, dict):
        raise ValueError("suite.training precisa ser um mapa.")
    if tuple(suite.get("seeds", ())) != (42,):
        raise ValueError("O benchmark requer suite.seeds: [42].")
    if tuple(suite.get("normalizations", ())) != ("unit_interval",):
        raise ValueError("O benchmark requer normalizations: [unit_interval].")
    if tuple(suite.get("balance_modes", ())) != ("all_raw",):
        raise ValueError("O benchmark requer balance_modes: [all_raw].")
    if (suite.get("train_fraction"), suite.get("validation_fraction"), suite.get("test_fraction")) != (0.70, 0.15, 0.15):
        raise ValueError("O benchmark requer o split 70/15/15.")
    if int(training.get("max_epochs", 0)) != EXPECTED_EPOCHS:
        raise ValueError("O benchmark requer exatamente 50 épocas.")
    if int(training.get("batch_size", 0)) != 256:
        raise ValueError("O benchmark requer batch_size: 256.")
    if float(training.get("extra_fraction", -1)) != 0.0 or training.get("augmentation") != NEUTRAL_AUGMENTATION:
        raise ValueError("O benchmark requer all_raw sem augmentation nem exemplos extras.")
    if int(training.get("early_stopping_patience", 0)) < EXPECTED_EPOCHS:
        raise ValueError("early_stopping_patience deve preservar as 50 épocas.")
    if int(training.get("reduce_lr_patience", 0)) < EXPECTED_EPOCHS:
        raise ValueError("reduce_lr_patience deve preservar as 50 épocas.")
    return value


def _write_variant_suite(base: dict[str, Any], destination: Path, output_root: Path, variant: Any) -> None:
    payload = copy.deepcopy(base)
    suite = payload["suite"]
    training = suite["training"]
    suite["output_root"] = str(output_root)
    training["dtype_policy"] = variant.dtype_policy
    training["qat_weight_bits"] = variant.qat_weight_bits
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _prepared_datasets(suite_path: Path, registry_path: Path) -> tuple[Any, Any, Any, Any]:
    """Recreate the fixed local split for conversion and LiteRT evaluation."""

    from tcc_benchmark.config import load_dataset_registry, load_suite_settings
    from tcc_benchmark.data import prepare_run_datasets
    from tcc_benchmark.runner import _augmentation_from_training, _join_source_samples, _load_entry, _validate_fixed_resolution

    settings = load_suite_settings(suite_path)
    registry = load_dataset_registry(registry_path)
    entry = registry["kmnist"]
    materials = _join_source_samples(_load_entry(entry), image_size=_validate_fixed_resolution(entry, settings.training))
    prepared = prepare_run_datasets(
        materials.samples,
        materials.labels,
        image_size=_validate_fixed_resolution(entry, settings.training),
        channels=materials.channels,
        batch_size=settings.training.batch_size,
        normalization_mode="unit_interval",
        balance_mode="all_raw",
        seed=42,
        train_fraction=settings.train_fraction,
        validation_fraction=settings.validation_fraction,
        test_fraction=settings.test_fraction,
        extra_fraction=0.0,
        augmentation=_augmentation_from_training(settings.training),
        preprocess_cache_max_mib=settings.training.preprocess_cache_max_mib,
        shuffle_buffer_max_mib=settings.training.shuffle_buffer_max_mib,
    )
    return settings, entry, materials, prepared


def _save_model_size(artifacts: Path, model: Any, checkpoint: Path, variant: Any, conversion: dict[str, Any] | None) -> dict[str, Any]:
    from tcc_benchmark.quantization import quantized_parameter_size
    from tcc_benchmark.state import atomic_write_json

    bits = 4 if variant.key == "int4_qat" else 8 if variant.key in {"int8_qat", "int8_ptq"} else 16 if variant.key == "fp16" else 32
    payload = {
        "variant": variant.key,
        "serialized_model_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        **quantized_parameter_size(model, weight_bits=bits),
        "efficiency_interpretation": "emulated_int4" if variant.emulated else "measured",
        "quantization_validity": "emulated_int4" if variant.emulated else "not_exported",
    }
    if conversion:
        payload["litert"] = conversion
        payload["quantization_validity"] = conversion.get("quantization_validity", conversion.get("status"))
        if conversion.get("bytes") is not None:
            payload["serialized_litert_bytes"] = conversion["bytes"]
    atomic_write_json(artifacts / "model_size.json", payload)
    return payload


def _postprocess_training_variant(variant: Any, run_root: Path, suite_path: Path, registry_path: Path, *, resume: bool) -> dict[str, Any]:
    from tcc_benchmark.quantization import benchmark_litert, convert_litert_model, write_litert_results
    from tcc_benchmark.state import atomic_write_json

    artifacts = run_root / "artifacts"
    marker = artifacts / "quantization_postprocess.json"
    if resume and marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("status") == "completed":
            return saved
    settings, _entry, materials, prepared = _prepared_datasets(suite_path, registry_path)
    import tensorflow as tf

    checkpoint = run_root / "checkpoints" / "best.keras"
    if not checkpoint.exists():
        checkpoint = run_root / "checkpoints" / "last.keras"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado para {variant.key}: {run_root}")
    model = tf.keras.models.load_model(checkpoint, compile=False)
    mode = "fp32" if variant.key == "fp32" else "fp16" if variant.key == "fp16" else "int8" if variant.key == "int8_qat" else None
    conversion: dict[str, Any] | None = None
    outcome: dict[str, Any] = {"variant": variant.key, "started_at": _timestamp(), "status": "completed"}
    if mode is not None:
        conversion = convert_litert_model(
            model,
            artifacts / f"{variant.key}.tflite",
            mode=mode,
            representative_dataset=prepared.train_ds if mode == "int8" else None,
        )
        if conversion.get("status") == "completed":
            benchmark, labels, logits = benchmark_litert(
                artifacts / f"{variant.key}.tflite", prepared.test_ds, num_classes=materials.num_classes
            )
            benchmark["conversion"] = conversion
            write_litert_results(artifacts, benchmark, labels, logits)
        else:
            outcome["conversion_error"] = conversion.get("error")
    _save_model_size(artifacts, model, checkpoint, variant, conversion)
    outcome["finished_at"] = _timestamp()
    atomic_write_json(marker, outcome)
    tf.keras.backend.clear_session()
    return outcome


def _run_ptq(root: Path, base_suite: Path, registry_path: Path, *, resume: bool) -> dict[str, Any]:
    """Create the no-training INT8-PTQ cell from the completed FP32 checkpoint."""

    from tcc_benchmark.quantization import benchmark_litert, convert_litert_model, variant_for, write_litert_results
    from tcc_benchmark.state import RunPaths, atomic_write_json, initialize_run, set_run_status

    variant = variant_for("int8_ptq")
    fp32_root = root / "fp32" / "kmnist" / "runs" / RUN_ID
    ptq_root = root / "int8_ptq" / "kmnist" / "runs" / RUN_ID
    artifacts = RunPaths.from_root(ptq_root).ensure().artifacts
    marker = artifacts / "quantization_postprocess.json"
    if resume and marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("status") == "completed":
        return json.loads(marker.read_text(encoding="utf-8"))
    checkpoint = fp32_root / "checkpoints" / "best.keras"
    if not checkpoint.exists():
        checkpoint = fp32_root / "checkpoints" / "last.keras"
    if not checkpoint.exists():
        raise FileNotFoundError("INT8-PTQ exige o checkpoint concluído da variante FP32.")
    source_manifest = json.loads((fp32_root / "manifest.json").read_text(encoding="utf-8"))
    config = dict(source_manifest["config"])
    config["quantization_experiment"] = {
        "variant": "int8_ptq",
        "parent_fp32_run": str(fp32_root),
        "training": "none; conversion from completed FP32 checkpoint",
        "calibration_examples": CALIBRATION_EXAMPLES,
    }
    initialize_run(ptq_root, config, split_fingerprint=source_manifest.get("split_fingerprint"))
    set_run_status(ptq_root, "running", detail="Convertendo checkpoint FP32 para INT8 PTQ")
    try:
        _settings, _entry, materials, prepared = _prepared_datasets(base_suite, registry_path)
        import tensorflow as tf

        model = tf.keras.models.load_model(checkpoint, compile=False)
        conversion = convert_litert_model(
            model, artifacts / "int8_ptq.tflite", mode="int8", representative_dataset=prepared.train_ds
        )
        _save_model_size(artifacts, model, checkpoint, variant, conversion)
        if conversion.get("status") == "completed":
            benchmark, labels, logits = benchmark_litert(
                artifacts / "int8_ptq.tflite", prepared.test_ds, num_classes=materials.num_classes
            )
            benchmark["conversion"] = conversion
            write_litert_results(artifacts, benchmark, labels, logits)
            atomic_write_json(
                artifacts / "test_metrics.json",
                {"keras_metrics": {}, "classification": benchmark["classification"], "metric_source": "litert_int8_ptq"},
            )
        marker_payload = {"variant": "int8_ptq", "status": "completed", "finished_at": _timestamp(), "conversion": conversion}
        atomic_write_json(marker, marker_payload)
        set_run_status(ptq_root, "completed", detail="Conversão INT8 PTQ e benchmark LiteRT concluídos")
        tf.keras.backend.clear_session()
        return marker_payload
    except Exception as exc:
        set_run_status(ptq_root, "failed", detail="Falha na conversão INT8 PTQ", error=exc, force=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("configs/kmnist-quantization.yaml"))
    parser.add_argument("--registry", type=Path, default=Path("configs/datasets.wsl.yaml"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Retoma somente runs interrompidas e pós-processamentos pendentes.")
    parser.add_argument("--rerun-failed", action="store_true", help="Reexecuta variantes marcadas como failed.")
    parser.add_argument("--dry-run", action="store_true", help="Valida e gera as quatro suítes, sem TensorFlow.")
    parser.add_argument("--fail-fast", action="store_true", help="Interrompe a fila no primeiro erro.")
    args = parser.parse_args()

    from tcc_benchmark.quantization import TRAINING_VARIANTS, build_comparison_report

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    suite_path = _project_path(args.suite)
    registry_path = _project_path(args.registry)
    if not suite_path.is_file() or not registry_path.is_file():
        parser.error("Suite ou registro de datasets não encontrado.")
    base = _read_base_suite(suite_path)
    configured_root = Path(str(base["suite"]["output_root"]))
    root = _project_path(args.output_root) if args.output_root else configured_root
    root.mkdir(parents=True, exist_ok=True)
    generated = root / "generated-suites"
    queue_status: dict[str, Any] = {
        "experiment": "kmnist-quantization-benchmark",
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "status": "running",
        "protocol": {
            "dataset": "kmnist",
            "seed": 42,
            "split": "stratified 70/15/15",
            "source": "all_raw 70,000 examples",
            "normalization": "unit_interval",
            "augmentation": NEUTRAL_AUGMENTATION,
            "batch_size": 256,
            "epochs": 50,
            "gpu_memory_growth": True,
            "calibration_examples": CALIBRATION_EXAMPLES,
        },
        "variants": {},
    }
    status_path = root / "quantization-status.json"
    failures = 0
    for variant in TRAINING_VARIANTS:
        variant_root = root / variant.key
        generated_suite = generated / f"kmnist-{variant.key}.yaml"
        _write_variant_suite(base, generated_suite, variant_root.resolve(), variant)
        run_root = variant_root / "kmnist" / "runs" / RUN_ID
        record: dict[str, Any] = {"suite": str(generated_suite), "status_before": _status(run_root), "started_at": _timestamp()}
        queue_status["variants"][variant.key] = record
        _write_json(status_path, queue_status)
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
        started = time.perf_counter()
        print(f"[{_timestamp()}] {variant.key}: iniciando", flush=True)
        record["return_code"] = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
        record["duration_seconds"] = round(time.perf_counter() - started, 3)
        record["status_after_training"] = _status(run_root)
        if not args.dry_run and record["return_code"] == 0 and record["status_after_training"] == "completed":
            try:
                record["postprocess"] = _postprocess_training_variant(variant, run_root, generated_suite, registry_path, resume=args.resume)
            except Exception as exc:
                record["postprocess_error"] = repr(exc)
                failures += 1
        elif record["return_code"] != 0:
            failures += 1
        record["finished_at"] = _timestamp()
        queue_status["updated_at"] = _timestamp()
        _write_json(status_path, queue_status)
        print(f"[{_timestamp()}] {variant.key}: {record['status_after_training']} (rc={record['return_code']})", flush=True)
        if failures and args.fail_fast:
            queue_status["status"] = "failed"
            _write_json(status_path, queue_status)
            return 1

    if not args.dry_run and failures == 0:
        try:
            queue_status["variants"]["int8_ptq"] = _run_ptq(root, suite_path, registry_path, resume=args.resume)
            queue_status["comparison"] = {key: str(value) for key, value in build_comparison_report(root).items()}
        except Exception as exc:
            queue_status["variants"]["int8_ptq"] = {"status": "failed", "error": repr(exc)}
            failures += 1
    queue_status["status"] = "completed" if failures == 0 else "completed_with_failures"
    queue_status["failures"] = failures
    queue_status["updated_at"] = _timestamp()
    _write_json(status_path, queue_status)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
