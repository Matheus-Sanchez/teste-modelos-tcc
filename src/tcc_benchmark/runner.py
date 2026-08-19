"""Orquestração sequencial, retomável e observável da matriz de benchmarks.

This module deliberately keeps the command layer thin and the training protocol
explicit.  It never downloads data, never changes batch size/resolution after a
failure, and creates a durable manifest before TensorFlow starts doing work.
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import io
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .adapters import LoadedDataset, load_image_file, load_local_dataset
from .audit import audit_dataset, write_audit_report
from .config import (
    DATASET_ORDER,
    ConfigurationError,
    DatasetEntry,
    SuiteSettings,
    TrainingSettings,
    fingerprint,
    load_dataset_registry,
    load_suite_settings,
    run_config_payload,
)
from .data import (
    AugmentationConfig,
    PreparedRunDatasets,
    select_samples,
    stratified_split_indices,
)
from .metrics import EvaluationResult, evaluate_model, make_training_callbacks
from .model import build_and_compile_legacy_cnn, set_dtype_policy
from .preflight import run_preflight
from .reporting import build_dataset_report, build_global_index, write_run_report
from .state import (
    RunPaths,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    initialize_run,
    set_run_status,
    stable_run_id,
    update_manifest,
)
from .telemetry import TelemetrySampler, make_keras_telemetry_callback


class RunExecutionError(RuntimeError):
    """A dataset or matrix cell cannot follow the fixed benchmark protocol."""


@dataclasses.dataclass(frozen=True)
class DatasetMaterials:
    """All source examples joined across official splits, before re-splitting."""

    samples: Sequence[Any]
    labels: np.ndarray
    channels: int
    class_names: tuple[str, ...]
    source_manifest: dict[str, Any]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclasses.dataclass(frozen=True)
class RunOutcome:
    """Small, serializable result for one cell of the experiment matrix."""

    run_id: str
    status: str
    action: str
    message: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "failed"


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_settings_and_registry(args: Any) -> tuple[SuiteSettings, dict[str, DatasetEntry]]:
    settings = load_suite_settings(args.suite)
    if getattr(args, "output_root", None) is not None:
        settings = dataclasses.replace(settings, output_root=_resolve_path(args.output_root))
    else:
        settings = dataclasses.replace(settings, output_root=_resolve_path(settings.output_root))
    settings.validate()
    registry = load_dataset_registry(args.registry)
    return settings, registry


def _select_entries(registry: Mapping[str, DatasetEntry], args: Any) -> list[DatasetEntry]:
    dataset = getattr(args, "dataset", None)
    if dataset:
        if dataset not in registry:
            raise ConfigurationError(
                f"'{dataset}' não está configurado no registro local. "
                "Use configs/datasets.yaml para o layout padrão datasets/<nome>, "
                "ou informe um registro alternativo com --registry."
            )
        return [registry[dataset]]
    selected = [registry[name] for name in DATASET_ORDER if name in registry]
    if not selected:
        raise ConfigurationError("O registro local não contém nenhum dos datasets suportados.")
    return selected


def _load_entry(entry: DatasetEntry) -> LoadedDataset:
    dataset = load_local_dataset(entry.adapter, entry.root, **entry.options)
    if dataset.name != entry.name:
        # An adapter alias is harmless, but the directory layout and reports are
        # keyed by the registry name.  Keep that identity explicit in metadata.
        dataset.metadata.setdefault("registry_name", entry.name)
    return dataset


def _infer_channels(sample: Any) -> int:
    if isinstance(sample, (str, Path)):
        image = load_image_file(sample)
    else:
        image = np.asarray(sample)
    if image.ndim == 2:
        return 1
    if image.ndim == 3 and image.shape[-1] in {1, 3}:
        return int(image.shape[-1])
    raise RunExecutionError(
        "Não foi possível inferir os canais nativos da imagem: "
        f"shape recebido={getattr(image, 'shape', None)}."
    )


def _join_source_samples(dataset: LoadedDataset, *, image_size: int) -> DatasetMaterials:
    """Join official partitions while preserving paths lazily where possible."""

    samples: list[Any] = []
    labels: list[int] = []
    saw_paths = False
    saw_arrays = False
    for image_or_path, label, _record in dataset.iter_samples(load_images=False):
        if isinstance(image_or_path, Path):
            saw_paths = True
            samples.append(str(image_or_path))
        else:
            saw_arrays = True
            samples.append(np.asarray(image_or_path))
        labels.append(int(label))
    if not samples:
        raise RunExecutionError(f"O dataset '{dataset.name}' não contém exemplos.")

    # Standard adapters are homogeneous.  Decoding only the exceptional mixed
    # representation avoids asking tf.data to infer an unsafe union dtype.
    if saw_paths and saw_arrays:
        samples = [load_image_file(item) if isinstance(item, (str, Path)) else np.asarray(item) for item in samples]

    label_values = np.asarray(labels, dtype=np.int64)
    expected_labels = set(range(dataset.num_classes))
    observed_labels = set(int(value) for value in np.unique(label_values))
    if observed_labels != expected_labels:
        missing = sorted(expected_labels - observed_labels)
        invalid = sorted(observed_labels - expected_labels)
        raise RunExecutionError(
            f"Classes inválidas/incompletas em '{dataset.name}': ausentes={missing}, fora_do_intervalo={invalid}. "
            "Execute 'audit' para localizar os arquivos problemáticos."
        )

    channels = _infer_channels(samples[0])
    source_manifest = dataset.to_manifest_dict()
    source_manifest["joined_samples"] = int(len(samples))
    source_manifest["native_channels"] = channels
    source_manifest["target_size"] = int(image_size)
    return DatasetMaterials(
        samples=samples,
        labels=label_values,
        channels=channels,
        class_names=tuple(dataset.class_names),
        source_manifest=source_manifest,
    )


def _validate_fixed_resolution(entry: DatasetEntry, training: TrainingSettings) -> int:
    """Read resolution solely from the central training configuration."""

    image_size = int(training.image_size_for(entry.name))
    if image_size < 64:
        raise RunExecutionError(f"A resolução de treino para '{entry.name}' deve ser pelo menos 64.")
    return image_size


def _augmentation_from_training(training: TrainingSettings) -> AugmentationConfig:
    """Build augmentation exclusively from the central training settings."""

    return AugmentationConfig(**dict(training.augmentation))


def _build_run_config(
    *,
    settings: SuiteSettings,
    entry: DatasetEntry,
    materials: DatasetMaterials,
    normalization: str,
    balance_mode: str,
    seed: int,
    image_size: int,
    smoke: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = run_config_payload(settings, entry, normalization, balance_mode, seed)
    config.update(
        {
            "target_size": int(image_size),
            "native_channels": int(materials.channels),
            "num_classes": int(materials.num_classes),
            "class_names": list(materials.class_names),
            "source_manifest": dict(materials.source_manifest),
            "source_fingerprint": fingerprint(materials.source_manifest),
            "protocol": {
                "training_from_scratch": True,
                "pretrained_weights": False,
                "input_channels": "native",
                "resize": "proportional_padding",
                "evaluation_augmentation": False,
                "test_time_augmentation": False,
                "optimizer": "Adam",
                "dtype_policy": settings.training.dtype_policy,
                "hidden_activation": settings.training.hidden_activation,
                "qat_weight_bits": settings.training.qat_weight_bits,
                "op_determinism": settings.training.qat_weight_bits is None,
                "checkpoint_monitor": "val_macro_f1",
                "early_stopping": {
                    "patience": settings.training.early_stopping_patience,
                    "restore_best_weights": True,
                },
                "reduce_lr_on_plateau": {
                    "factor": settings.training.reduce_lr_factor,
                    "patience": settings.training.reduce_lr_patience,
                    "min_lr": settings.training.reduce_lr_min_lr,
                },
            },
        }
    )
    if smoke:
        config["smoke"] = dict(smoke)
    return config


def _execution_action(status: str, *, resume: bool, rerun_failed: bool, dry_run: bool) -> str:
    """Return execute/skip according to durable state and explicit user intent."""

    if dry_run:
        return "planned"
    if status == "pending":
        return "execute"
    if status == "completed":
        return "skip_completed"
    if status in {"interrupted", "running"}:
        return "execute" if resume else "skip_needs_resume"
    if status == "failed":
        return "execute" if rerun_failed else "skip_failed"
    return "skip_unknown_state"


def _configure_tensorflow_runtime(training: TrainingSettings) -> dict[str, Any]:
    """Configure only memory growth; no adaptive training parameter is changed."""

    from .data import require_tensorflow

    # This must be set before the first TensorFlow device initialization. The
    # command entrypoints set it before preflight as well; keeping it here makes
    # direct/private callers deterministic.
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tensorflow = require_tensorflow()
    set_dtype_policy(training.dtype_policy)
    physical_gpus = list(tensorflow.config.list_physical_devices("GPU"))
    memory_growth: list[dict[str, Any]] = []
    for gpu in physical_gpus:
        try:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
            memory_growth.append({"device": gpu.name, "enabled": True})
        except (RuntimeError, ValueError) as exc:
            # It is common for a resumed process to have initialized the device
            # already.  Record the fact; it does not alter benchmark settings.
            memory_growth.append({"device": gpu.name, "enabled": False, "reason": str(exc)})
    return {
        "tensorflow_version": getattr(tensorflow, "__version__", None),
        "dtype_policy": tensorflow.keras.mixed_precision.global_policy().name,
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "physical_gpus": [device.name for device in physical_gpus],
        "memory_growth": memory_growth,
    }


def _tensorflow_gpu_memory_info() -> dict[str, Any]:
    """Read TensorFlow allocator counters separately from global NVML telemetry."""

    try:
        from .data import require_tensorflow

        tensorflow = require_tensorflow()
        values: dict[str, Any] = {}
        for index, _gpu in enumerate(tensorflow.config.list_logical_devices("GPU")):
            try:
                info = tensorflow.config.experimental.get_memory_info(f"GPU:{index}")
                values[f"GPU:{index}"] = {str(key): int(value) for key, value in info.items()}
            except (AttributeError, RuntimeError, ValueError):
                continue
        return values
    except Exception:
        return {}


def _epoch_rows(callbacks: Sequence[Any]) -> list[dict[str, Any]]:
    for callback in callbacks:
        rows = getattr(callback, "rows", None)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _write_predictions(path: Path, evaluation: EvaluationResult) -> Path:
    labels = np.asarray(evaluation.y_true, dtype=np.int64)
    probabilities = np.asarray(evaluation.probabilities, dtype=np.float32)
    predicted = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities.max(axis=1)
    destination = io.StringIO(newline="")
    writer = csv.DictWriter(destination, fieldnames=("sample_index", "true_label", "predicted_label", "confidence"))
    writer.writeheader()
    for index, (truth, prediction, score) in enumerate(zip(labels, predicted, confidence, strict=True)):
        writer.writerow(
            {
                "sample_index": index,
                "true_label": int(truth),
                "predicted_label": int(prediction),
                "confidence": float(score),
            }
        )
    return atomic_write_text(path, destination.getvalue())


def _copy_epoch_log(paths: RunPaths) -> None:
    source = paths.checkpoints / "epoch_metrics.csv"
    if source.exists():
        atomic_write_text(paths.logs / "epoch_metrics.csv", source.read_text(encoding="utf-8"))


def _safe_stop_sampler(sampler: TelemetrySampler | None, *, final_event: str) -> dict[str, Any]:
    if sampler is None:
        return {}
    try:
        return sampler.stop(final_event=final_event)
    except Exception as exc:  # Telemetry must not hide a training failure.
        return {"telemetry_stop_error": repr(exc)}


def _clear_tensorflow_session() -> None:
    try:
        from .data import require_tensorflow

        require_tensorflow().keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def _run_cell(
    *,
    settings: SuiteSettings,
    entry: DatasetEntry,
    materials: DatasetMaterials,
    normalization: str,
    balance_mode: str,
    seed: int,
    resume: bool,
    rerun_failed: bool,
    dry_run: bool,
    smoke: Mapping[str, Any] | None = None,
) -> RunOutcome:
    """Plan or execute exactly one matrix cell, preserving all fixed settings."""

    image_size = _validate_fixed_resolution(entry, settings.training)
    split = stratified_split_indices(
        materials.labels,
        seed=int(seed),
        train_fraction=settings.train_fraction,
        validation_fraction=settings.validation_fraction,
        test_fraction=settings.test_fraction,
    )
    config = _build_run_config(
        settings=settings,
        entry=entry,
        materials=materials,
        normalization=normalization,
        balance_mode=balance_mode,
        seed=seed,
        image_size=image_size,
        smoke=smoke,
    )
    run_id = stable_run_id(entry.name, normalization, balance_mode, seed)
    run_dir = settings.output_root / entry.name / "runs" / run_id
    paths = RunPaths.from_root(run_dir).ensure()
    manifest = initialize_run(paths.root, config, run_id=run_id, split_fingerprint=split.fingerprint())
    action = _execution_action(
        str(manifest.get("status", "pending")),
        resume=resume,
        rerun_failed=rerun_failed,
        dry_run=dry_run,
    )
    if action != "execute":
        if action == "planned":
            update_manifest(
                paths.root,
                {
                    "planning": {
                        "split": split.to_dict(),
                        "source_manifest": materials.source_manifest,
                        "dry_run": True,
                    }
                },
                expected_config=config,
            )
            return RunOutcome(run_id, str(manifest.get("status", "pending")), action, "job validado sem TensorFlow")
        return RunOutcome(run_id, str(manifest.get("status", "unknown")), action)

    set_run_status(
        paths.root,
        "running",
        detail="Retomando" if manifest.get("status") in {"interrupted", "running"} else "Iniciando",
        extra={
            "planning": {
                "split": split.to_dict(),
                "source_manifest": materials.source_manifest,
                "dry_run": False,
            }
        },
    )

    sampler: TelemetrySampler | None = None
    started = time.perf_counter()
    runtime: dict[str, Any] = {}
    try:
        runtime = _configure_tensorflow_runtime(settings.training)
        runtime["tensorflow_gpu_memory_before"] = _tensorflow_gpu_memory_info()
        sampler = TelemetrySampler(
            paths.telemetry,
            interval_seconds=settings.hardware_interval_seconds,
            disk_paths={"data": entry.root, "output": paths.root},
        ).start()
        sampler.capture("data_preparation_start")

        from .data import prepare_run_datasets

        prepared: PreparedRunDatasets = prepare_run_datasets(
            materials.samples,
            materials.labels,
            image_size=image_size,
            channels=materials.channels,
            batch_size=settings.training.batch_size,
            normalization_mode=normalization,
            balance_mode=balance_mode,
            seed=int(seed),
            train_fraction=settings.train_fraction,
            validation_fraction=settings.validation_fraction,
            test_fraction=settings.test_fraction,
            extra_fraction=settings.training.extra_fraction,
            augmentation=_augmentation_from_training(settings.training),
            preprocess_cache_max_mib=settings.training.preprocess_cache_max_mib,
            shuffle_buffer_max_mib=settings.training.shuffle_buffer_max_mib,
        )
        if prepared.split.fingerprint() != split.fingerprint():
            raise RunExecutionError("O fingerprint do split preparado diverge do planejamento salvo.")
        sampler.capture("data_preparation_end", metadata={"data_metadata": prepared.metadata})
        update_manifest(
            paths.root,
            {
                "data_metadata": prepared.metadata,
                "split_fingerprint": prepared.split.fingerprint(),
                "runtime": runtime,
            },
            expected_config=config,
        )

        model = build_and_compile_legacy_cnn(
            image_size=image_size,
            channels=materials.channels,
            num_classes=materials.num_classes,
            learning_rate=settings.training.learning_rate,
            dtype_policy=settings.training.dtype_policy,
            hidden_activation=settings.training.hidden_activation,
            qat_weight_bits=settings.training.qat_weight_bits,
            seed=int(seed),
        )
        model_lines: list[str] = []
        model.summary(print_fn=model_lines.append)
        atomic_write_text(paths.logs / "model_summary.txt", "\n".join(model_lines) + "\n")

        callbacks = make_training_callbacks(
            validation_data=prepared.validation_ds,
            num_classes=materials.num_classes,
            run_dir=paths.checkpoints,
            training=settings.training,
            include_backup_restore=True,
            train_examples_per_epoch=int(prepared.metadata["datasets"]["train"]["total_examples"]),
        )
        callbacks.append(make_keras_telemetry_callback(sampler, manage_lifecycle=False))

        fit_started = time.perf_counter()
        train_steps = int(prepared.metadata["datasets"]["train"]["batches"])
        validation_steps = int(prepared.metadata["datasets"]["validation"]["batches"])
        history = model.fit(
            prepared.train_ds,
            validation_data=prepared.validation_ds,
            epochs=settings.training.max_epochs,
            steps_per_epoch=train_steps,
            validation_steps=validation_steps,
            class_weight=prepared.class_weights,
            callbacks=callbacks,
            # ``verbose=1`` keeps Keras' live progress bar visible: it shows
            # batches/steps, loss and accuracy while an epoch is running, then
            # the validation metrics at its end.  The former ``verbose=2``
            # printed only one compact line per epoch, which hid progress on
            # the long benchmark runs.
            verbose=settings.training.keras_verbose,
        )
        fit_seconds = time.perf_counter() - fit_started
        epoch_rows = _epoch_rows(callbacks)
        _copy_epoch_log(paths)
        atomic_write_json(paths.logs / "keras_history.json", {str(key): list(value) for key, value in history.history.items()})

        # Evaluate with the checkpoint selected by validation Macro-F1, never
        # with test-time augmentation.  It remains an untrained-from-scratch
        # model; loading only restores this run's own checkpoint.
        from .data import require_tensorflow
        from .model import compile_legacy_cnn

        best_path = paths.checkpoints / "best.keras"
        selected_model = model
        if best_path.exists():
            tensorflow = require_tensorflow()
            selected_model = tensorflow.keras.models.load_model(str(best_path), compile=True)
            # Older Keras installations occasionally restore a model without a
            # compiled metric object.  Recompiling leaves its learned weights untouched.
            if not getattr(selected_model, "optimizer", None):
                selected_model = compile_legacy_cnn(
                    selected_model,
                    learning_rate=settings.training.learning_rate,
                    dtype_policy=settings.training.dtype_policy,
                )

        sampler.capture("evaluation_start")
        evaluation_started = time.perf_counter()
        evaluation = evaluate_model(selected_model, prepared.test_ds, num_classes=materials.num_classes)
        evaluation_seconds = time.perf_counter() - evaluation_started
        sampler.capture("evaluation_end", evaluation_seconds=evaluation_seconds)

        epoch_seconds = [
            float(row["epoch_seconds"])
            for row in epoch_rows
            if isinstance(row.get("epoch_seconds"), (float, int)) and math.isfinite(float(row["epoch_seconds"]))
        ]
        total_epoch_seconds = float(sum(epoch_seconds)) if epoch_seconds else float(fit_seconds)
        epoch_throughputs = [
            float(row["train_examples_per_second"])
            for row in epoch_rows
            if isinstance(row.get("train_examples_per_second"), (float, int))
            and math.isfinite(float(row["train_examples_per_second"]))
        ]
        training_summary = {
            # This is deliberately the sum of epoch durations saved across the
            # whole run history.  Unlike wall clock of the current process it
            # remains meaningful after BackupAndRestore resumes an interruption.
            "total_seconds": total_epoch_seconds,
            "training_seconds": total_epoch_seconds,
            "fit_seconds_current_attempt": fit_seconds,
            "wall_seconds_current_attempt": time.perf_counter() - started,
            "evaluation_seconds": evaluation_seconds,
            "epochs_completed": len(epoch_rows) or len(history.history.get("loss", [])),
            "mean_epoch_seconds": float(np.mean(epoch_seconds)) if epoch_seconds else None,
            "mean_train_examples_per_second": float(np.mean(epoch_throughputs)) if epoch_throughputs else None,
            "median_train_examples_per_second": float(np.median(epoch_throughputs)) if epoch_throughputs else None,
            "max_epochs": int(settings.training.max_epochs),
            "selected_checkpoint": str(best_path if best_path.exists() else paths.checkpoints / "last.keras"),
            "tensorflow_gpu_memory_after": _tensorflow_gpu_memory_info(),
        }
        telemetry_summary = _safe_stop_sampler(sampler, final_event="run_completed")
        sampler = None
        atomic_write_json(paths.logs / "training_summary.json", training_summary)
        test_payload = {
            "keras_metrics": evaluation.keras_metrics,
            "classification": evaluation.classification.to_dict(),
        }
        atomic_write_json(paths.artifacts / "test_metrics.json", test_payload)
        logits_buffer = io.BytesIO()
        np.save(logits_buffer, evaluation.logits.astype(np.float32))
        atomic_write_bytes(paths.artifacts / "logits.npy", logits_buffer.getvalue())
        _write_predictions(paths.artifacts / "predictions.csv", evaluation)
        write_run_report(
            paths.root,
            y_true=evaluation.y_true,
            y_pred=evaluation.probabilities,
            class_names=materials.class_names,
            test_metrics=evaluation.keras_metrics,
            history=epoch_rows or history.history,
            training_summary=training_summary,
            telemetry_summary=telemetry_summary,
            config=config,
        )
        set_run_status(
            paths.root,
            "completed",
            detail="Treino e avaliação concluídos",
            extra={
                "training": training_summary,
                "test_metrics": test_payload,
                "telemetry": telemetry_summary,
            },
        )
        return RunOutcome(run_id, "completed", "executed")
    except KeyboardInterrupt:
        telemetry_summary = _safe_stop_sampler(sampler, final_event="interrupted")
        sampler = None
        set_run_status(
            paths.root,
            "interrupted",
            detail="Interrompida pelo usuário; BackupAndRestore preserva a última época concluída.",
            extra={"runtime": runtime, "telemetry": telemetry_summary},
            force=True,
        )
        raise
    except Exception as exc:
        telemetry_summary = _safe_stop_sampler(sampler, final_event="failed")
        sampler = None
        error_text = traceback.format_exc()
        atomic_write_text(paths.logs / "error.txt", error_text)
        detail = "Falhou sem alterar batch, resolução ou hiperparâmetros."
        if "resourceexhausted" in repr(exc).lower() or "out of memory" in repr(exc).lower():
            detail = "Falhou por OOM; batch, resolução e hiperparâmetros foram mantidos fixos."
        set_run_status(
            paths.root,
            "failed",
            detail=detail,
            error=exc,
            extra={"runtime": runtime, "telemetry": telemetry_summary},
            force=True,
        )
        return RunOutcome(run_id, "failed", "failed", str(exc))
    finally:
        if sampler is not None:
            _safe_stop_sampler(sampler, final_event="cleanup")
        _clear_tensorflow_session()


def _report_dataset(settings: SuiteSettings, entry: DatasetEntry) -> dict[str, Path]:
    dataset_dir = settings.output_root / entry.name
    return build_dataset_report(dataset_dir, dataset_name=entry.name)


def _seeds_for_dataset(settings: SuiteSettings, entry: DatasetEntry) -> tuple[int, ...]:
    """Return the explicitly configured replications for one dataset."""
    return tuple(settings.dataset_seeds.get(entry.name, settings.seeds))


def _run_dataset_matrix(
    *,
    settings: SuiteSettings,
    entry: DatasetEntry,
    resume: bool,
    rerun_failed: bool,
    dry_run: bool,
    fail_fast: bool,
    smoke: Mapping[str, Any] | None = None,
) -> list[RunOutcome]:
    dataset = _load_entry(entry)
    materials = _join_source_samples(
        dataset,
        image_size=_validate_fixed_resolution(entry, settings.training),
    )
    outcomes: list[RunOutcome] = []
    for normalization in settings.normalizations:
        for balance_mode in settings.balance_modes:
            for seed in _seeds_for_dataset(settings, entry):
                outcome = _run_cell(
                    settings=settings,
                    entry=entry,
                    materials=materials,
                    normalization=str(normalization),
                    balance_mode=str(balance_mode),
                    seed=int(seed),
                    resume=resume,
                    rerun_failed=rerun_failed,
                    dry_run=dry_run,
                    smoke=smoke,
                )
                outcomes.append(outcome)
                print(f"[{entry.name}] {outcome.run_id}: {outcome.action} ({outcome.status})")
                if outcome.failed and fail_fast:
                    return outcomes
    _report_dataset(settings, entry)
    return outcomes


def _write_training_preflight(settings: SuiteSettings, entries: Iterable[DatasetEntry]) -> dict[str, Any]:
    report = run_preflight(
        output_root=settings.output_root,
        data_paths=[entry.root for entry in entries],
        require_tensorflow=True,
    )
    report["training"] = settings.training.to_dict()
    atomic_write_json(settings.output_root / "preflight.json", report)
    if not report.get("ok", False):
        errors = "; ".join(str(item) for item in report.get("errors", []))
        raise RunExecutionError(f"Preflight falhou: {errors}")
    return report


def command_audit(args: Any) -> int:
    settings, registry = _load_settings_and_registry(args)
    entries = _select_entries(registry, args)
    failures = 0
    max_samples = getattr(args, "max_samples", None)
    labels_only = bool(getattr(args, "labels_only", False))
    no_hash = bool(getattr(args, "no_hash", False))
    for entry in entries:
        try:
            dataset = _load_entry(entry)
            report = audit_dataset(
                dataset,
                verify_images=not labels_only,
                hash_images=not labels_only and not no_hash,
                max_samples=max_samples,
            )
            path = write_audit_report(report, settings.output_root / entry.name / "audit" / "audit.json")
            print(
                f"[{entry.name}] auditado: {report.checked_samples}/{report.total_samples} exemplos, "
                f"{report.error_count} erro(s), {report.warning_count} aviso(s): {path}"
            )
            if not report.is_healthy:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"[{entry.name}] auditoria falhou: {exc}")
    return 1 if failures else 0


def command_run(args: Any) -> int:
    # Set before preflight imports TensorFlow, otherwise its first device query
    # can reserve all VRAM and make memory growth unavailable to the runner.
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    settings, registry = _load_settings_and_registry(args)
    entries = _select_entries(registry, args)
    if not args.dry_run:
        _write_training_preflight(settings, entries)

    outcomes: list[RunOutcome] = []
    for entry in entries:
        try:
            outcomes.extend(
                _run_dataset_matrix(
                    settings=settings,
                    entry=entry,
                    resume=bool(args.resume),
                    rerun_failed=bool(args.rerun_failed),
                    dry_run=bool(args.dry_run),
                    fail_fast=bool(args.fail_fast),
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[{entry.name}] não foi possível executar a matriz: {exc}")
            outcomes.append(RunOutcome(entry.name, "failed", "dataset_failed", str(exc)))
            if args.fail_fast:
                break
    build_global_index(settings.output_root)
    failures = sum(outcome.failed for outcome in outcomes)
    print(f"Finalizado: {len(outcomes)} job(s), {failures} falha(s). Saída: {settings.output_root}")
    return 1 if failures else 0


def command_report(args: Any) -> int:
    output_root = _resolve_path(args.output_root)
    dataset_dir = output_root / args.dataset
    artifacts = build_dataset_report(dataset_dir, dataset_name=args.dataset)
    build_global_index(output_root)
    print(f"Relatório reconstruído: {artifacts['html']}")
    return 0


def _subset_for_smoke(materials: DatasetMaterials, *, examples_per_class: int, seed: int) -> DatasetMaterials:
    if examples_per_class < 3:
        raise RunExecutionError("Smoke exige ao menos 3 exemplos por classe para manter o split 70/15/15.")
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for label in range(materials.num_classes):
        candidates = np.flatnonzero(materials.labels == label)
        if len(candidates) < examples_per_class:
            raise RunExecutionError(
                f"Smoke pediu {examples_per_class} exemplos, mas a classe {label} possui apenas {len(candidates)}."
            )
        selected.append(rng.permutation(candidates)[:examples_per_class])
    indices = rng.permutation(np.concatenate(selected)).astype(np.int64, copy=False)
    manifest = dict(materials.source_manifest)
    manifest["smoke_subset"] = {
        "examples_per_class": int(examples_per_class),
        "seed": int(seed),
        "samples": int(len(indices)),
        "selection_fingerprint": fingerprint(indices.astype(int).tolist()),
    }
    return DatasetMaterials(
        samples=select_samples(materials.samples, indices),
        labels=materials.labels[indices],
        channels=materials.channels,
        class_names=materials.class_names,
        source_manifest=manifest,
    )


def command_smoke(args: Any) -> int:
    if int(args.epochs) < 1 or int(args.examples_per_class) < 3:
        raise RunExecutionError("--epochs deve ser positivo e --examples-per-class deve ser pelo menos 3.")
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    settings, registry = _load_settings_and_registry(args)
    if args.dataset not in registry:
        raise ConfigurationError(f"'{args.dataset}' não está configurado no registro local.")
    # Keep smoke artifacts outside the actual matrix, with exactly one compact
    # cell.  The model/batch/resolution/augmentation remain protocol-identical.
    settings = dataclasses.replace(
        settings,
        output_root=settings.output_root / "smoke",
        seeds=(42,),
        normalizations=("unit_interval",),
        balance_modes=("all_raw",),
        training=dataclasses.replace(settings.training, max_epochs=int(args.epochs)),
    )
    entry = registry[args.dataset]
    _write_training_preflight(settings, [entry])
    dataset = _load_entry(entry)
    materials = _subset_for_smoke(
        _join_source_samples(
            dataset,
            image_size=_validate_fixed_resolution(entry, settings.training),
        ),
        examples_per_class=int(args.examples_per_class),
        seed=42,
    )
    outcome = _run_cell(
        settings=settings,
        entry=entry,
        materials=materials,
        normalization="unit_interval",
        balance_mode="all_raw",
        seed=42,
        resume=True,
        rerun_failed=True,
        dry_run=False,
        smoke={"examples_per_class": int(args.examples_per_class), "epochs": int(args.epochs)},
    )
    _report_dataset(settings, entry)
    print(f"Smoke [{entry.name}]: {outcome.action} ({outcome.status})")
    return 1 if outcome.failed else 0


__all__ = [
    "DatasetMaterials",
    "RunExecutionError",
    "RunOutcome",
    "command_audit",
    "command_report",
    "command_run",
    "command_smoke",
]
