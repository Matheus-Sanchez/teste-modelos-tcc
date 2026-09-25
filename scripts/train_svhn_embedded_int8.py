"""Train a LiteRT-friendly SVHN CNN and verify a fully INT8 deployment export.

The architecture intentionally avoids GroupNormalization and swish, whose
lowered operations prevented a fully integer export of the legacy CNN.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import psutil
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


DEFAULT_DATASET = Path("datasets/svhn")
DEFAULT_OUTPUT = Path("outputs/svhn-embedded-int8-2026-09-16")
SEED = 42


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value)!r} as JSON")


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = loadmat(path)
    images = np.transpose(payload["X"], (3, 0, 1, 2)).astype(np.float32) / 255.0
    labels = np.asarray(payload["y"], dtype=np.int64).reshape(-1) % 10
    return images, labels


def make_model() -> tf.keras.Model:
    """Small CNN that exports through standard LiteRT integer kernels."""

    inputs = tf.keras.Input(shape=(32, 32, 3), dtype="float32", name="image")

    def conv_block(value: Any, filters: int, *, pool: bool) -> Any:
        value = tf.keras.layers.SeparableConv2D(
            filters, 3, padding="same", use_bias=False, depth_multiplier=1
        )(value)
        value = tf.keras.layers.BatchNormalization()(value)
        value = tf.keras.layers.ReLU(max_value=6.0)(value)
        if pool:
            value = tf.keras.layers.MaxPooling2D(pool_size=2)(value)
        return value

    x = tf.keras.layers.Conv2D(16, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = conv_block(x, 24, pool=True)
    x = conv_block(x, 32, pool=True)
    x = conv_block(x, 48, pool=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="svhn_embedded_cnn")


def representative_dataset(values: np.ndarray, samples: int) -> Iterator[list[np.ndarray]]:
    selected = values[: int(samples)]
    for image in selected:
        yield [image[None, ...].astype(np.float32, copy=False)]


def export_int8(model: tf.keras.Model, values: np.ndarray, destination: Path, samples: int) -> dict[str, Any]:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(values, samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    destination.write_bytes(converter.convert())
    return {"path": destination, "bytes": destination.stat().st_size, "calibration_samples": samples}


def quantize(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if not scale:
        raise RuntimeError(f"Missing quantization data for input: {detail}")
    dtype = np.dtype(detail["dtype"])
    bounds = np.iinfo(dtype)
    raw = np.round(values / float(scale) + int(zero_point))
    return np.clip(raw, bounds.min, bounds.max).astype(dtype)


def inspect_and_evaluate(model_path: Path, images: np.ndarray, labels: np.ndarray, *, timed_samples: int) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    details = interpreter.get_tensor_details()
    dtype_counts: dict[str, int] = {}
    for detail in details:
        dtype = np.dtype(detail["dtype"]).name
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    fully_int8 = bool(dtype_counts.get("int8")) and not any(
        name.startswith("float") for name in dtype_counts if dtype_counts[name]
    )
    if not fully_int8:
        raise RuntimeError(f"LiteRT export is not fully integer: {dtype_counts}")

    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    predictions = np.empty(len(labels), dtype=np.int64)
    timings: list[float] = []
    peak_rss = rss_before
    for index, image in enumerate(images):
        value = quantize(image[None, ...], input_detail)
        interpreter.set_tensor(input_detail["index"], value)
        started = time.perf_counter() if index < timed_samples else None
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if started is not None:
            timings.append(time.perf_counter() - started)
        predictions[index] = int(np.argmax(output[0]))
        peak_rss = max(peak_rss, int(process.memory_info().rss))

    operation_names = []
    try:
        operation_names = sorted({str(item.get("op_name", "unknown")) for item in interpreter._get_ops_details()})
    except AttributeError:
        pass
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "samples": int(len(labels)),
        "tensor_dtypes": dtype_counts,
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "operators": operation_names,
        "latency_median_ms": float(statistics.median(timings) * 1_000),
        "latency_p95_ms": float(np.quantile(timings, 0.95) * 1_000),
        "throughput_examples_per_second": float(1.0 / statistics.median(timings)),
        "process_rss_before_bytes": rss_before,
        "process_rss_peak_bytes": peak_rss,
        "process_rss_delta_bytes": peak_rss - rss_before,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--timed-samples", type=int, default=500)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    tf.keras.mixed_precision.set_global_policy("float32")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_images, train_labels = load_split(dataset / "train_32x32.mat")
    test_images, test_labels = load_split(dataset / "test_32x32.mat")
    indices = np.arange(len(train_labels))
    train_index, validation_index = train_test_split(
        indices, test_size=0.1, random_state=SEED, stratify=train_labels
    )
    x_train, y_train = train_images[train_index], train_labels[train_index]
    x_validation, y_validation = train_images[validation_index], train_labels[validation_index]
    del train_images, train_labels, indices

    model = make_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", patience=2, factor=0.5, min_lr=1e-5),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    float_loss, float_accuracy = model.evaluate(test_images, test_labels, batch_size=args.batch_size, verbose=0)
    model_path = output / "svhn_embedded_float32.keras"
    model.save(model_path)
    tflite_path = output / "svhn_embedded_int8.tflite"
    export = export_int8(model, x_train, tflite_path, args.calibration_samples)
    int8 = inspect_and_evaluate(tflite_path, test_images, test_labels, timed_samples=args.timed_samples)

    result = {
        "status": "completed",
        "purpose": "fully INT8 LiteRT export and accuracy validation",
        "seed": SEED,
        "dataset": {"train": int(len(x_train)), "validation": int(len(x_validation)), "test": int(len(test_images))},
        "architecture": "Conv2D + BatchNorm + ReLU6 + SeparableConv2D + GAP + Dense; no GroupNormalization",
        "model_total_params": int(model.count_params()),
        "training": {"epochs_completed": len(history.history["loss"]), "history": history.history},
        "float32": {"test_loss": float(float_loss), "test_accuracy": float(float_accuracy), "keras_bytes": model_path.stat().st_size},
        "int8": {**export, **int8, "accuracy_delta_vs_float32": float(int8["accuracy"] - float_accuracy)},
        "runtime": {
            "tensorflow_version": tf.__version__,
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "host": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "ram_total_bytes": int(psutil.virtual_memory().total),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps(result, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
