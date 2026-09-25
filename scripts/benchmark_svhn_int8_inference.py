"""Export and benchmark the existing SVHN checkpoint as fully INT8 LiteRT.

This is a deployment-resource benchmark: it uses the trained checkpoint but a
synthetic, z-score-like representative set only to calibrate integer ranges.
It must not be used to report post-quantization accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import tensorflow as tf

try:
    from tensorflow.lite.python import schema_py_generated as tflite_schema
except ImportError:  # pragma: no cover - TensorFlow package layout differs by release.
    tflite_schema = None


DEFAULT_CHECKPOINT = Path(
    "outputs/remaining-ram-capped/svhn/runs/"
    "svhn__zscore__all_raw__seed-42/checkpoints/best.keras"
)
DEFAULT_OUTPUT = Path("outputs/svhn-int8-inference-benchmark-2026-09-16")
INPUT_SHAPE = (1, 64, 64, 3)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot encode {type(value)!r} as JSON")


def representative_dataset(samples: int) -> Any:
    """Yield deterministic inputs with the model's z-score-like distribution."""

    generator = np.random.default_rng(42)
    values = generator.normal(loc=0.0, scale=1.0, size=(samples, *INPUT_SHAPE[1:])).astype(np.float32)
    for value in values:
        yield [value[None, ...]]


def quantize_input(images: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if dtype == np.dtype(np.float32):
        return images.astype(dtype, copy=False)
    scale, zero_point = detail["quantization"]
    if not scale:
        raise RuntimeError(f"Input tensor has no quantization scale: {detail}")
    bounds = np.iinfo(dtype)
    quantized = np.round(images / float(scale) + int(zero_point))
    return np.clip(quantized, bounds.min, bounds.max).astype(dtype)


def float32_export_clone(model: Any) -> Any:
    """Rebuild mixed-FP16 layers in FP32 while preserving their trained weights."""

    def clone_layer(layer: Any) -> Any:
        config = layer.get_config()
        if "dtype" in config:
            config["dtype"] = "float32"
        return layer.__class__.from_config(config)

    clone = tf.keras.models.clone_model(model, clone_function=clone_layer)
    clone.set_weights(model.get_weights())
    return clone


def export_int8(model: Any, destination: Path, calibration_samples: int) -> dict[str, Any]:
    conversion_model = float32_export_clone(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(conversion_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calibration_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    payload = converter.convert()
    destination.write_bytes(payload)
    return {
        "path": destination,
        "bytes": len(payload),
        "calibration": "synthetic_zscore_normal",
        "calibration_samples": calibration_samples,
        "export_graph": "float32 clone of the mixed_float16 checkpoint; trained weights preserved",
    }


def benchmark_litert(model_path: Path, *, iterations: int, warmup: int, threads: int) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    shape = tuple(int(value) for value in input_detail["shape"])
    if shape != INPUT_SHAPE:
        raise RuntimeError(f"Unexpected exported input shape: {shape}; expected {INPUT_SHAPE}")

    random_input = np.random.default_rng(123).normal(0.0, 1.0, size=shape).astype(np.float32)
    input_value = quantize_input(random_input, input_detail)
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)

    def invoke() -> None:
        interpreter.set_tensor(input_detail["index"], input_value)
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])

    for _ in range(warmup):
        invoke()

    elapsed_seconds: list[float] = []
    peak_rss = int(process.memory_info().rss)
    for _ in range(iterations):
        started = time.perf_counter()
        invoke()
        elapsed_seconds.append(time.perf_counter() - started)
        peak_rss = max(peak_rss, int(process.memory_info().rss))

    tensor_details = interpreter.get_tensor_details()
    tensor_dtype_counts: dict[str, int] = {}
    tensor_bytes_by_dtype: dict[str, int] = {}
    tensor_bytes_by_storage = {"constant": 0, "runtime": 0}
    serialized_model = None
    subgraph = None
    if tflite_schema is not None:
        serialized_model = tflite_schema.Model.GetRootAsModel(model_path.read_bytes(), 0)
        subgraph = serialized_model.Subgraphs(0)
    for detail in tensor_details:
        name = np.dtype(detail["dtype"]).name
        tensor_dtype_counts[name] = tensor_dtype_counts.get(name, 0) + 1
        byte_count = int(np.prod(detail["shape"])) * np.dtype(detail["dtype"]).itemsize
        tensor_bytes_by_dtype[name] = tensor_bytes_by_dtype.get(name, 0) + byte_count
        if serialized_model is not None and subgraph is not None:
            tensor = subgraph.Tensors(int(detail["index"]))
            buffer = serialized_model.Buffers(tensor.Buffer())
            storage = "constant" if buffer.DataLength() else "runtime"
            tensor_bytes_by_storage[storage] += byte_count
    try:
        op_names = sorted({str(item.get("op_name", "unknown")) for item in interpreter._get_ops_details()})
    except AttributeError:
        op_names = []
    elapsed_ms = [value * 1_000 for value in elapsed_seconds]
    median_seconds = statistics.median(elapsed_seconds)
    return {
        "runtime": "LiteRT Python CPU",
        "threads": threads,
        "batch_size": shape[0],
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "latency_median_ms": statistics.median(elapsed_ms),
        "latency_p95_ms": float(np.quantile(elapsed_ms, 0.95)),
        "throughput_examples_per_second": 1.0 / median_seconds,
        "process_rss_before_bytes": rss_before,
        "process_rss_peak_bytes": peak_rss,
        "process_rss_delta_bytes": peak_rss - rss_before,
        "declared_tensor_bytes_sum": int(
            sum(int(np.prod(detail["shape"])) * np.dtype(detail["dtype"]).itemsize for detail in tensor_details)
        ),
        "declared_tensor_bytes_by_dtype": tensor_bytes_by_dtype,
        "declared_tensor_bytes_by_storage": tensor_bytes_by_storage if serialized_model is not None else None,
        "tensor_dtypes": tensor_dtype_counts,
        "quantization_validity": "native_int8"
        if tensor_dtype_counts.get("int8") and not any(name.startswith("float") for name in tensor_dtype_counts)
        else "float_or_hybrid",
        "input": {
            "shape": shape,
            "dtype": np.dtype(input_detail["dtype"]).name,
            "quantization": input_detail["quantization"],
        },
        "output": {
            "shape": tuple(int(value) for value in output_detail["shape"]),
            "dtype": np.dtype(output_detail["dtype"]).name,
            "quantization": output_detail["quantization"],
        },
        "operators": op_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = tf.keras.models.load_model(checkpoint, compile=False)
    tflite_path = output / "svhn_int8_ptq.tflite"
    result: dict[str, Any] = {
        "purpose": "deployment-resource benchmark only; no post-quantization accuracy claim",
        "checkpoint": checkpoint,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_total_params": int(model.count_params()),
        "input_shape": INPUT_SHAPE,
        "tensorflow_version": tf.__version__,
        "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "host": {
            "platform": platform.platform(),
            "cpu_count_logical": os.cpu_count(),
            "ram_total_bytes": int(psutil.virtual_memory().total),
        },
    }
    try:
        result["export"] = export_int8(model, tflite_path, args.calibration_samples)
        result["benchmark"] = benchmark_litert(
            tflite_path,
            iterations=args.iterations,
            warmup=args.warmup,
            threads=args.threads,
        )
        result["status"] = "completed"
    except Exception as error:  # Preserve evidence when conversion is not deployable.
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        (output / "benchmark_summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
