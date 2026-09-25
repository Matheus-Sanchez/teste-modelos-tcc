"""Profile LiteRT tensor declarations and single-image CPU latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as tflite_schema


def quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if dtype == np.dtype(np.float32):
        return values.astype(dtype, copy=False)
    scale, zero_point = detail["quantization"]
    raw = np.round(values / float(scale) + int(zero_point))
    bounds = np.iinfo(dtype)
    return np.clip(raw, bounds.min, bounds.max).astype(dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    model_path = args.model.resolve()
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    shape = tuple(int(value) for value in input_detail["shape"])
    values = np.random.default_rng(42).uniform(0, 1, size=shape).astype(np.float32)
    encoded = quantize(values, input_detail)
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)

    def invoke() -> None:
        interpreter.set_tensor(input_detail["index"], encoded)
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])

    for _ in range(30):
        invoke()
    timings = []
    peak_rss = int(process.memory_info().rss)
    for _ in range(args.iterations):
        started = time.perf_counter()
        invoke()
        timings.append(time.perf_counter() - started)
        peak_rss = max(peak_rss, int(process.memory_info().rss))

    schema_model = tflite_schema.Model.GetRootAsModel(model_path.read_bytes(), 0)
    graph = schema_model.Subgraphs(0)
    by_dtype: dict[str, int] = {}
    by_storage = {"constant": 0, "runtime": 0}
    for detail in interpreter.get_tensor_details():
        index = int(detail["index"])
        tensor = graph.Tensors(index)
        byte_count = int(np.prod(detail["shape"])) * np.dtype(detail["dtype"]).itemsize
        dtype = np.dtype(detail["dtype"]).name
        by_dtype[dtype] = by_dtype.get(dtype, 0) + byte_count
        storage = "constant" if schema_model.Buffers(tensor.Buffer()).DataLength() else "runtime"
        by_storage[storage] += byte_count
    output = {
        "model_bytes": model_path.stat().st_size,
        "input_shape": shape,
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "declared_tensor_bytes_by_dtype": by_dtype,
        "declared_tensor_bytes_by_storage": by_storage,
        "note": "Runtime bytes are the sum of declared tensors, not the TFLite Micro tensor arena; buffer reuse can lower its final allocation.",
        "latency_median_ms": float(statistics.median(timings) * 1000),
        "latency_p95_ms": float(np.quantile(timings, 0.95) * 1000),
        "throughput_examples_per_second": float(1 / statistics.median(timings)),
        "process_rss_delta_bytes": peak_rss - rss_before,
    }
    args.output.resolve().write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
