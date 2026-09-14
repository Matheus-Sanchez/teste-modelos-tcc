"""Quantization helpers shared by the reproducible KMNIST experiment.

The project deliberately implements its small QAT surface with TensorFlow fake
quant operations instead of TensorFlow Model Optimization.  The latter still
requires legacy Keras in the supported release line, whereas this project uses
the Keras bundled with TensorFlow 2.21.  The 4-bit path is therefore an
*emulation* for numerical robustness research, not a claim of a deployable
INT4 LiteRT model.
"""

from __future__ import annotations

import csv
import dataclasses
import html
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .state import atomic_write_bytes, atomic_write_json, atomic_write_text


@dataclasses.dataclass(frozen=True)
class QuantizationVariant:
    """One named cell in the fixed quantization experiment."""

    key: str
    label: str
    dtype_policy: str
    qat_weight_bits: int | None = None
    post_training: bool = False
    emulated: bool = False

    @property
    def trains(self) -> bool:
        return not self.post_training


VARIANTS: tuple[QuantizationVariant, ...] = (
    QuantizationVariant("fp32", "CNN FP32", "float32"),
    QuantizationVariant("fp16", "CNN FP16", "mixed_float16"),
    QuantizationVariant("int8_qat", "CNN INT8 QAT", "float32", qat_weight_bits=8),
    QuantizationVariant("int4_qat", "CNN INT4 QAT (emulado)", "float32", qat_weight_bits=4, emulated=True),
    QuantizationVariant("int8_ptq", "CNN INT8 PTQ", "float32", post_training=True),
)
VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}
TRAINING_VARIANTS = tuple(variant for variant in VARIANTS if variant.trains)


def variant_for(value: str) -> QuantizationVariant:
    try:
        return VARIANT_BY_KEY[str(value)]
    except KeyError as exc:
        raise ValueError(f"Variante de quantização desconhecida: {value!r}.") from exc


try:  # Keep report/audit imports usable when the TensorFlow profile is absent.
    import tensorflow as _tf
except Exception:  # pragma: no cover - depends on local TensorFlow installation
    _tf = None


if _tf is not None:

    @_tf.keras.utils.register_keras_serializable(package="tcc_benchmark")
    class FakeQuantize(_tf.keras.layers.Layer):
        """Moving-range fake activation quantizer used during QAT.

        Variables remain float32 and gradients use TensorFlow's straight-through
        estimator.  Inference reuses the calibrated moving range recorded during
        training, which makes saved checkpoints deterministic and serializable.
        """

        def __init__(self, *, num_bits: int = 8, momentum: float = 0.99, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            if int(num_bits) not in {4, 8}:
                raise ValueError("FakeQuantize aceita apenas 4 ou 8 bits neste protocolo.")
            self.num_bits = int(num_bits)
            self.momentum = float(momentum)

        def build(self, input_shape: Any) -> None:
            self.minimum = self.add_weight(
                name="minimum", shape=(), dtype="float32", trainable=False, initializer=_tf.keras.initializers.Constant(-6.0)
            )
            self.maximum = self.add_weight(
                name="maximum", shape=(), dtype="float32", trainable=False, initializer=_tf.keras.initializers.Constant(6.0)
            )
            super().build(input_shape)

        def call(self, inputs: Any, training: bool | None = None) -> Any:
            values = _tf.cast(inputs, _tf.float32)
            if training is True:
                observed_minimum = _tf.minimum(_tf.reduce_min(values), _tf.constant(-1e-6, dtype=_tf.float32))
                observed_maximum = _tf.maximum(_tf.reduce_max(values), _tf.constant(1e-6, dtype=_tf.float32))
                self.minimum.assign(self.momentum * self.minimum + (1.0 - self.momentum) * observed_minimum)
                self.maximum.assign(self.momentum * self.maximum + (1.0 - self.momentum) * observed_maximum)
            return _tf.quantization.fake_quant_with_min_max_vars(
                values,
                min=self.minimum,
                max=self.maximum,
                num_bits=self.num_bits,
                narrow_range=False,
            )

        def get_config(self) -> dict[str, Any]:
            return {**super().get_config(), "num_bits": self.num_bits, "momentum": self.momentum}


    def _fake_quantize_weight(values: Any, *, num_bits: int) -> Any:
        """Symmetric fake quantization for a trainable kernel tensor."""

        values = _tf.cast(values, _tf.float32)
        bound = _tf.maximum(_tf.reduce_max(_tf.abs(values)), _tf.constant(1e-6, dtype=_tf.float32))
        return _tf.quantization.fake_quant_with_min_max_vars(
            values,
            min=-bound,
            max=bound,
            num_bits=int(num_bits),
            narrow_range=True,
        )


    @_tf.keras.utils.register_keras_serializable(package="tcc_benchmark")
    class QATSeparableConv2D(_tf.keras.layers.Layer):
        """SeparableConv2D-equivalent layer with W4/W8 fake quantized kernels."""

        def __init__(
            self,
            filters: int,
            kernel_size: int | tuple[int, int],
            *,
            strides: int | tuple[int, int] = 1,
            padding: str = "same",
            use_bias: bool = False,
            depthwise_initializer: Any = "he_normal",
            pointwise_initializer: Any = "he_normal",
            weight_bits: int = 8,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            if int(weight_bits) not in {4, 8}:
                raise ValueError("QATSeparableConv2D aceita weight_bits 4 ou 8.")
            self.filters = int(filters)
            self.kernel_size = self._pair(kernel_size, "kernel_size")
            self.strides = self._pair(strides, "strides")
            self.padding = str(padding).upper()
            self.use_bias = bool(use_bias)
            self.depthwise_initializer = _tf.keras.initializers.get(depthwise_initializer)
            self.pointwise_initializer = _tf.keras.initializers.get(pointwise_initializer)
            self.weight_bits = int(weight_bits)

        @staticmethod
        def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
            if isinstance(value, int):
                return (int(value), int(value))
            pair = tuple(int(item) for item in value)
            if len(pair) != 2 or any(item < 1 for item in pair):
                raise ValueError(f"{name} deve ser um inteiro positivo ou uma dupla positiva.")
            return pair

        def build(self, input_shape: Any) -> None:
            channels = int(input_shape[-1])
            if channels < 1:
                raise ValueError("QATSeparableConv2D exige canais de entrada conhecidos.")
            self.depthwise_kernel = self.add_weight(
                name="depthwise_kernel",
                shape=(*self.kernel_size, channels, 1),
                initializer=self.depthwise_initializer,
                trainable=True,
            )
            self.pointwise_kernel = self.add_weight(
                name="pointwise_kernel",
                shape=(1, 1, channels, self.filters),
                initializer=self.pointwise_initializer,
                trainable=True,
            )
            self.bias = (
                self.add_weight(name="bias", shape=(self.filters,), initializer="zeros", trainable=True)
                if self.use_bias
                else None
            )
            super().build(input_shape)

        def call(self, inputs: Any) -> Any:
            values = _tf.cast(inputs, _tf.float32)
            depthwise = _tf.nn.depthwise_conv2d(
                values,
                _fake_quantize_weight(self.depthwise_kernel, num_bits=self.weight_bits),
                strides=(1, *self.strides, 1),
                padding=self.padding,
            )
            output = _tf.nn.conv2d(
                depthwise,
                _fake_quantize_weight(self.pointwise_kernel, num_bits=self.weight_bits),
                strides=(1, 1, 1, 1),
                padding="SAME",
            )
            if self.bias is not None:
                output = _tf.nn.bias_add(output, self.bias)
            return output

        def compute_output_shape(self, input_shape: Any) -> Any:
            height, width = input_shape[1], input_shape[2]
            if self.padding == "SAME":
                height = math.ceil(height / self.strides[0]) if height is not None else None
                width = math.ceil(width / self.strides[1]) if width is not None else None
            else:
                height = math.floor((height - self.kernel_size[0]) / self.strides[0]) + 1 if height is not None else None
                width = math.floor((width - self.kernel_size[1]) / self.strides[1]) + 1 if width is not None else None
            return (input_shape[0], height, width, self.filters)

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding.lower(),
                "use_bias": self.use_bias,
                "depthwise_initializer": _tf.keras.initializers.serialize(self.depthwise_initializer),
                "pointwise_initializer": _tf.keras.initializers.serialize(self.pointwise_initializer),
                "weight_bits": self.weight_bits,
            }


    @_tf.keras.utils.register_keras_serializable(package="tcc_benchmark")
    class QATDense(_tf.keras.layers.Layer):
        """Dense-equivalent layer with an explicitly fake-quantized kernel."""

        def __init__(
            self,
            units: int,
            *,
            activation: str | None = None,
            use_bias: bool = True,
            kernel_initializer: Any = "glorot_uniform",
            weight_bits: int = 8,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            if int(weight_bits) not in {4, 8}:
                raise ValueError("QATDense aceita weight_bits 4 ou 8.")
            self.units = int(units)
            self.activation = _tf.keras.activations.get(activation)
            self.activation_name = activation
            self.use_bias = bool(use_bias)
            self.kernel_initializer = _tf.keras.initializers.get(kernel_initializer)
            self.weight_bits = int(weight_bits)

        def build(self, input_shape: Any) -> None:
            features = int(input_shape[-1])
            self.kernel = self.add_weight(
                name="kernel", shape=(features, self.units), initializer=self.kernel_initializer, trainable=True
            )
            self.bias = (
                self.add_weight(name="bias", shape=(self.units,), initializer="zeros", trainable=True)
                if self.use_bias
                else None
            )
            super().build(input_shape)

        def call(self, inputs: Any) -> Any:
            output = _tf.linalg.matmul(_tf.cast(inputs, _tf.float32), _fake_quantize_weight(self.kernel, num_bits=self.weight_bits))
            if self.bias is not None:
                output = _tf.nn.bias_add(output, self.bias)
            return self.activation(output) if self.activation is not None else output

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "units": self.units,
                "activation": _tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": _tf.keras.initializers.serialize(self.kernel_initializer),
                "weight_bits": self.weight_bits,
            }


else:

    class _TensorFlowRequiredLayer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("TensorFlow é necessário para construir camadas QAT.")

    FakeQuantize = _TensorFlowRequiredLayer
    QATSeparableConv2D = _TensorFlowRequiredLayer
    QATDense = _TensorFlowRequiredLayer


def quantized_parameter_size(model: Any, *, weight_bits: int | None) -> dict[str, int | None]:
    """Return physical and deployment-size estimates without making claims about INT4 export."""

    quantizable = 0
    other = 0
    for variable in model.weights:
        name = str(getattr(variable, "name", ""))
        count = int(np.prod(tuple(int(value) for value in variable.shape)))
        if any(token in name for token in ("kernel", "depthwise_kernel", "pointwise_kernel")):
            quantizable += count
        else:
            other += count
    estimated = None
    if weight_bits is not None:
        estimated = math.ceil(quantizable * int(weight_bits) / 8) + other * 4
    return {
        "quantizable_parameter_count": quantizable,
        "non_quantized_parameter_count": other,
        "estimated_deployment_parameter_bytes": estimated,
        "estimated_weight_bits": int(weight_bits) if weight_bits is not None else None,
    }


def _representative_dataset(dataset: Any, *, batches: int = 4) -> Iterator[list[np.ndarray]]:
    """Yield a deterministic calibration subset from the raw prepared training split."""

    yielded = 0
    for batch in dataset:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        values = images.numpy() if hasattr(images, "numpy") else images
        yield [np.asarray(values, dtype=np.float32)]
        yielded += 1
        if yielded >= int(batches):
            return


def convert_litert_model(
    model: Any,
    destination: str | Path,
    *,
    mode: str,
    representative_dataset: Any | None = None,
) -> dict[str, Any]:
    """Convert a Keras model and write conversion facts even when it is unsupported."""

    if _tf is None:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow é necessário para exportar LiteRT.")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "mode": str(mode),
        "path": str(target),
        "status": "failed",
        "requires_flex_delegate": False,
    }

    def build_converter(conversion_model: Any) -> Any:
        converter = _tf.lite.TFLiteConverter.from_keras_model(conversion_model)
        if mode == "fp16":
            converter.optimizations = [_tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [_tf.float16]
        elif mode == "int8":
            if representative_dataset is None:
                raise ValueError("INT8 PTQ/QAT exige dados representativos para calibração.")
            converter.optimizations = [_tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: _representative_dataset(representative_dataset)
            converter.target_spec.supported_ops = [_tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = _tf.int8
            converter.inference_output_type = _tf.int8
        elif mode != "fp32":
            raise ValueError(f"Modo LiteRT desconhecido: {mode}")
        return converter

    def float32_export_clone() -> Any:
        """Keep trained weights while rebuilding mixed-FP16 layers for native LiteRT."""

        def clone_layer(layer: Any) -> Any:
            config = layer.get_config()
            if "dtype" in config:
                config["dtype"] = "float32"
            return layer.__class__.from_config(config)

        cloned_model = _tf.keras.models.clone_model(model, clone_function=clone_layer)
        cloned_model.set_weights(model.get_weights())
        return cloned_model

    try:
        converter = build_converter(model)
        model_bytes = converter.convert()
        atomic_write_bytes(target, model_bytes)
        payload.update(inspect_litert_model(target))
        payload["status"] = "completed"
    except Exception as exc:  # A non-convertible graph is a recorded benchmark result.
        if mode != "fp16" or "ERROR_NEEDS_FLEX_OPS" not in str(exc):
            payload["error"] = repr(exc)
        else:
            try:
                converter = build_converter(float32_export_clone())
                model_bytes = converter.convert()
                atomic_write_bytes(target, model_bytes)
                payload.update(inspect_litert_model(target))
                payload.update(
                    {
                        "status": "completed",
                        "converter_fallback": "float32_clone_for_fp16_export",
                    }
                )
            except Exception as fallback_exc:
                payload.update(
                    {
                        "error": repr(fallback_exc),
                        "standard_conversion_error": repr(exc),
                    }
                )
    atomic_write_json(target.with_suffix(target.suffix + ".metadata.json"), payload)
    return payload


def inspect_litert_model(path: str | Path) -> dict[str, Any]:
    """Inspect tensor dtypes to distinguish native INT8 from a hybrid export."""

    if _tf is None:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow é necessário para inspecionar LiteRT.")
    source = Path(path)
    interpreter = _tf.lite.Interpreter(model_path=str(source))
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    dtype_counts: dict[str, int] = {}
    for detail in tensor_details:
        dtype = np.dtype(detail["dtype"]).name
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    input_types = [np.dtype(detail["dtype"]).name for detail in interpreter.get_input_details()]
    output_types = [np.dtype(detail["dtype"]).name for detail in interpreter.get_output_details()]
    native_int8 = bool(dtype_counts.get("int8")) and not any(
        name.startswith("float") for name in dtype_counts if dtype_counts[name]
    )
    return {
        "bytes": source.stat().st_size,
        "tensor_dtypes": dtype_counts,
        "input_dtypes": input_types,
        "output_dtypes": output_types,
        "quantization_validity": "native_int8" if native_int8 else "float_or_hybrid",
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / exponents.sum(axis=1, keepdims=True)


def _quantize_input(values: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if dtype != np.dtype("int8"):
        return np.asarray(values, dtype=dtype)
    scale, zero_point = detail["quantization"]
    if not scale:
        raise ValueError("Entrada INT8 LiteRT sem escala de quantização.")
    quantized = np.rint(np.asarray(values, dtype=np.float32) / float(scale) + int(zero_point))
    return np.clip(quantized, -128, 127).astype(np.int8)


def _dequantize_output(values: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    dtype = np.dtype(detail["dtype"])
    if dtype != np.dtype("int8"):
        return np.asarray(values, dtype=np.float32)
    scale, zero_point = detail["quantization"]
    if not scale:
        raise ValueError("Saída INT8 LiteRT sem escala de quantização.")
    return (np.asarray(values, dtype=np.float32) - int(zero_point)) * float(scale)


def _metric_payload(labels: np.ndarray, logits: np.ndarray, *, num_classes: int) -> dict[str, Any]:
    from .metrics import classification_metrics

    probabilities = _softmax(logits)
    return {
        "classification": classification_metrics(labels, probabilities, num_classes=num_classes).to_dict(),
        "samples": int(len(labels)),
    }


def benchmark_litert(
    model_path: str | Path,
    test_dataset: Any,
    *,
    num_classes: int,
    warmup_batches: int = 10,
    timed_passes: int = 30,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate LiteRT and time complete test-set passes at the configured batch size."""

    if _tf is None:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow é necessário para executar LiteRT.")
    try:
        import psutil
    except ImportError:  # pragma: no cover - project dependency
        psutil = None
    source = Path(model_path)
    interpreter = _tf.lite.Interpreter(model_path=str(source), num_threads=max(1, (os.cpu_count() or 2) // 2))
    process = psutil.Process() if psutil is not None else None
    rss_before = int(process.memory_info().rss) if process is not None else None
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for value in test_dataset:
        images, labels = value[0], value[1]
        batches.append(
            (
                np.asarray(images.numpy() if hasattr(images, "numpy") else images, dtype=np.float32),
                np.asarray(labels.numpy() if hasattr(labels, "numpy") else labels, dtype=np.int64).reshape(-1),
            )
        )
    if not batches:
        raise ValueError("O dataset de teste LiteRT não contém batches.")
    timed_batches = [batch for batch in batches if len(batch[0]) == len(batches[0][0])]
    if not timed_batches:
        raise ValueError("Não há batch completo para cronometrar a inferência LiteRT.")

    def invoke(images: np.ndarray) -> np.ndarray:
        expected_shape = tuple(int(item) for item in input_detail["shape"])
        actual_shape = tuple(int(item) for item in images.shape)
        if expected_shape != actual_shape:
            interpreter.resize_tensor_input(input_detail["index"], actual_shape, strict=False)
            interpreter.allocate_tensors()
        interpreter.set_tensor(input_detail["index"], _quantize_input(images, input_detail))
        interpreter.invoke()
        return _dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)

    for index in range(int(warmup_batches)):
        invoke(timed_batches[index % len(timed_batches)][0])

    outputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in batches:
        outputs.append(invoke(images))
        labels.append(targets)
    all_logits = np.concatenate(outputs, axis=0)
    all_labels = np.concatenate(labels, axis=0)

    pass_seconds: list[float] = []
    peak_rss = int(process.memory_info().rss) if process is not None else None
    for _ in range(int(timed_passes)):
        started = time.perf_counter()
        for images, _targets in timed_batches:
            invoke(images)
        pass_seconds.append(time.perf_counter() - started)
        if process is not None:
            peak_rss = max(int(peak_rss or 0), int(process.memory_info().rss))
    total_samples = sum(len(targets) for _images, targets in timed_batches) * int(timed_passes)
    total_seconds = float(sum(pass_seconds))
    payload = {
        "runtime": "LiteRT Python CPU",
        "batch_size": int(timed_batches[0][0].shape[0]),
        "warmup_batches": int(warmup_batches),
        "timed_passes": int(timed_passes),
        "timed_test_samples": int(total_samples),
        "pass_seconds": pass_seconds,
        "median_pass_seconds": float(np.median(pass_seconds)),
        "p95_pass_seconds": float(np.quantile(pass_seconds, 0.95)),
        "throughput_examples_per_second": float(total_samples / total_seconds) if total_seconds else None,
        "timed_full_batches_per_pass": len(timed_batches),
        "median_batch_latency_ms": float(np.median(pass_seconds) * 1000.0 / len(timed_batches)),
        "process_rss_before_bytes": rss_before,
        "process_rss_peak_bytes": peak_rss,
        "process_rss_delta_bytes": (int(peak_rss) - int(rss_before)) if peak_rss is not None and rss_before is not None else None,
        "inference_vram": "not_applicable_cpu_runtime",
    }
    payload.update(_metric_payload(all_labels, all_logits, num_classes=num_classes))
    return payload, all_labels, all_logits


def write_litert_results(
    destination: str | Path,
    payload: Mapping[str, Any],
    labels: np.ndarray,
    logits: np.ndarray,
) -> None:
    target = Path(destination)
    atomic_write_json(target / "litert_benchmark.json", dict(payload))
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(logits, dtype=np.float32))
    atomic_write_bytes(target / "litert_logits.npy", buffer.getvalue())
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(labels, dtype=np.int64))
    atomic_write_bytes(target / "litert_labels.npy", buffer.getvalue())


def logits_mse(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    if reference.shape != candidate.shape:
        return None
    return float(np.mean(np.square(np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64))))


def build_comparison_report(output_root: str | Path) -> dict[str, Path]:
    """Build a compact self-contained comparison from per-variant artefacts."""

    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    fp32_logits: np.ndarray | None = None
    fp32_litert_logits: np.ndarray | None = None
    for variant in VARIANTS:
        run_root = root / variant.key / "kmnist" / "runs" / "kmnist__unit_interval__all_raw__seed-42"
        artifacts = run_root / "artifacts"
        status_path = run_root / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        test_path = artifacts / "test_metrics.json"
        test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
        inference_path = artifacts / "litert_benchmark.json"
        inference = json.loads(inference_path.read_text(encoding="utf-8")) if inference_path.exists() else {}
        size_path = artifacts / "model_size.json"
        size = json.loads(size_path.read_text(encoding="utf-8")) if size_path.exists() else {}
        keras_logits_path = artifacts / "logits.npy"
        litert_logits_path = artifacts / "litert_logits.npy"
        keras_logits = np.load(keras_logits_path) if keras_logits_path.exists() else None
        litert_logits = np.load(litert_logits_path) if litert_logits_path.exists() else None
        if variant.key == "fp32":
            fp32_logits = keras_logits
            fp32_litert_logits = litert_logits
        classification = inference.get("classification") or test.get("classification", {})
        row = {
            "variant": variant.key,
            "label": variant.label,
            "status": status.get("status", "missing"),
            "metric_source": "litert" if inference.get("classification") else "keras_qat_or_training",
            "accuracy": classification.get("accuracy"),
            "macro_precision": classification.get("macro_precision"),
            "macro_recall": classification.get("macro_recall"),
            "macro_f1": classification.get("macro_f1"),
            "macro_ovr_auc": classification.get("macro_ovr_auc"),
            "serialized_model_bytes": size.get("serialized_model_bytes"),
            "estimated_deployment_parameter_bytes": size.get("estimated_deployment_parameter_bytes"),
            "inference_throughput_examples_per_second": inference.get("throughput_examples_per_second"),
            "inference_median_batch_latency_ms": inference.get("median_batch_latency_ms"),
            "inference_ram_delta_bytes": inference.get("process_rss_delta_bytes"),
            "inference_vram": inference.get("inference_vram", "not_available"),
            "quantization_validity": size.get("quantization_validity", "not_available"),
            "efficiency_interpretation": "emulated_int4" if variant.emulated else "measured",
            "logits_mse_vs_fp32": None,
        }
        if variant.key == "int8_ptq" and fp32_litert_logits is not None and litert_logits is not None:
            row["logits_mse_vs_fp32"] = logits_mse(fp32_litert_logits, litert_logits)
        elif fp32_logits is not None and keras_logits is not None:
            row["logits_mse_vs_fp32"] = logits_mse(fp32_logits, keras_logits)
        rows.append(row)

    baseline = next((row for row in rows if row["variant"] == "fp32"), {})
    for row in rows:
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_ovr_auc"):
            value, reference = row.get(key), baseline.get(key)
            row[f"delta_{key}_vs_fp32"] = float(value) - float(reference) if value is not None and reference is not None else None

    reports = root / "comparison"
    reports.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["variant"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(reports / "quantization_comparison.csv", stream.getvalue())
    atomic_write_json(reports / "quantization_comparison.json", {"rows": rows, "baseline": "fp32"})
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in fieldnames)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape('' if row.get(name) is None else str(row.get(name)))}</td>" for name in fieldnames) + "</tr>"
        for row in rows
    )
    atomic_write_text(
        reports / "quantization_comparison.html",
        "<!doctype html><meta charset='utf-8'><title>KMNIST quantization benchmark</title>"
        "<h1>KMNIST quantization benchmark</h1><p>INT4 QAT is numerical emulation; its physical deployment metrics are estimates.</p>"
        f"<table border='1'><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>",
    )
    return {
        "csv": reports / "quantization_comparison.csv",
        "json": reports / "quantization_comparison.json",
        "html": reports / "quantization_comparison.html",
    }


__all__ = [
    "FakeQuantize",
    "QATDense",
    "QATSeparableConv2D",
    "QuantizationVariant",
    "TRAINING_VARIANTS",
    "VARIANTS",
    "benchmark_litert",
    "build_comparison_report",
    "convert_litert_model",
    "logits_mse",
    "quantized_parameter_size",
    "variant_for",
    "write_litert_results",
]
