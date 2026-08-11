from __future__ import annotations

import numpy as np

from tcc_benchmark.config import ConfigurationError, TrainingSettings
from tcc_benchmark.metrics import classification_metrics
from tcc_benchmark.quantization import TRAINING_VARIANTS, VARIANTS, logits_mse


def test_fixed_quantization_variant_matrix_has_four_trainings_and_one_ptq() -> None:
    assert [variant.key for variant in VARIANTS] == ["fp32", "fp16", "int8_qat", "int4_qat", "int8_ptq"]
    assert [variant.key for variant in TRAINING_VARIANTS] == ["fp32", "fp16", "int8_qat", "int4_qat"]
    assert VARIANTS[-1].post_training is True
    assert VARIANTS[-2].emulated is True


def test_training_settings_accept_float32_and_reject_unsupported_qat_bits() -> None:
    TrainingSettings(dtype_policy="float32", qat_weight_bits=4).validate()
    TrainingSettings(dtype_policy="mixed_float16", qat_weight_bits=8).validate()
    try:
        TrainingSettings(qat_weight_bits=6).validate()
    except ConfigurationError as exc:
        assert "qat_weight_bits" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("6-bit QAT não pode ser aceito pelo protocolo fixo.")


def test_classification_metrics_include_macro_precision_recall_and_auc() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.2, 0.7],
            [0.05, 0.05, 0.9],
        ]
    )
    metrics = classification_metrics(truth, probabilities, num_classes=3)
    assert metrics.accuracy == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.macro_ovr_auc == 1.0


def test_logits_mse_requires_equal_shape_and_has_zero_for_identical_outputs() -> None:
    baseline = np.array([[1.0, -1.0], [2.0, 0.0]], dtype=np.float32)
    assert logits_mse(baseline, baseline.copy()) == 0.0
    assert logits_mse(baseline, np.zeros((1, 2), dtype=np.float32)) is None
    assert logits_mse(baseline, np.zeros_like(baseline)) == 1.5
