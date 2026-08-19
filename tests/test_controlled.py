from tcc_benchmark.controlled import select_activation, select_batch, select_quantization


def test_batch_selection_prioritizes_macro_f1_then_epoch_time() -> None:
    winner = select_batch(
        [
            {"status": "completed", "batch_size": 32, "macro_f1": 0.80, "mean_epoch_seconds": 10},
            {"status": "completed", "batch_size": 64, "macro_f1": 0.81, "mean_epoch_seconds": 30},
            {"status": "completed", "batch_size": 128, "macro_f1": 0.81, "mean_epoch_seconds": 20},
        ]
    )
    assert winner["batch_size"] == 128


def test_quantization_selection_requires_fp32_tolerance_then_latency() -> None:
    winner = select_quantization(
        [
            {"status": "completed", "variant": "fp32", "macro_f1": 0.90, "median_batch_latency_ms": 9, "serialized_model_bytes": 100},
            {"status": "completed", "variant": "fp16", "macro_f1": 0.895, "median_batch_latency_ms": 6, "serialized_model_bytes": 70},
            {"status": "completed", "variant": "int8_ptq", "macro_f1": 0.88, "median_batch_latency_ms": 2, "serialized_model_bytes": 30},
        ]
    )
    assert winner["variant"] == "fp16"
    assert winner["macro_f1_threshold"] == 0.89


def test_activation_selection_uses_epoch_time_only_after_metric_tie() -> None:
    winner = select_activation(
        [
            {"status": "completed", "hidden_activation": "relu", "macro_f1": 0.80, "mean_epoch_seconds": 3},
            {"status": "completed", "hidden_activation": "sigmoid", "macro_f1": 0.82, "mean_epoch_seconds": 100},
        ]
    )
    assert winner["hidden_activation"] == "sigmoid"
