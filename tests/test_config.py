from __future__ import annotations

from pathlib import Path

from tcc_benchmark.config import (
    BALANCE_MODES,
    DATASET_ORDER,
    NORMALIZATION_MODES,
    SuiteSettings,
    fingerprint,
    load_dataset_registry,
    load_suite_settings,
    run_key,
)


def test_suite_defaults_describe_400_jobs() -> None:
    settings = SuiteSettings()
    assert len(DATASET_ORDER) == 10
    assert len(settings.seeds) * len(NORMALIZATION_MODES) * len(BALANCE_MODES) == 40
    settings.validate()


def test_dataset_seed_overrides_are_loaded(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text("suite:\n  dataset_seeds:\n    mnist: [42, 43]\n    fashion_mnist: [42]\n", encoding="utf-8")
    settings = load_suite_settings(suite)
    assert settings.dataset_seeds == {"mnist": (42, 43), "fashion_mnist": (42,)}


def test_training_block_is_the_single_runtime_source_of_truth(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """suite:
  training:
    max_epochs: 100
    batch_size: 64
    learning_rate: 0.0003
    extra_fraction: 2.0
    dtype_policy: mixed_float16
    keras_verbose: 1
    default_image_size: 64
    image_size_overrides: {gtsrb: 128}
    augmentation: {flip_lr: true}
    shuffle_buffer_max_mib: 1024
    early_stopping_patience: 10
    reduce_lr_factor: 0.3
    reduce_lr_patience: 4
    reduce_lr_min_lr: 0.0000001
""",
        encoding="utf-8",
    )
    settings = load_suite_settings(suite)
    assert settings.training.batch_size == 64
    assert settings.training.image_size_for("mnist") == 64
    assert settings.training.image_size_for("gtsrb") == 128
    assert settings.training.augmentation["flip_lr"] is True
    assert settings.training.shuffle_buffer_max_mib == 1024


def test_fingerprint_and_run_key_are_stable() -> None:
    payload = {"b": 2, "a": 1}
    assert fingerprint(payload) == fingerprint({"a": 1, "b": 2})
    assert run_key("mnist", "unit_interval", "all_raw", 42) == "mnist__unit_interval__all_raw__seed-42"


def test_default_registry_resolves_project_dataset_subfolders() -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry = load_dataset_registry(project_root / "configs" / "datasets.yaml")

    configured_datasets = tuple(name for name in DATASET_ORDER if name != "cinic10")
    assert tuple(registry) == configured_datasets
    for name in configured_datasets:
        assert registry[name].root == project_root / "datasets" / name


def test_full_profile_has_one_explicit_training_block() -> None:
    project_root = Path(__file__).resolve().parents[1]
    settings = load_suite_settings(project_root / "configs" / "full-100-epochs.yaml")

    assert settings.output_root == project_root / "outputs" / "full-100-epochs-batch64"
    assert settings.training.batch_size == 512
    assert settings.training.dtype_policy == "mixed_float16"
    assert settings.training.image_size_for("gtsrb") == 128


def test_wsl_profile_caps_total_shuffle_memory() -> None:
    project_root = Path(__file__).resolve().parents[1]
    settings = load_suite_settings(project_root / "configs" / "full-100-epochs-wsl-optimized.yaml")

    assert settings.training.preprocess_cache_max_mib == 2048
    assert settings.training.shuffle_buffer_max_mib == 1024
