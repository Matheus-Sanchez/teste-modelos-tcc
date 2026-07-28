import numpy as np

from tcc_benchmark.data import (
    balance_training_data,
    class_counts,
    normalize_image,
    stratified_split_indices,
    unit_interval_stats,
)
from tcc_benchmark.metrics import classification_metrics


def test_stratified_split_is_deterministic_disjoint_and_complete() -> None:
    labels = np.repeat(np.arange(3), 10)
    first = stratified_split_indices(labels, seed=42)
    second = stratified_split_indices(labels, seed=42)

    assert np.array_equal(first.train, second.train)
    assert first.fingerprint() == second.fingerprint()
    first.validate(total_size=len(labels))
    assert class_counts(labels[first.train]) == {0: 7, 1: 7, 2: 7}
    assert class_counts(labels[first.validation]) == {0: 2, 1: 2, 2: 2}
    assert class_counts(labels[first.test]) == {0: 1, 1: 1, 2: 1}


def test_all_four_balance_modes_affect_only_the_training_collection() -> None:
    samples = np.arange(12)
    labels = np.array([0] * 6 + [1] * 4 + [2] * 2)

    raw = balance_training_data(samples, labels, mode="all_raw", seed=9)
    under = balance_training_data(samples, labels, mode="undersample", seed=9)
    over = balance_training_data(samples, labels, mode="oversample", seed=9)
    weighted = balance_training_data(samples, labels, mode="class_weight", seed=9)

    assert raw.output_class_counts == {0: 6, 1: 4, 2: 2}
    assert under.output_class_counts == {0: 2, 1: 2, 2: 2}
    assert over.output_class_counts == {0: 6, 1: 6, 2: 6}
    assert weighted.labels.tolist() == labels.tolist()
    assert weighted.class_weights == {0: 12 / 18, 1: 1.0, 2: 2.0}


def test_unit_interval_normalization_is_identity_for_numpy() -> None:
    values = np.array([[[0.25], [0.75]]], dtype=np.float32)
    stats = unit_interval_stats(channels=1, image_size=64)
    assert np.array_equal(normalize_image(values, stats), values)


def test_classification_metrics_include_missing_prediction_class() -> None:
    metrics = classification_metrics(
        [0, 1, 2, 2],
        [0, 0, 2, 0],
        num_classes=3,
    )
    assert metrics.confusion_matrix.tolist() == [[1, 0, 0], [1, 0, 0], [1, 0, 1]]
    assert metrics.per_class[1].precision == 0.0
    assert metrics.per_class[1].recall == 0.0
    assert metrics.macro_f1 < metrics.accuracy


def test_tensorflow_model_and_pipeline_when_profile_is_installed(tmp_path) -> None:
    """Exercise the TensorFlow-only public surface without breaking audit CI."""

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        return

    from tcc_benchmark.data import AugmentationConfig, build_tf_dataset, compute_normalization_stats
    from tcc_benchmark.metrics import make_training_callbacks
    from tcc_benchmark.model import build_and_compile_legacy_cnn

    images = np.zeros((12, 13, 17, 1), dtype=np.uint8)
    labels = np.repeat(np.arange(3), 4)
    stats = compute_normalization_stats(images, image_size=64, channels=1, mode="zscore", batch_size=4)
    assert stats.mean == (0.0,)
    assert stats.std[0] > 0
    dataset, info = build_tf_dataset(
        images,
        labels,
        image_size=64,
        channels=1,
        batch_size=4,
        normalization=stats,
        training=True,
        augmentation=AugmentationConfig(
            flip_lr=False,
            brightness_delta=0.0,
            contrast_lower=1.0,
            contrast_upper=1.0,
            translate_frac=0.0,
            zoom_min=1.0,
            zoom_max=1.0,
            noise_std=0.0,
            cutout_prob=0.0,
            cutout_max_frac=0.0,
        ),
        extra_fraction=0.5,
        seed=42,
    )
    assert info.total_examples == 18
    batch_images, batch_labels = next(iter(dataset))
    assert batch_images.shape[1:] == (64, 64, 1)
    assert batch_images.dtype.name == "float16"
    assert batch_labels.dtype.name == "int32"

    model = build_and_compile_legacy_cnn(image_size=64, channels=1, num_classes=3, seed=42)
    assert model.input_shape == (None, 64, 64, 1)
    assert model.output_shape == (None, 3)
    assert tf.keras.mixed_precision.global_policy().name == "mixed_float16"
    assert model.get_layer("block1_sepconv5").compute_dtype == "float16"
    assert model.get_layer("predictions").compute_dtype == "float32"
    validation = dataset.take(info.batches)
    history = model.fit(
        dataset,
        validation_data=validation,
        steps_per_epoch=info.batches,
        validation_steps=3,
        epochs=2,
        verbose=0,
    )
    assert history.epoch == [0, 1]
    rgb_model = build_and_compile_legacy_cnn(image_size=128, channels=3, num_classes=5, seed=43)
    assert rgb_model.input_shape == (None, 128, 128, 3)
    assert rgb_model.output_shape == (None, 5)
    callbacks = make_training_callbacks(validation_data=dataset, num_classes=3, run_dir=tmp_path)
    names = {type(callback).__name__ for callback in callbacks}
    assert {"EpochMetricsCallback", "ModelCheckpoint", "EarlyStopping", "ReduceLROnPlateau", "BackupAndRestore"} <= names


def test_backup_and_restore_resumes_after_simulated_interrupt_when_tensorflow_is_installed(tmp_path) -> None:
    """The second fit must begin after the epoch checkpoint saved before interruption."""

    try:
        import tensorflow as tf
    except ImportError:
        return

    from tcc_benchmark.metrics import make_training_callbacks
    from tcc_benchmark.model import build_and_compile_legacy_cnn

    images = np.zeros((12, 64, 64, 1), dtype=np.float32)
    labels = np.repeat(np.arange(3, dtype=np.int32), 4)
    train = tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)
    validation = tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)

    class InterruptAfterFirstEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):  # type: ignore[no-untyped-def]
            if epoch == 0:
                raise KeyboardInterrupt

    first = build_and_compile_legacy_cnn(image_size=64, channels=1, num_classes=3, seed=42)
    try:
        first.fit(
            train,
            validation_data=validation,
            epochs=3,
            verbose=0,
            callbacks=[
                *make_training_callbacks(validation_data=validation, num_classes=3, run_dir=tmp_path),
                InterruptAfterFirstEpoch(),
            ],
        )
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - only reached if Keras fails to propagate interruption
        raise AssertionError("A interrupção simulada deveria parar o primeiro fit.")

    assert (tmp_path / "backup").exists()
    assert (tmp_path / "last.keras").exists()
    restored = build_and_compile_legacy_cnn(image_size=64, channels=1, num_classes=3, seed=42)
    history = restored.fit(
        train,
        validation_data=validation,
        epochs=3,
        verbose=0,
        callbacks=make_training_callbacks(validation_data=validation, num_classes=3, run_dir=tmp_path),
    )
    assert history.epoch
    assert min(history.epoch) >= 1
    tf.keras.backend.clear_session()
