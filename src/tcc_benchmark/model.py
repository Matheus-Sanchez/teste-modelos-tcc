"""Scratch-only reconstruction of the legacy separable-convolution CNN."""

from __future__ import annotations

from typing import Any

from .config import DEFAULT_TRAINING_SETTINGS
from .data import require_tensorflow


LEGACY_FILTERS: tuple[int, ...] = (32, 48, 64, 96, 128)
TRAINING_DTYPE_POLICY = DEFAULT_TRAINING_SETTINGS.dtype_policy


def set_dtype_policy(dtype_policy: str) -> None:
    """Enable float16 compute while retaining float32 variables and outputs.

    Keras' ``mixed_float16`` policy is the safe GPU-memory-oriented form of
    float16 training: convolution compute and activations use float16, model
    variables remain float32, and the final classification layer is explicitly
    float32 for a stable softmax/loss path.
    """

    tensorflow = require_tensorflow()
    current = tensorflow.keras.mixed_precision.global_policy()
    if current.name != str(dtype_policy):
        tensorflow.keras.mixed_precision.set_global_policy(str(dtype_policy))


def set_mixed_float16_policy() -> None:
    """Compatibility alias for the central default training policy."""

    set_dtype_policy(TRAINING_DTYPE_POLICY)


def set_training_seed(seed: int, *, enable_op_determinism: bool = True) -> None:
    """Set TensorFlow/Keras random state for a reproducible matrix cell."""

    tensorflow = require_tensorflow()
    tensorflow.keras.utils.set_random_seed(int(seed))
    if enable_op_determinism:
        try:
            tensorflow.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            # Older TF builds and already-initialized runtimes can reject this;
            # stateless data augmentation still stays reproducible.
            pass


def build_legacy_cnn(
    *,
    image_size: int,
    channels: int,
    num_classes: int,
    name: str = "legacy_scratch_cnn",
    dtype_policy: str = DEFAULT_TRAINING_SETTINGS.dtype_policy,
) -> Any:
    """Build the old CNN topology from scratch with dynamic I/O dimensions.

    The architecture is intentionally unchanged from the reusable legacy CNN:
    five ``SeparableConv -> GroupNorm -> swish`` pairs, max pooling per block,
    concatenated GAP/GMP and a three-dropout dense head.  It carries no
    pretrained layers or weights.  Sizes of 64 and 128 are the benchmark's
    intended inputs; any value >=64 is safe with the five pooling stages.
    """

    tensorflow = require_tensorflow()
    image_size = int(image_size)
    channels = int(channels)
    num_classes = int(num_classes)
    if image_size < 64:
        raise ValueError("A CNN legada exige image_size >= 64 por causa dos cinco blocos de pooling.")
    if channels not in {1, 3}:
        raise ValueError("channels deve ser 1 (cinza) ou 3 (RGB).")
    if num_classes < 2:
        raise ValueError("num_classes deve ser pelo menos 2 para classificação.")
    set_dtype_policy(dtype_policy)
    layers = tensorflow.keras.layers
    if not hasattr(layers, "GroupNormalization"):
        raise RuntimeError(
            "Esta versão do TensorFlow/Keras não oferece GroupNormalization; "
            "use o perfil TensorFlow 2.21 do projeto."
        )

    inputs = layers.Input(shape=(image_size, image_size, channels), dtype="float32", name="image")
    x = inputs
    for index, filters in enumerate(LEGACY_FILTERS, start=1):
        x = layers.SeparableConv2D(
            filters,
            5,
            strides=2 if index == 1 else 1,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            name=f"block{index}_sepconv5",
        )(x)
        x = layers.GroupNormalization(groups=16, axis=-1, name=f"block{index}_gn1")(x)
        x = layers.Activation("swish", name=f"block{index}_swish1")(x)
        x = layers.SeparableConv2D(
            filters,
            3,
            strides=1,
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            name=f"block{index}_sepconv3",
        )(x)
        x = layers.GroupNormalization(groups=16, axis=-1, name=f"block{index}_gn2")(x)
        x = layers.Activation("swish", name=f"block{index}_swish2")(x)
        x = layers.MaxPooling2D(2, name=f"block{index}_pool")(x)

    gap = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    gmp = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = layers.Concatenate(name="global_pool_concat")((gap, gmp))
    x = layers.Dropout(0.4, name="dropout1")(x)
    x = layers.Dense(256, activation="silu", name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout2")(x)
    x = layers.Dense(256, activation="silu", name="dense2")(x)
    x = layers.Dropout(0.4, name="dropout3")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return tensorflow.keras.Model(inputs=inputs, outputs=outputs, name=name)


def compile_legacy_cnn(
    model: Any,
    *,
    learning_rate: float = DEFAULT_TRAINING_SETTINGS.learning_rate,
    dtype_policy: str = DEFAULT_TRAINING_SETTINGS.dtype_policy,
) -> Any:
    """Compile with the fixed benchmark optimizer and sparse classification loss."""

    tensorflow = require_tensorflow()
    if float(learning_rate) <= 0:
        raise ValueError("learning_rate deve ser positivo.")
    set_dtype_policy(dtype_policy)
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tensorflow.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )
    return model


def build_and_compile_legacy_cnn(
    *,
    image_size: int,
    channels: int,
    num_classes: int,
    learning_rate: float = DEFAULT_TRAINING_SETTINGS.learning_rate,
    dtype_policy: str = DEFAULT_TRAINING_SETTINGS.dtype_policy,
    seed: int | None = None,
) -> Any:
    """Convenience constructor used by the runner for one fully fresh run."""

    if seed is not None:
        set_training_seed(int(seed))
    model = build_legacy_cnn(
        image_size=image_size,
        channels=channels,
        num_classes=num_classes,
        dtype_policy=dtype_policy,
    )
    return compile_legacy_cnn(model, learning_rate=learning_rate, dtype_policy=dtype_policy)


# Familiar aliases make the public intent obvious in small experiment scripts.
build_model = build_legacy_cnn
compile_model = compile_legacy_cnn
