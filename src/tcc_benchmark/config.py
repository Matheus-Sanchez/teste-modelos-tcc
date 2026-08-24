"""Configuração estável da suíte e carregamento do registro de dados local."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - erro explicado no uso
    yaml = None


DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
NORMALIZATION_MODES: tuple[str, ...] = ("unit_interval", "zscore")
BALANCE_MODES: tuple[str, ...] = (
    "all_raw",
    "undersample",
    "oversample",
    "class_weight",
)
DATASET_ORDER: tuple[str, ...] = (
    "mnist",
    "fashion_mnist",
    "kmnist",
    "emnist_balanced",
    "cifar10",
    "cifar100_coarse",
    "svhn",
    "gtsrb",
    "fer2013",
)
DEFAULT_AUGMENTATION: dict[str, float | bool] = {
    "flip_lr": True,
    "brightness_delta": 0.03,
    "contrast_lower": 0.92,
    "contrast_upper": 1.08,
    "translate_frac": 0.05,
    "zoom_min": 0.96,
    "zoom_max": 1.04,
    "noise_std": 0.005,
    "cutout_prob": 0.25,
    "cutout_max_frac": 0.08,
}
HIDDEN_ACTIVATIONS: tuple[str, ...] = ("swish", "relu", "sigmoid", "softmax")


class ConfigurationError(ValueError):
    """Raised when a suite or local-dataset configuration is invalid."""


@dataclasses.dataclass(frozen=True)
class DatasetEntry:
    name: str
    adapter: str
    root: Path
    options: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def target_size(self) -> int:
        """Legacy registry value; run resolution is defined by TrainingSettings."""

        return int(self.options.get("target_size", 64))


@dataclasses.dataclass(frozen=True)
class TrainingSettings:
    """The single source of truth for all training-time parameters.

    Every model build, data pipeline, callback and run manifest receives this
    object.  Dataset identity and the experiment matrix intentionally remain in
    :class:`SuiteSettings`; parameters that influence optimization, memory use
    or the training protocol live only here.
    """

    max_epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 3e-4
    extra_fraction: float = 2.0
    dtype_policy: str = "mixed_float16"
    # This controls every non-linearity in the feature extractor and dense
    # head.  The classifier always emits logits so loss/evaluation stay
    # comparable across activation experiments.
    hidden_activation: str = "swish"
    keras_verbose: int = 1
    default_image_size: int = 64
    image_size_overrides: dict[str, int] = dataclasses.field(default_factory=lambda: {"gtsrb": 128})
    augmentation: dict[str, float | bool] = dataclasses.field(default_factory=lambda: dict(DEFAULT_AUGMENTATION))
    preprocess_cache_max_mib: int = 0
    shuffle_buffer_max_mib: int = 0
    early_stopping_patience: int = 10
    early_stopping_enabled: bool = True
    reduce_lr_factor: float = 0.3
    reduce_lr_patience: int = 4
    reduce_lr_min_lr: float = 1e-7
    reduce_lr_enabled: bool = True
    # QAT keeps float32 master variables and fake-quantizes kernels during the
    # forward pass. ``None`` is the ordinary floating-point architecture.
    qat_weight_bits: int | None = None

    def validate(self) -> None:
        if self.max_epochs < 1 or self.batch_size < 1:
            raise ConfigurationError("training.max_epochs e training.batch_size devem ser positivos")
        if self.learning_rate <= 0 or self.extra_fraction < 0:
            raise ConfigurationError("training.learning_rate deve ser positivo e extra_fraction não pode ser negativo")
        if self.dtype_policy not in {"float32", "mixed_float16"}:
            raise ConfigurationError("training.dtype_policy deve ser 'float32' ou 'mixed_float16'.")
        if self.hidden_activation not in HIDDEN_ACTIVATIONS:
            raise ConfigurationError(
                "training.hidden_activation deve ser um de "
                f"{', '.join(HIDDEN_ACTIVATIONS)}."
            )
        if self.keras_verbose not in {0, 1, 2}:
            raise ConfigurationError("training.keras_verbose deve ser 0, 1 ou 2")
        if self.default_image_size < 64 or any(int(size) < 64 for size in self.image_size_overrides.values()):
            raise ConfigurationError("As resoluções de treino devem ser pelo menos 64.")
        if self.early_stopping_patience < 1 or self.reduce_lr_patience < 1:
            raise ConfigurationError("As paciências dos callbacks devem ser positivas.")
        if not isinstance(self.early_stopping_enabled, bool) or not isinstance(self.reduce_lr_enabled, bool):
            raise ConfigurationError("As flags de callbacks devem ser booleanas.")
        if self.preprocess_cache_max_mib < 0:
            raise ConfigurationError("training.preprocess_cache_max_mib não pode ser negativo.")
        if self.shuffle_buffer_max_mib < 0:
            raise ConfigurationError("training.shuffle_buffer_max_mib não pode ser negativo.")
        if not 0 < self.reduce_lr_factor < 1 or self.reduce_lr_min_lr <= 0:
            raise ConfigurationError("Os parâmetros de ReduceLROnPlateau são inválidos.")
        if self.qat_weight_bits is not None and int(self.qat_weight_bits) not in {4, 8}:
            raise ConfigurationError("training.qat_weight_bits deve ser 4, 8 ou nulo.")
        unknown_augmentation = set(self.augmentation) - set(DEFAULT_AUGMENTATION)
        if unknown_augmentation:
            raise ConfigurationError(f"Parâmetros de augmentação desconhecidos: {unknown_augmentation}")

    def image_size_for(self, dataset: str) -> int:
        return int(self.image_size_overrides.get(str(dataset), self.default_image_size))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


DEFAULT_TRAINING_SETTINGS = TrainingSettings()


@dataclasses.dataclass(frozen=True)
class SuiteSettings:
    output_root: Path = Path("artifacts")
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    dataset_seeds: dict[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    normalizations: tuple[str, ...] = NORMALIZATION_MODES
    balance_modes: tuple[str, ...] = BALANCE_MODES
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    training: TrainingSettings = dataclasses.field(default_factory=TrainingSettings)
    hardware_interval_seconds: float = 5.0

    def validate(self) -> None:
        if not abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) < 1e-9:
            raise ConfigurationError("train_fraction + validation_fraction + test_fraction deve ser 1.0")
        self.training.validate()
        unknown_normalization = set(self.normalizations) - set(NORMALIZATION_MODES)
        unknown_balance = set(self.balance_modes) - set(BALANCE_MODES)
        if unknown_normalization or unknown_balance:
            raise ConfigurationError(f"Opções desconhecidas: {unknown_normalization | unknown_balance}")
        for dataset, dataset_seeds in self.dataset_seeds.items():
            if not str(dataset):
                raise ConfigurationError("dataset_seeds não pode conter um nome de dataset vazio")
            if dataset not in DATASET_ORDER:
                raise ConfigurationError(f"dataset_seeds contém dataset fora da matriz ativa: '{dataset}'")
            if not dataset_seeds or any(int(seed) < 0 for seed in dataset_seeds):
                raise ConfigurationError(f"dataset_seeds inválido para '{dataset}'")


def _expand_path(value: str | Path, base: Path) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(text)
    return path if path.is_absolute() else (base / path).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML não está instalado. Instale um dos perfis em requirements/.")
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f"O YAML deve conter um mapa no topo: {source}")
    return payload


def load_dataset_registry(path: str | Path) -> dict[str, DatasetEntry]:
    source = Path(path).expanduser().resolve()
    payload = load_yaml(source)
    raw_datasets = payload.get("datasets", payload)
    if not isinstance(raw_datasets, Mapping):
        raise ConfigurationError("O registro de datasets precisa conter o mapa 'datasets'.")

    entries: dict[str, DatasetEntry] = {}
    for name, raw in raw_datasets.items():
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"Entrada do dataset '{name}' precisa ser um mapa.")
        adapter = str(raw.get("adapter", name))
        root_raw = raw.get("root")
        if not root_raw:
            raise ConfigurationError(f"Dataset '{name}' não possui 'root'.")
        options = {key: value for key, value in raw.items() if key not in {"adapter", "root"}}
        entries[str(name)] = DatasetEntry(
            name=str(name), adapter=adapter, root=_expand_path(root_raw, source.parent), options=options
        )
    return entries


def load_suite_settings(path: str | Path | None = None) -> SuiteSettings:
    if path is None:
        settings = SuiteSettings()
        settings.validate()
        return settings
    source = Path(path).expanduser().resolve()
    payload = load_yaml(source)
    raw = payload.get("suite", payload)
    kwargs: dict[str, Any] = {}
    aliases = {"validation_fraction": "validation_fraction", "val_fraction": "validation_fraction"}
    for field in dataclasses.fields(SuiteSettings):
        if field.name == "training":
            continue
        key = aliases.get(field.name, field.name)
        if key in raw:
            value = raw[key]
            if field.name == "output_root":
                value = _expand_path(value, source.parent)
            elif field.name in {"seeds", "normalizations", "balance_modes"}:
                value = tuple(value)
            elif field.name == "dataset_seeds":
                if not isinstance(value, Mapping):
                    raise ConfigurationError("dataset_seeds precisa ser um mapa dataset -> lista de seeds")
                value = {str(dataset): tuple(int(seed) for seed in seeds) for dataset, seeds in value.items()}
            kwargs[field.name] = value
    raw_training = raw.get("training", {})
    if not isinstance(raw_training, Mapping):
        raise ConfigurationError("training precisa ser um mapa.")
    # Accept flat legacy suite files while the checked-in profiles use the
    # explicit training block.  The resulting object is still the one source
    # of truth used by the runtime.
    training_values = dict(raw_training)
    for field in dataclasses.fields(TrainingSettings):
        if field.name in raw and field.name not in training_values:
            training_values[field.name] = raw[field.name]
    if "image_size_overrides" in training_values:
        value = training_values["image_size_overrides"]
        if not isinstance(value, Mapping):
            raise ConfigurationError("training.image_size_overrides precisa ser um mapa.")
        training_values["image_size_overrides"] = {str(name): int(size) for name, size in value.items()}
    if "augmentation" in training_values:
        value = training_values["augmentation"]
        if not isinstance(value, Mapping):
            raise ConfigurationError("training.augmentation precisa ser um mapa.")
        training_values["augmentation"] = {str(name): value for name, value in value.items()}
    kwargs["training"] = TrainingSettings(**training_values)
    settings = SuiteSettings(**kwargs)
    settings.validate()
    return settings


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def run_key(dataset: str, normalization: str, balance_mode: str, seed: int) -> str:
    return f"{dataset}__{normalization}__{balance_mode}__seed-{int(seed)}"


def run_config_payload(settings: SuiteSettings, entry: DatasetEntry, normalization: str, balance_mode: str, seed: int) -> dict[str, Any]:
    return {
        "dataset": entry.name,
        "adapter": entry.adapter,
        "dataset_root": str(entry.root),
        "dataset_options": entry.options,
        "target_size": settings.training.image_size_for(entry.name),
        "normalization": normalization,
        "balance_mode": balance_mode,
        "seed": int(seed),
        "train_fraction": settings.train_fraction,
        "validation_fraction": settings.validation_fraction,
        "test_fraction": settings.test_fraction,
        "training": settings.training.to_dict(),
    }
