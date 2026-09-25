"""Ponto leve de inicialização, imports e configuração global do projeto.

Este módulo é deliberadamente seguro de importar em perfis sem TensorFlow.
Ele centraliza os caminhos padrão, o carregamento da configuração global e
o inventário dos imports do pacote, mas não antecipa imports opcionais ou
pesados. Cada módulo continua importando suas dependências próximo ao uso.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Final

from .config import DatasetEntry, SuiteSettings, load_dataset_registry, load_suite_settings


# Paths remain relative to the directory from which the CLI is invoked, which
# preserves the behavior of the existing command-line interface.
DEFAULT_CONFIG_DIRECTORY: Final = Path("configs")
DEFAULT_DATASET_REGISTRY: Final = DEFAULT_CONFIG_DIRECTORY / "datasets.yaml"
DEFAULT_SUITE_CONFIG: Final = DEFAULT_CONFIG_DIRECTORY / "suite.yaml"
DEFAULT_OUTPUT_ROOT: Final = Path("artifacts")


@dataclasses.dataclass(frozen=True)
class ProjectPaths:
    """Default paths consumed by the CLI and global configuration loader."""

    dataset_registry: Path = DEFAULT_DATASET_REGISTRY
    suite_config: Path = DEFAULT_SUITE_CONFIG
    output_root: Path = DEFAULT_OUTPUT_ROOT


DEFAULT_PROJECT_PATHS: Final = ProjectPaths()


@dataclasses.dataclass(frozen=True)
class GlobalConfiguration:
    """Validated runtime configuration resolved from the suite and registry."""

    suite: SuiteSettings
    datasets: dict[str, DatasetEntry]
    suite_path: Path
    registry_path: Path


# Static inventory of every import family used by modules in ``src/tcc_benchmark``.
# It is documentation data, not a set of eager imports: importing TensorFlow,
# Matplotlib or Pillow here would break the lightweight audit-only profile.
IMPORT_CATALOG: Final[dict[str, tuple[str, ...]]] = {
    "standard_library": (
        "argparse",
        "collections",
        "contextlib",
        "csv",
        "dataclasses",
        "datetime",
        "gzip",
        "hashlib",
        "html",
        "importlib.metadata",
        "io",
        "json",
        "math",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "re",
        "shutil",
        "statistics",
        "struct",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "traceback",
        "typing",
        "unicodedata",
        "zlib",
    ),
    "third_party": (
        "matplotlib",
        "matplotlib.pyplot",
        "numpy",
        "PIL",
        "PIL.Image",
        "psutil",
        "pynvml",
        "scipy.io",
        "sklearn.metrics",
        "tensorflow",
        "yaml",
    ),
    "local_package": (
        "tcc_benchmark.adapters",
        "tcc_benchmark.audit",
        "tcc_benchmark.cli",
        "tcc_benchmark.config",
        "tcc_benchmark.controlled",
        "tcc_benchmark.data",
        "tcc_benchmark.metrics",
        "tcc_benchmark.model",
        "tcc_benchmark.preflight",
        "tcc_benchmark.quantization",
        "tcc_benchmark.reporting",
        "tcc_benchmark.runner",
        "tcc_benchmark.state",
        "tcc_benchmark.telemetry",
    ),
}


def resolve_project_path(path: str | Path) -> Path:
    """Expand and resolve a path using the current working directory."""

    return Path(path).expanduser().resolve()


def load_global_configuration(
    *,
    suite_path: str | Path = DEFAULT_SUITE_CONFIG,
    registry_path: str | Path = DEFAULT_DATASET_REGISTRY,
    output_root: str | Path | None = None,
) -> GlobalConfiguration:
    """Load the suite and local dataset registry from one central entry point."""

    resolved_suite_path = resolve_project_path(suite_path)
    resolved_registry_path = resolve_project_path(registry_path)
    settings = load_suite_settings(resolved_suite_path)
    resolved_output_root = resolve_project_path(output_root if output_root is not None else settings.output_root)
    settings = dataclasses.replace(settings, output_root=resolved_output_root)
    settings.validate()
    return GlobalConfiguration(
        suite=settings,
        datasets=load_dataset_registry(resolved_registry_path),
        suite_path=resolved_suite_path,
        registry_path=resolved_registry_path,
    )


def configure_training_runtime() -> None:
    """Apply project-wide TensorFlow environment settings before importing it."""

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


__all__ = [
    "DEFAULT_CONFIG_DIRECTORY",
    "DEFAULT_DATASET_REGISTRY",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PROJECT_PATHS",
    "DEFAULT_SUITE_CONFIG",
    "GlobalConfiguration",
    "IMPORT_CATALOG",
    "ProjectPaths",
    "configure_training_runtime",
    "load_global_configuration",
    "resolve_project_path",
]
