"""Create a reproducible, local inventory of every configured dataset.

The benchmark adapters are reused so the inventory observes the same labels and
splits that the training suite will consume.  No dataset is downloaded and this
script does not initialise TensorFlow or reserve GPU memory.

For array-backed datasets, native image dimensions, channel counts, dtypes and
value ranges are calculated across all loaded arrays.  Folder-backed datasets
can contain many files and varying dimensions, so they are inspected using a
deterministic, label-stratified sample; the report records that scope rather
than presenting sampled properties as a full-file guarantee.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcc_benchmark.adapters import LoadedDataset, SampleRecord, load_local_dataset  # noqa: E402
from tcc_benchmark.config import DATASET_ORDER, DatasetEntry, load_dataset_registry  # noqa: E402


CONTEXT: dict[str, dict[str, str]] = {
    "mnist": {
        "domain": "dígitos manuscritos",
        "content": "algarismos de 0 a 9 escritos à mão",
        "label_scheme": "dígito numérico 0–9",
    },
    "fashion_mnist": {
        "domain": "artigos de moda",
        "content": "imagens em escala de cinza de vestuário e acessórios",
        "label_scheme": "10 categorias de artigo de moda",
    },
    "kmnist": {
        "domain": "caracteres japoneses manuscritos",
        "content": "caracteres Kuzushiji",
        "label_scheme": "10 classes de caracteres Kuzushiji",
    },
    "emnist_balanced": {
        "domain": "caracteres manuscritos",
        "content": "dígitos e letras do EMNIST Balanced",
        "label_scheme": "47 classes balanceadas na definição original",
    },
    "cifar10": {
        "domain": "objetos e cenas naturais",
        "content": "pequenas imagens coloridas de 10 categorias",
        "label_scheme": "10 categorias semânticas",
    },
    "cifar100_coarse": {
        "domain": "objetos e cenas naturais",
        "content": "CIFAR-100 agregado em superclasses",
        "label_scheme": "20 superclasses (rótulos coarse)",
    },
    "cinic10": {
        "domain": "objetos e cenas naturais",
        "content": "imagens de 10 categorias, combinação de CIFAR-10 e imagens derivadas do ImageNet",
        "label_scheme": "10 categorias compatíveis com CIFAR-10",
    },
    "svhn": {
        "domain": "dígitos em cenas urbanas",
        "content": "números de casas fotografados em ambiente real",
        "label_scheme": "dígitos 0–9; o adaptador normaliza o 10 do arquivo SVHN para 0",
    },
    "gtsrb": {
        "domain": "sinalização de trânsito",
        "content": "placas de trânsito alemãs fotografadas em condições reais",
        "label_scheme": "identificador numérico de 43 classes de placa",
    },
    "fer2013": {
        "domain": "expressões faciais",
        "content": "faces em escala de cinza para reconhecimento de emoção",
        "label_scheme": "7 emoções: angry, disgust, fear, happy, sad, surprise, neutral",
    },
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Não é possível serializar {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_human(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def _counter_rows(counter: Counter[Any], *, key: str = "value") -> list[dict[str, Any]]:
    return [{key: str(value), "count": int(count)} for value, count in sorted(counter.items(), key=lambda item: str(item[0]))]


def _disk_profile(root: Path) -> dict[str, Any]:
    total_bytes = 0
    total_files = 0
    extension_counts: Counter[str] = Counter()
    extension_bytes: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        suffix = path.suffix.lower() or "[sem extensão]"
        total_bytes += size
        total_files += 1
        extension_counts[suffix] += 1
        extension_bytes[suffix] += size
    return {
        "root_bytes": total_bytes,
        "root_bytes_human": _bytes_human(total_bytes),
        "root_file_count": total_files,
        "extension_distribution": [
            {"extension": extension, "files": int(extension_counts[extension]), "bytes": int(extension_bytes[extension])}
            for extension in sorted(extension_counts, key=lambda item: (-extension_bytes[item], item))
        ],
    }


def _source_metadata(root: Path) -> dict[str, Any]:
    source_path = root / "SOURCE.json"
    if not source_path.exists():
        return {"present": False, "sources": []}
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"present": True, "read_error": str(error), "sources": []}
    records = []
    for item in source.get("sources", []):
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "filename": item.get("filename"),
                "url": item.get("url"),
                "md5": item.get("md5"),
                "sha256": item.get("sha256"),
                "declared_bytes": item.get("size"),
            }
        )
    return {
        "present": True,
        "dataset": source.get("dataset"),
        "installed_at_utc": source.get("installed_at_utc"),
        "sources": records,
        "source_file_sha256": _sha256(source_path),
    }


def _load_audit(dataset_name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "artifacts" / dataset_name / "audit" / "audit.json"
    if not path.exists():
        return {"available": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"available": True, "read_error": str(error)}
    options = payload.get("options", {})
    summary = payload.get("summary", {})
    return {
        "available": True,
        "generated_at_utc": payload.get("generated_at_utc"),
        "total_samples": payload.get("total_samples"),
        "split_counts": payload.get("split_counts", {}),
        "healthy": summary.get("healthy"),
        "errors": summary.get("errors"),
        "warnings": summary.get("warnings"),
        "hash_images": bool(options.get("hash_images", False)),
        "verify_images": bool(options.get("verify_images", False)),
        "path": str(path.relative_to(PROJECT_ROOT)),
    }


def _array_native_profile(dataset: LoadedDataset) -> dict[str, Any]:
    shape_counts: Counter[tuple[int, int, int]] = Counter()
    dtype_counts: Counter[str] = Counter()
    channel_counts: Counter[int] = Counter()
    value_min: float | None = None
    value_max: float | None = None
    image_count = 0
    for split in dataset.splits.values():
        if split.images is None:
            continue
        images = np.asarray(split.images)
        if images.ndim != 4:
            raise ValueError(f"'{dataset.name}' retornou matriz de imagens com shape inesperado: {images.shape}")
        shape = tuple(int(value) for value in images.shape[1:])
        shape_counts[shape] += int(images.shape[0])
        dtype_counts[str(images.dtype)] += int(images.shape[0])
        channel_counts[int(shape[-1])] += int(images.shape[0])
        image_count += int(images.shape[0])
        current_min = float(np.min(images))
        current_max = float(np.max(images))
        value_min = current_min if value_min is None else min(value_min, current_min)
        value_max = current_max if value_max is None else max(value_max, current_max)
    return {
        "inspection_scope": "completo: todas as matrizes carregadas pelo adaptador",
        "decoded_samples": image_count,
        "shape_distribution": [
            {"height": shape[0], "width": shape[1], "channels": shape[2], "count": int(count)}
            for shape, count in sorted(shape_counts.items())
        ],
        "dtype_distribution": _counter_rows(dtype_counts, key="dtype"),
        "channel_distribution": _counter_rows(channel_counts, key="channels"),
        "value_range": {"minimum": value_min, "maximum": value_max},
        "storage_format_distribution": [],
        "decode_errors": [],
    }


def _select_records(records: Iterable[SampleRecord], per_class: int) -> list[SampleRecord]:
    grouped: dict[int, list[SampleRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: (item.label, str(item.path))):
        if len(grouped[record.label]) < per_class:
            grouped[record.label].append(record)
    return [record for label in sorted(grouped) for record in grouped[label]]


def _path_native_profile(paths: Iterable[Path], inspection_scope: str) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment prerequisite
        raise RuntimeError("Pillow é necessário para inspecionar imagens em arquivos.") from error

    shapes: Counter[tuple[int, int, int]] = Counter()
    modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    channels: Counter[int] = Counter()
    decode_errors: list[dict[str, str]] = []
    selected = list(paths)
    for path in selected:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                image_format = image.format or "desconhecido"
                channels_count = 1 if mode in {"1", "L", "I", "I;16", "F", "LA"} else 3
                shapes[(height, width, channels_count)] += 1
                modes[mode] += 1
                formats[image_format] += 1
                dtypes[str(np.asarray(image).dtype)] += 1
                channels[channels_count] += 1
        except Exception as error:  # record errors instead of hiding a corrupt image
            decode_errors.append({"path": str(path), "error": str(error)})
    return {
        "inspection_scope": inspection_scope,
        "decoded_samples": len(selected),
        "shape_distribution": [
            {"height": shape[0], "width": shape[1], "channels": shape[2], "count": int(count)}
            for shape, count in sorted(shapes.items())
        ],
        "dtype_distribution": _counter_rows(dtypes, key="dtype"),
        "channel_distribution": _counter_rows(channels, key="channels"),
        "source_mode_distribution": _counter_rows(modes, key="mode"),
        "storage_format_distribution": _counter_rows(formats, key="format"),
        "decode_errors": decode_errors,
    }


def _record_native_profile(dataset: LoadedDataset, per_class: int) -> dict[str, Any]:
    selected = _select_records(dataset.iter_records(), per_class)
    return _path_native_profile(
        (record.path for record in selected),
        "amostra estratificada determinística de até "
        f"{per_class} arquivo(s) por classe; não é uma verificação integral de todos os arquivos",
    )


def _class_profiles(dataset: LoadedDataset) -> tuple[list[dict[str, Any]], dict[str, int]]:
    overall = Counter(int(value) for value in dataset.labels_for())
    split_counts = {split_name: Counter(int(value) for value in dataset.labels_for(split_name)) for split_name in dataset.split_names}
    rows: list[dict[str, Any]] = []
    for label, class_name in enumerate(dataset.class_names):
        row = {"class_index": label, "class_name": class_name, "total": int(overall[label])}
        row.update({f"split_{split_name}": int(counts[label]) for split_name, counts in split_counts.items()})
        rows.append(row)
    return rows, {split_name: int(sum(counts.values())) for split_name, counts in split_counts.items()}


def _native_summary(native: dict[str, Any]) -> tuple[str, str, str]:
    shape_rows = native.get("shape_distribution", [])
    if not shape_rows:
        return "não observado", "não observado", "não observado"
    shape_text = "; ".join(
        f"{row['height']}×{row['width']}×{row['channels']} (n={row['count']})" for row in shape_rows[:8]
    )
    if len(shape_rows) > 8:
        shape_text += f"; +{len(shape_rows) - 8} forma(s)"
    channel_text = ", ".join(f"{row['channels']} canal(is): {row['count']}" for row in native.get("channel_distribution", []))
    formats = native.get("storage_format_distribution", [])
    format_text = ", ".join(f"{row['format']}: {row['count']}" for row in formats) if formats else "matriz/binário local"
    return shape_text, channel_text, format_text


def _dataset_profile(entry: DatasetEntry, sample_per_class: int, disk_profiles: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset = load_local_dataset(entry.adapter, entry.root, **entry.options)
    classes, split_totals = _class_profiles(dataset)
    native = _array_native_profile(dataset) if any(split.has_arrays for split in dataset.splits.values()) else _record_native_profile(dataset, sample_per_class)
    counts = [row["total"] for row in classes]
    disk = (disk_profiles or {}).get(entry.name) or _disk_profile(entry.root)
    shape_text, channel_text, format_text = _native_summary(native)
    channels_seen = {int(row["channels"]) for row in native.get("channel_distribution", [])}
    if channels_seen == {1}:
        color_mode = "grayscale (1 canal)"
    elif channels_seen == {3}:
        color_mode = "RGB (3 canais após decodificação)"
    elif channels_seen:
        color_mode = "misto/variável; consulte distribuição de canais"
    else:
        color_mode = "não observado"
    return {
        "dataset": entry.name,
        "adapter": entry.adapter,
        "context": CONTEXT.get(entry.name, {}),
        "local_root": str(entry.root),
        "source_metadata": _source_metadata(entry.root),
        "disk": disk,
        "adapter_metadata": dataset.metadata,
        "original_splits": split_totals,
        "total_images": dataset.sample_count(),
        "num_classes": dataset.num_classes,
        "class_names": dataset.class_names,
        "class_distribution": classes,
        "class_min_count": int(min(counts)),
        "class_max_count": int(max(counts)),
        "imbalance_ratio_max_over_min": float(max(counts) / min(counts)),
        "native_images": native,
        "native_summary": {
            "color_mode": color_mode,
            "shapes": shape_text,
            "channels": channel_text,
            "storage_formats": format_text,
        },
        "audit": _load_audit(entry.name),
    }


def _folder_index_profile(entry: DatasetEntry, indexed: dict[str, Any], disk_profiles: dict[str, Any] | None = None) -> dict[str, Any]:
    classes = list(indexed["class_distribution"])
    class_names = [str(value) for value in indexed["class_names"]]
    counts = [int(row["total"]) for row in classes]
    sample_paths = [PROJECT_ROOT / str(path) for path in indexed.get("sample_paths", [])]
    native = _path_native_profile(sample_paths, str(indexed["image_inspection_scope"]))
    disk = (disk_profiles or {}).get(entry.name) or _disk_profile(entry.root)
    shape_text, channel_text, format_text = _native_summary(native)
    channels_seen = {int(row["channels"]) for row in native.get("channel_distribution", [])}
    color_mode = "grayscale (1 canal)" if channels_seen == {1} else "RGB (3 canais após decodificação)" if channels_seen == {3} else "misto/variável; consulte distribuição de canais"
    return {
        "dataset": entry.name,
        "adapter": entry.adapter,
        "context": CONTEXT.get(entry.name, {}),
        "local_root": str(entry.root),
        "source_metadata": _source_metadata(entry.root),
        "disk": disk,
        "adapter_metadata": {"profile_source": "índice nativo do Windows para evitar enumeração lenta por /mnt/c"},
        "original_splits": {str(name): int(value) for name, value in indexed["original_splits"].items()},
        "total_images": int(sum(counts)),
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_distribution": classes,
        "class_min_count": int(min(counts)),
        "class_max_count": int(max(counts)),
        "imbalance_ratio_max_over_min": float(max(counts) / min(counts)),
        "native_images": native,
        "native_summary": {"color_mode": color_mode, "shapes": shape_text, "channels": channel_text, "storage_formats": format_text},
        "audit": _load_audit(entry.name),
    }


def _summary_row(profile: dict[str, Any]) -> dict[str, Any]:
    context = profile.get("context", {})
    return {
        "dataset": profile["dataset"],
        "domain": context.get("domain", "não documentado"),
        "total_images": profile["total_images"],
        "classes": profile["num_classes"],
        "color_mode": profile["native_summary"]["color_mode"],
        "native_shapes": profile["native_summary"]["shapes"],
        "storage_formats": profile["native_summary"]["storage_formats"],
        "disk_bytes": profile["disk"]["root_bytes"],
        "disk_human": profile["disk"]["root_bytes_human"],
        "files": profile["disk"]["root_file_count"],
        "class_min": profile["class_min_count"],
        "class_max": profile["class_max_count"],
        "imbalance_ratio": profile["imbalance_ratio_max_over_min"],
        "original_splits": "; ".join(f"{name}: {count}" for name, count in profile["original_splits"].items()),
        "image_property_scope": profile["native_images"]["inspection_scope"],
        "audit_healthy": profile["audit"].get("healthy"),
        "audit_verified_images": profile["audit"].get("verify_images", False),
        "audit_hashed_images": profile["audit"].get("hash_images", False),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_summary(profiles: list[dict[str, Any]]) -> str:
    lines = [
        "# Inventário local dos datasets",
        "",
        "Este arquivo é um guia do inventário. Os valores completos e auditáveis estão em `dataset_profile.json`, "
        "`dataset_summary.csv` e `class_distribution.csv`.",
        "",
        "## Escopo e método",
        "",
        "- Os rótulos, nomes de classes e splits foram lidos pelos adaptadores locais usados pelo benchmark.",
        "- Dados em matrizes foram inspecionados integralmente. Imagens armazenadas como arquivos foram decodificadas "
        "em amostra estratificada por classe; o tamanho em disco e as contagens de arquivo percorrem todo o diretório.",
        "- Não foi feito download, treinamento ou inicialização de TensorFlow/GPU.",
        "- Os arquivos `artifacts/<dataset>/audit/audit.json` existentes não habilitaram verificação/hashing integral de "
        "imagens; portanto, eles não comprovam ausência de duplicatas ou corrupção em todos os pixels.",
        "",
        "## Resumo por dataset",
        "",
        "| Dataset | Imagens | Classes | Modalidade nativa | Formas nativas observadas | Disco | Menor–maior classe | Razão |",
        "|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for profile in profiles:
        native = profile["native_summary"]
        lines.append(
            "| {dataset} | {total_images:,} | {num_classes} | {color_mode} | {shapes} | {disk} | {minimum:,}–{maximum:,} | {ratio:.2f} |".format(
                dataset=profile["dataset"],
                total_images=profile["total_images"],
                num_classes=profile["num_classes"],
                color_mode=native["color_mode"],
                shapes=native["shapes"].replace("|", "/"),
                disk=profile["disk"]["root_bytes_human"],
                minimum=profile["class_min_count"],
                maximum=profile["class_max_count"],
                ratio=profile["imbalance_ratio_max_over_min"],
            )
        )
    lines.extend(["", "## Leitura recomendada", "", "1. Comece por `dataset_summary.csv` para comparações entre datasets.", "2. Use `class_distribution.csv` para as contagens exatas por classe e split original.", "3. Use `dataset_profile.json` para detalhes de formatos, modos, metadados da origem, distribuição de formas e limitações.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "dataset-inventory")
    parser.add_argument("--sample-per-class", type=int, default=24, help="Máximo de arquivos decodificados por classe para datasets folder-backed.")
    parser.add_argument(
        "--disk-profile",
        type=Path,
        help="JSON gerado por scripts/profile_dataset_disk.ps1; evita a varredura lenta de /mnt/c no WSL.",
    )
    parser.add_argument(
        "--folder-index",
        type=Path,
        help="JSON de scripts/index_folder_datasets.ps1; substitui a enumeração lenta de CINIC-10/GTSRB no WSL.",
    )
    args = parser.parse_args()
    if args.sample_per_class < 1:
        parser.error("--sample-per-class deve ser positivo")

    registry = load_dataset_registry(args.registry)
    selected_names = [name for name in DATASET_ORDER if name in registry]
    missing = [name for name in DATASET_ORDER if name not in registry]
    if missing:
        raise RuntimeError(f"Datasets ausentes no registro: {', '.join(missing)}")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    disk_profiles: dict[str, Any] | None = None
    if args.disk_profile is not None:
        disk_payload = json.loads(args.disk_profile.expanduser().read_text(encoding="utf-8"))
        disk_profiles = disk_payload.get("datasets")
        if not isinstance(disk_profiles, dict):
            raise RuntimeError("O JSON de --disk-profile não possui o mapa 'datasets'.")
    folder_indexes: dict[str, Any] = {}
    if args.folder_index is not None:
        folder_payload = json.loads(args.folder_index.expanduser().read_text(encoding="utf-8"))
        folder_indexes = folder_payload.get("datasets")
        if not isinstance(folder_indexes, dict):
            raise RuntimeError("O JSON de --folder-index não possui o mapa 'datasets'.")
    profiles: list[dict[str, Any]] = []
    for index, name in enumerate(selected_names, start=1):
        print(f"[{index}/{len(selected_names)}] Perfilando {name}...", flush=True)
        if name in folder_indexes:
            profiles.append(_folder_index_profile(registry[name], folder_indexes[name], disk_profiles))
        else:
            profiles.append(_dataset_profile(registry[name], args.sample_per_class, disk_profiles))

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary_rows = [_summary_row(profile) for profile in profiles]
    class_rows = [
        {"dataset": profile["dataset"], **row}
        for profile in profiles
        for row in profile["class_distribution"]
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "registry": str(args.registry.expanduser().resolve()),
        "method": {
            "array_images": "inspeção integral de shapes, canais, dtype e faixa de valores",
            "file_images": f"amostra estratificada determinística de até {args.sample_per_class} arquivo(s) por classe",
            "disk_usage": "varredura de todos os arquivos abaixo da raiz configurada",
            "training_or_gpu": "não utilizado",
        },
        "datasets": profiles,
    }
    _write_json(output / "dataset_profile.json", payload)
    _write_csv(output / "dataset_summary.csv", summary_rows)
    _write_csv(output / "class_distribution.csv", class_rows)
    (output / "README.md").write_text(_markdown_summary(profiles), encoding="utf-8")
    print(f"Inventário salvo em: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
