"""Build the bounded report-artifact payload from the dataset inventory files.

The generated ``artifact.json`` is the exact payload validated and rendered by
the Data Analytics report surface.  Keeping it beside the CSV/JSON evidence
makes the reader-facing report reproducible without embedding a second report
implementation in the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _class_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "dataset": row["dataset"],
                "class_index": int(row["class_index"]),
                "class_name": row["class_name"],
                "total": int(row["total"]),
            }
            for row in csv.DictReader(handle)
        ]


def _comparison_rows(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        splits = "; ".join(f"{name}: {count}" for name, count in profile["original_splits"].items())
        archives = "; ".join(
            str(item["filename"]) for item in profile["source_metadata"].get("sources", []) if item.get("filename")
        )
        rows.append(
            {
                "dataset": profile["dataset"],
                "domain": profile.get("context", {}).get("domain", "não documentado"),
                "total_images": int(profile["total_images"]),
                "num_classes": int(profile["num_classes"]),
                "color_mode": profile["native_summary"]["color_mode"],
                "native_shapes": profile["native_summary"]["shapes"],
                "disk_bytes": int(profile["disk"]["root_bytes"]),
                "disk_mib": round(int(profile["disk"]["root_bytes"]) / 1024**2, 2),
                "disk_human": profile["disk"]["root_bytes_human"],
                "file_count": int(profile["disk"]["root_file_count"]),
                "class_min": int(profile["class_min_count"]),
                "class_max": int(profile["class_max_count"]),
                "imbalance_ratio": round(float(profile["imbalance_ratio_max_over_min"]), 4),
                "original_splits": splits,
                "inspection_scope": profile["native_images"]["inspection_scope"],
                "source_archives": archives or "não informado no SOURCE.json",
            }
        )
    return rows


def _profile_block(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": f"dataset-{row['dataset']}",
        "type": "markdown",
        "sourceId": "local_inventory",
        "body": (
            f"## {row['dataset']}\n\n"
            f"- **Domínio:** {row['domain']}.\n"
            f"- **Volume local:** {row['total_images']:,} imagens, {row['num_classes']} classes e "
            f"{row['file_count']:,} arquivos ({row['disk_human']}).\n"
            f"- **Imagem nativa observada:** {row['color_mode']}; {row['native_shapes']}.\n"
            f"- **Splits de origem:** {row['original_splits']}.\n"
            f"- **Distribuição por classe:** mínimo {row['class_min']:,}, máximo {row['class_max']:,}; "
            f"razão máximo/mínimo {row['imbalance_ratio']:.2f}.\n"
            f"- **Escopo da propriedade de imagem:** {row['inspection_scope']}.\n"
            f"- **Arquivos de origem registrados:** {row['source_archives']}."
        ),
    }


def build_artifact(profile_payload: dict[str, Any], class_distribution: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = _comparison_rows(profile_payload["datasets"])
    total_images = sum(row["total_images"] for row in comparison)
    total_bytes = sum(row["disk_bytes"] for row in comparison)
    headline = [
        {
            "dataset_count": len(comparison),
            "total_images": total_images,
            "total_storage_gib": round(total_bytes / 1024**3, 2),
            "grayscale_datasets": sum(row["color_mode"].startswith("grayscale") for row in comparison),
            "rgb_datasets": sum(row["color_mode"].startswith("RGB") for row in comparison),
        }
    ]
    generated_at = profile_payload["generated_at_utc"]
    title = "Inventário técnico dos datasets locais do TCC"
    source = {
        "id": "local_inventory",
        "label": "Inventário local reprodutível dos datasets",
        "path": "outputs/dataset-inventory-2026-07-28/dataset_profile.json",
        "query": {
            "id": "profile_datasets_local",
            "engine": "local CSV/JSON",
            "language": "sql",
            "description": "Perfil local produzido pelos adaptadores do benchmark, com medidas de disco e indexação de arquivos no Windows.",
            "executed_at": generated_at,
            "sql": (
                "SELECT dataset, domain, total_images, num_classes, color_mode, native_shapes, disk_bytes, "
                "file_count, class_min, class_max, imbalance_ratio, original_splits "
                "FROM dataset_summary ORDER BY total_images DESC;"
            ),
            "tables_used": [
                "configs/datasets.yaml",
                "datasets/<dataset>/",
                "outputs/dataset-inventory-2026-07-28/dataset_profile.json",
                "outputs/dataset-inventory-2026-07-28/class_distribution.csv",
                "outputs/dataset-inventory-2026-07-28/disk_profile.json",
                "outputs/dataset-inventory-2026-07-28/folder_index.json",
            ],
            "filters": [
                "10 datasets configurados em configs/datasets.yaml",
                "até 24 arquivos por classe em datasets armazenados como arquivos",
                "matrizes de imagem inspecionadas integralmente",
            ],
            "metric_definitions": [
                "total_images = soma dos exemplos de todos os splits locais expostos pelo adaptador",
                "imbalance_ratio = maior contagem de classe / menor contagem de classe nos splits locais unidos",
                "disk_bytes = soma dos tamanhos de todos os arquivos abaixo da raiz local do dataset",
                "class_distribution = contagens exatas por classe e split original",
            ],
        },
    }
    manifest: dict[str, Any] = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Perfil local e reprodutível dos 10 datasets configurados para o benchmark do TCC.",
        "generatedAt": generated_at,
        "sources": [source],
        "cards": [
            {
                "id": "coverage",
                "dataset": "headline",
                "sourceId": "local_inventory",
                "metrics": [
                    {"label": "Datasets", "field": "dataset_count", "format": "number"},
                    {"label": "Imagens locais", "field": "total_images", "format": "compact"},
                    {"label": "Armazenamento local (GiB)", "field": "total_storage_gib", "format": "number"},
                    {"label": "Datasets em tons de cinza", "field": "grayscale_datasets", "format": "number"},
                    {"label": "Datasets RGB", "field": "rgb_datasets", "format": "number"},
                ],
            }
        ],
        "charts": [
            {
                "id": "images-by-dataset",
                "title": "Quantidade de imagens por dataset",
                "subtitle": "Contagem local total antes da redistribuição estratificada 70%/15%/15% do benchmark.",
                "type": "bar",
                "dataset": "dataset_comparison",
                "sourceId": "local_inventory",
                "layout": "full",
                "encodings": {
                    "x": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                    "y": {"field": "total_images", "type": "quantitative", "format": "compact", "label": "Imagens"},
                    "tooltip": [
                        {"field": "dataset", "type": "nominal"},
                        {"field": "total_images", "type": "quantitative", "format": "number"},
                        {"field": "num_classes", "type": "quantitative", "format": "number"},
                    ],
                },
            },
            {
                "id": "disk-by-dataset",
                "title": "Armazenamento local por dataset",
                "subtitle": "Tamanho da raiz configurada, incluindo imagens, rótulos e metadados presentes localmente.",
                "type": "bar",
                "dataset": "dataset_comparison",
                "sourceId": "local_inventory",
                "layout": "full",
                "encodings": {
                    "x": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                    "y": {"field": "disk_mib", "type": "quantitative", "format": "number", "label": "MiB"},
                    "tooltip": [
                        {"field": "dataset", "type": "nominal"},
                        {"field": "disk_mib", "type": "quantitative", "format": "number"},
                        {"field": "file_count", "type": "quantitative", "format": "number"},
                    ],
                },
            },
            {
                "id": "imbalance-by-dataset",
                "title": "Razão entre a maior e a menor classe",
                "subtitle": "Uma razão igual a 1 indica classes naturalmente balanceadas; valores maiores mostram desequilíbrio natural local.",
                "type": "bar",
                "dataset": "dataset_comparison",
                "sourceId": "local_inventory",
                "layout": "full",
                "encodings": {
                    "x": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                    "y": {"field": "imbalance_ratio", "type": "quantitative", "format": "number", "label": "Razão máximo/mínimo"},
                    "tooltip": [
                        {"field": "dataset", "type": "nominal"},
                        {"field": "class_min", "type": "quantitative", "format": "number"},
                        {"field": "class_max", "type": "quantitative", "format": "number"},
                        {"field": "imbalance_ratio", "type": "quantitative", "format": "number"},
                    ],
                },
            },
        ],
        "tables": [
            {
                "id": "dataset-comparison-table",
                "dataset": "dataset_comparison",
                "sourceId": "local_inventory",
                "columns": [
                    {"field": "dataset", "label": "Dataset"},
                    {"field": "domain", "label": "Domínio"},
                    {"field": "total_images", "label": "Imagens", "format": "number"},
                    {"field": "num_classes", "label": "Classes", "format": "number"},
                    {"field": "color_mode", "label": "Canais/cor"},
                    {"field": "native_shapes", "label": "Formas nativas observadas"},
                    {"field": "disk_human", "label": "Disco"},
                    {"field": "file_count", "label": "Arquivos", "format": "number"},
                    {"field": "imbalance_ratio", "label": "Razão máx./mín.", "format": "number"},
                    {"field": "original_splits", "label": "Splits originais"},
                ],
                "defaultSort": {"field": "total_images", "direction": "desc"},
            },
            {
                "id": "class-distribution-table",
                "dataset": "class_distribution",
                "sourceId": "local_inventory",
                "columns": [
                    {"field": "dataset", "label": "Dataset"},
                    {"field": "class_index", "label": "Índice", "format": "number"},
                    {"field": "class_name", "label": "Classe"},
                    {"field": "total", "label": "Imagens locais", "format": "number"},
                ],
                "defaultSort": {"field": "total", "direction": "desc"},
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "summary",
                "type": "markdown",
                "sourceId": "local_inventory",
                "body": (
                    "## Resumo técnico\n\n"
                    "O inventário cobre **10 datasets**, **906.046 imagens locais** e aproximadamente **1,99 GiB** de "
                    "armazenamento. Cinco datasets são nativamente em tons de cinza (1 canal) e cinco são RGB "
                    "(3 canais após decodificação). CINIC-10 concentra o maior volume de exemplos (270.000) e também "
                    "o maior diretório local; GTSRB e FER2013 apresentam o maior desequilíbrio natural de classes."
                ),
            },
            {"id": "metric-strip", "type": "metric-strip", "cardIds": ["coverage"]},
            {
                "id": "method",
                "type": "markdown",
                "sourceId": "local_inventory",
                "body": (
                    "## Método e definições\n\n"
                    "Os rótulos, nomes de classes e splits vieram dos mesmos adaptadores usados no benchmark. Para arrays, "
                    "shape, canais, dtype e faixa de valores foram inspecionados integralmente. Para datasets em arquivos, "
                    "a propriedade visual foi observada em até 24 imagens por classe, com escopo registrado em cada perfil; "
                    "a contagem de arquivos e o tamanho em disco foram medidos integralmente no Windows. O benchmark depois "
                    "reúne os splits locais e cria nova divisão estratificada 70%/15%/15% por seed."
                ),
            },
            {"id": "images-chart", "type": "chart", "chartId": "images-by-dataset"},
            {"id": "disk-chart", "type": "chart", "chartId": "disk-by-dataset"},
            {"id": "imbalance-chart", "type": "chart", "chartId": "imbalance-by-dataset"},
            {
                "id": "comparison-heading",
                "type": "markdown",
                "body": "## Comparação estruturada\n\nA tabela reúne tamanho, cor/canais, formas nativas observadas, número de classes, arquivos e splits de origem.",
            },
            {"id": "comparison-table", "type": "table", "tableId": "dataset-comparison-table"},
            {
                "id": "profiles-heading",
                "type": "markdown",
                "body": "## Perfil individual de cada dataset\n\nOs valores por classe completos estão na tabela seguinte e no CSV de distribuição, incluindo colunas de split original.",
            },
            *[_profile_block(row) for row in comparison],
            {
                "id": "class-heading",
                "type": "markdown",
                "body": "## Contagem por classe\n\nA tabela contém as 177 classes expostas pelos adaptadores. O arquivo `class_distribution.csv` preserva também as colunas de cada split de origem.",
            },
            {"id": "class-table", "type": "table", "tableId": "class-distribution-table"},
            {
                "id": "limits",
                "type": "markdown",
                "sourceId": "local_inventory",
                "body": (
                    "## Limitações e qualidade dos dados\n\n"
                    "Os relatórios de auditoria existentes foram lidos e não habilitaram `verify_images` nem `hash_images`. "
                    "Portanto, este relatório não afirma que todos os pixels foram verificados contra corrupção nem que uma "
                    "deduplicação completa por imagem foi realizada. Para datasets em pastas, dimensões/formato são uma amostra "
                    "declarada; os totais, rótulos, splits e tamanhos de diretório são contagens locais completas."
                ),
            },
            {
                "id": "next",
                "type": "markdown",
                "body": (
                    "## Próximos passos\n\n"
                    "Use o inventário para decidir a matriz de normalização e balanceamento. Priorize a análise de `class_weight`, "
                    "`undersample` e `oversample` em GTSRB, FER2013 e SVHN; nos datasets com razão 1, as estratégias de "
                    "balanceamento devem ser marcadas como equivalentes/no-op conforme a suíte."
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "headline": headline,
                "dataset_comparison": comparison,
                "class_distribution": class_distribution,
            },
        },
        "sources": [source],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=PROJECT_ROOT / "outputs" / "dataset-inventory-2026-07-28")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inventory = args.inventory.expanduser().resolve()
    output = (args.output or inventory / "artifact.json").expanduser().resolve()
    artifact = build_artifact(_read_json(inventory / "dataset_profile.json"), _class_rows(inventory / "class_distribution.csv"))
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
