from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


WSL_ROOT = "/mnt/c/source/repos/teste-modelos-tcc/"
WINDOWS_ROOT = "C:\\source\\repos\\teste-modelos-tcc\\"


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def portable_value(value: str) -> str:
    value = value.replace(WSL_ROOT, "")
    value = value.replace(WINDOWS_ROOT, "")
    return value.replace("\\", "/")


def normalize_source(source: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(source)
    if isinstance(result.get("path"), str):
        result["path"] = portable_value(result["path"])

    query = result.get("query")
    if isinstance(query, dict):
        if isinstance(query.get("sql"), str):
            query["sql"] = portable_value(query["sql"])
        if isinstance(query.get("tables_used"), list):
            query["tables_used"] = [
                portable_value(item) if isinstance(item, str) else item
                for item in query["tables_used"]
            ]
    return result


def without_source_payloads(value: Any) -> Any:
    if isinstance(value, list):
        return [without_source_payloads(item) for item in value]
    if not isinstance(value, dict):
        return value

    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"sources", "source"}:
            continue
        cleaned[key] = without_source_payloads(item)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a portable copy of the validated training report artifact."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--expected-generated-at",
        default="2026-08-02T18:05:22.296524+00:00",
    )
    parser.add_argument(
        "--fix-scatter-label-overflow",
        action="store_true",
        help="Remove the long direct point label from the hardware scatter only.",
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    source_artifact = json.loads(source_path.read_text(encoding="utf-8"))
    artifact = copy.deepcopy(source_artifact)

    generated_at = artifact["manifest"]["generatedAt"]
    if generated_at != args.expected_generated_at:
        raise ValueError(
            f"generatedAt mismatch: expected {args.expected_generated_at}, got {generated_at}"
        )

    manifest_sources = artifact["manifest"].get("sources", [])
    top_sources = artifact.get("sources", [])
    artifact["manifest"]["sources"] = [normalize_source(item) for item in manifest_sources]
    artifact["sources"] = [normalize_source(item) for item in top_sources]

    inline_source_count = 0
    for collection_name in ("cards", "charts", "tables"):
        for item in artifact["manifest"].get(collection_name, []):
            if isinstance(item.get("source"), dict):
                item["source"] = normalize_source(item["source"])
                inline_source_count += 1

    portability_corrections: list[dict[str, Any]] = []
    if args.fix_scatter_label_overflow:
        for chart in artifact["manifest"].get("charts", []):
            if chart.get("id") != "hardware_scatter":
                continue
            encodings = chart.get("encodings", {})
            removed = encodings.pop("label", None)
            if removed is not None:
                portability_corrections.append(
                    {
                        "artifact_path": "manifest.charts[hardware_scatter].encodings.label",
                        "change": "removed direct point label to prevent SVG horizontal overflow",
                        "removed_value": removed,
                        "data_field_preserved": "snapshot.datasets.hardware[].run_label",
                    }
                )

    source_without_paths = without_source_payloads(source_artifact)
    portable_without_paths = without_source_payloads(artifact)
    expected_without_paths = copy.deepcopy(source_without_paths)
    if args.fix_scatter_label_overflow:
        for chart in expected_without_paths["manifest"].get("charts", []):
            if chart.get("id") == "hardware_scatter":
                chart.get("encodings", {}).pop("label", None)
    if expected_without_paths != portable_without_paths:
        raise ValueError("Portable preparation made an unexpected report-content change")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    serialized = output_path.read_text(encoding="utf-8")
    forbidden_roots = [root for root in (WSL_ROOT, WINDOWS_ROOT) if root in serialized]
    if forbidden_roots:
        raise ValueError(f"Machine-local roots remain in portable artifact: {forbidden_roots}")

    summary = {
        "status": "passed",
        "source_artifact": str(source_path),
        "portable_artifact": str(output_path),
        "generated_at": generated_at,
        "title": artifact["manifest"]["title"],
        "surface": artifact["surface"],
        "snapshot_status": artifact["snapshot"]["status"],
        "counts": {
            "blocks": len(artifact["manifest"].get("blocks", [])),
            "cards": len(artifact["manifest"].get("cards", [])),
            "charts": len(artifact["manifest"].get("charts", [])),
            "tables": len(artifact["manifest"].get("tables", [])),
            "datasets": len(artifact["snapshot"].get("datasets", {})),
            "manifest_sources": len(artifact["manifest"].get("sources", [])),
            "inline_sources": inline_source_count,
        },
        "content_hashes": {
            "snapshot_before": stable_hash(source_artifact["snapshot"]),
            "snapshot_after": stable_hash(artifact["snapshot"]),
            "blocks_before": stable_hash(source_artifact["manifest"]["blocks"]),
            "blocks_after": stable_hash(artifact["manifest"]["blocks"]),
            "report_without_sources_before": stable_hash(source_without_paths),
            "report_without_sources_after": stable_hash(portable_without_paths),
        },
        "normalized_source_roots": [WSL_ROOT, WINDOWS_ROOT],
        "forbidden_roots_remaining": forbidden_roots,
        "portability_corrections": portability_corrections,
        "only_expected_portability_changes": expected_without_paths == portable_without_paths,
    }

    if summary["content_hashes"]["snapshot_before"] != summary["content_hashes"]["snapshot_after"]:
        raise ValueError("Snapshot changed during portable preparation")
    if summary["content_hashes"]["blocks_before"] != summary["content_hashes"]["blocks_after"]:
        raise ValueError("Report blocks changed during portable preparation")

    audit_path = args.audit.resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
