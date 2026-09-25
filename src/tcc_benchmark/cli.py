"""Interface de linha de comando da suíte de benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .bootstrap import DEFAULT_DATASET_REGISTRY, DEFAULT_OUTPUT_ROOT, DEFAULT_SUITE_CONFIG
from .config import DATASET_ORDER


def _add_common_config_arguments(parser: argparse.ArgumentParser, *, registry_required: bool = False) -> None:
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_DATASET_REGISTRY,
        required=registry_required,
        help="Registro YAML de caminhos locais (padrão: configs/datasets.yaml).",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE_CONFIG,
        help="Arquivo YAML com parâmetros fixos da suíte.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Sobrescreve output_root do suite.yaml.",
    )


def _add_dataset_selector(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=DATASET_ORDER, help="Dataset a processar.")
    group.add_argument("--all", action="store_true", help="Processa todos os datasets do registro, sequencialmente.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcc-benchmark",
        description="Suíte local, retomável e observável de benchmarks de imagens.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Exibe diagnóstico de ambiente sem treinar.")

    preflight.add_argument("--require-tensorflow", action="store_true")
    preflight.add_argument("--data-path", type=Path, action="append", default=[])

    audit = subparsers.add_parser("audit", help="Audita datasets locais; não baixa dados.")
    _add_common_config_arguments(audit)
    _add_dataset_selector(audit)
    audit.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita a auditoria para diagnóstico rápido; o relatório será marcado como truncado.",
    )
    audit.add_argument(
        "--labels-only",
        action="store_true",
        help="Confere apenas rótulos/distribuição, sem abrir imagens nem verificar corrupção.",
    )
    audit.add_argument(
        "--no-hash",
        action="store_true",
        help="Abre imagens para validar corrupção, mas não calcula hashes de duplicatas.",
    )

    run = subparsers.add_parser("run", help="Executa a matriz configurada de uma ou todas as bases.")
    _add_common_config_arguments(run)
    _add_dataset_selector(run)
    run.add_argument("--dry-run", action="store_true", help="Cria/valida os jobs sem iniciar TensorFlow.")
    run.add_argument("--resume", action="store_true", help="Pula concluídas e retoma runs interrompidas compatíveis.")
    run.add_argument("--rerun-failed", action="store_true", help="Inclui runs com estado failed.")
    run.add_argument("--fail-fast", action="store_true", help="Interrompe a fila na primeira falha.")

    resume = subparsers.add_parser("resume", help="Atalho para run --resume.")
    _add_common_config_arguments(resume)
    _add_dataset_selector(resume)
    resume.add_argument("--rerun-failed", action="store_true")
    resume.add_argument("--fail-fast", action="store_true")

    report = subparsers.add_parser("report", help="Reconstrói relatório de um dataset já executado.")
    report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    report.add_argument("--dataset", choices=DATASET_ORDER, required=True)

    smoke = subparsers.add_parser("smoke", help="Executa validação curta em um subconjunto local.")
    _add_common_config_arguments(smoke)
    smoke.add_argument("--dataset", choices=DATASET_ORDER, required=True)
    smoke.add_argument("--examples-per-class", type=int, default=16)
    smoke.add_argument("--epochs", type=int, default=2)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            from .preflight import format_preflight, run_preflight

            report = run_preflight(
                output_root=args.output_root,
                data_paths=args.data_path,
                require_tensorflow=args.require_tensorflow,
            )
            print(format_preflight(report))
            return 0 if report.get("ok", False) else 1

        from . import runner

        if args.command == "audit":
            return runner.command_audit(args)
        if args.command == "run":
            return runner.command_run(args)
        if args.command == "resume":
            args.resume = True
            args.dry_run = False
            return runner.command_run(args)
        if args.command == "report":
            return runner.command_report(args)
        if args.command == "smoke":
            return runner.command_smoke(args)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Execute 'resume' para continuar.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"Comando não tratado: {args.command}")
