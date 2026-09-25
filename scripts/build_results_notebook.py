"""Build the reproducible Windows benchmark results notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "benchmark_results_windows.ipynb"


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip().splitlines(True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(True),
    }


cells = [
    markdown_cell(
        """
        # Relatório dos experimentos de treinamento e quantização

        **Projeto:** TCC Dataset Benchmark  
        **Ambiente:** Windows + Ubuntu-22.04/WSL2  
        **Fonte principal:** `outputs/controlled-augmentation05-batch-activation`

        Este notebook lê os artefatos produzidos localmente, reconstrói os resultados das etapas de batch, quantização e ativações, e apresenta gráficos editáveis. Os valores não são digitados manualmente: são carregados dos CSV, JSON e YAML gerados pelo pipeline.
        """
    ),
    markdown_cell(
        """
        ## tl;dr

        - A etapa de quantização contém **27/27 testes concluídos**: FP32, FP16 e INT8-PTQ para 9 datasets.
        - Não há falhas registradas no status final da etapa.
        - A quantização foi executada com `hidden_activation: swish`; portanto, este relatório não representa uma quantização treinada com ReLU.
        - Os resultados incluem acurácia, macro-F1, tamanho do modelo LiteRT, latência de inferência, tempo por época e telemetria de hardware.
        - O notebook também documenta os testes de batch e de ativações já existentes no mesmo output-root.
        """
    ),
    markdown_cell(
        """
        ## Contexto e métodos

        O pipeline foi executado de forma serial e retomável. A etapa de batch compara `32`, `64`, `128` e `256`; a etapa de quantização treina/exporta as variantes FP32 e FP16 e aplica INT8 pós-treinamento (INT8-PTQ) usando o vencedor de batch de cada dataset; a etapa de ativações compara ReLU, Sigmoid e Softmax.

        ### Assumptions and caveats

        - Os caminhos absolutos gravados nos relatórios apontam para o mesmo workspace montado no WSL. O notebook normaliza esses caminhos para `REPO_ROOT` e não depende deles para localizar os arquivos.
        - INT8-PTQ é derivado do modelo FP32; não é um novo treinamento quantizado.
        - As métricas de hardware são agregadas a partir da telemetria persistida. Latência LiteRT foi medida no runtime CPU, conforme os artefatos (`inference_vram: not_applicable_cpu_runtime`).
        """
    ),
    code_cell(
        """
        # 1. Imports, parâmetros e estilo visual
        from pathlib import Path
        import json
        import math

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import yaml

        try:
            from IPython.display import display, Markdown
        except ImportError:  # permite executar as células também como script simples
            display = print
            Markdown = lambda value: value

        def locate_repo_root():
            candidates = [Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parents[1] if "__file__" in globals() else None]
            candidates.append(Path("/mnt/c/Users/matheus.sduda/repos/teste-modelos-tcc/teste-modelos-tcc"))
            for candidate in candidates:
                if candidate is not None and (candidate / "outputs" / "controlled-augmentation05-batch-activation").exists():
                    return candidate.resolve()
            raise FileNotFoundError("Não foi possível localizar a raiz do repositório.")

        REPO_ROOT = locate_repo_root()
        OUTPUT_ROOT = REPO_ROOT / "outputs" / "controlled-augmentation05-batch-activation"
        REPORTS_ROOT = OUTPUT_ROOT / "reports"
        QUANT_ROOT = OUTPUT_ROOT / "quantization"
        GENERATED_ROOT = OUTPUT_ROOT / "generated-suites"
        FIGURES_ROOT = REPO_ROOT / "notebooks" / "figures"
        FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

        COLORS = {"fp32": "#4C78A8", "fp16": "#F58518", "int8_ptq": "#54A24B", "relu": "#4C78A8", "sigmoid": "#F58518", "softmax": "#54A24B"}
        plt.rcParams.update({"figure.figsize": (11, 5.5), "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.25})

        print(f"REPO_ROOT: {REPO_ROOT}")
        print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
        """
    ),
    markdown_cell("### 2. Carregamento das fontes locais"),
    code_cell(
        """
        def read_json(path, default=None):
            path = Path(path)
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))

        def read_csv(name):
            path = REPORTS_ROOT / name
            return pd.read_csv(path) if path.exists() else pd.DataFrame()

        pipeline_status = read_json(OUTPUT_ROOT / "pipeline-status.json", {})
        batch_df = read_csv("batch_comparison.csv")
        quant_df = read_csv("quantization_comparison.csv")
        activation_df = read_csv("activation_comparison.csv")

        for frame in (batch_df, quant_df, activation_df):
            for column in ("accuracy", "macro_f1", "mean_epoch_seconds", "median_batch_latency_ms", "serialized_litert_bytes", "serialized_model_bytes"):
                if column in frame.columns:
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")

        quant_df["accuracy_pct"] = quant_df["accuracy"] * 100
        quant_df["macro_f1_pct"] = quant_df["macro_f1"] * 100
        quant_df["serialized_litert_kb"] = quant_df["serialized_litert_bytes"] / 1024
        quant_df["serialized_model_mb"] = quant_df["serialized_model_bytes"] / (1024 ** 2)
        batch_df["accuracy_pct"] = batch_df["accuracy"] * 100
        batch_df["macro_f1_pct"] = batch_df["macro_f1"] * 100
        activation_df["accuracy_pct"] = activation_df["accuracy"] * 100
        activation_df["macro_f1_pct"] = activation_df["macro_f1"] * 100

        print("Pipeline status:", pipeline_status.get("status"))
        print("Batch rows:", len(batch_df), "Quantization rows:", len(quant_df), "Activation rows:", len(activation_df))
        """
    ),
    markdown_cell("### 3. Configuração efetivamente usada"),
    code_cell(
        """
        config_rows = []
        quant_suite_root = GENERATED_ROOT / "quantization"
        for suite_path in sorted(quant_suite_root.glob("*.yaml")):
            payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
            suite = payload["suite"]
            training = suite["training"]
            dataset, variant = suite_path.stem.rsplit("-", 1)
            config_rows.append({
                "dataset": dataset,
                "variant": variant,
                "seed": suite["seeds"][0],
                "normalization": suite["normalizations"][0],
                "balance_mode": suite["balance_modes"][0],
                "batch_size": training["batch_size"],
                "dtype_policy": training["dtype_policy"],
                "hidden_activation": training["hidden_activation"],
                "max_epochs": training["max_epochs"],
                "learning_rate": training["learning_rate"],
                "default_image_size": training["default_image_size"],
                "gtsrb_image_size": training["image_size_overrides"].get("gtsrb"),
                "early_stopping_patience": training["early_stopping_patience"],
                "augmentation": json.dumps(training["augmentation"], ensure_ascii=False, sort_keys=True),
                "source_suite": str(suite_path.relative_to(REPO_ROOT)),
            })

        config_df = pd.DataFrame(config_rows).sort_values(["dataset", "variant"]).reset_index(drop=True)
        display(config_df[["dataset", "variant", "batch_size", "dtype_policy", "hidden_activation", "max_epochs", "learning_rate", "default_image_size", "gtsrb_image_size"]])

        config_summary = {
            "seeds": sorted(config_df["seed"].unique().tolist()),
            "normalization": sorted(config_df["normalization"].unique().tolist()),
            "balance_mode": sorted(config_df["balance_mode"].unique().tolist()),
            "hidden_activations": sorted(config_df["hidden_activation"].unique().tolist()),
            "dtype_policies": sorted(config_df["dtype_policy"].unique().tolist()),
            "max_epochs": sorted(config_df["max_epochs"].unique().tolist()),
            "learning_rates": sorted(config_df["learning_rate"].unique().tolist()),
            "augmentation": json.loads(config_df.iloc[0]["augmentation"]),
        }
        display(pd.DataFrame({"parameter": list(config_summary), "value": [str(value) for value in config_summary.values()]}))
        """
    ),
    code_cell(
        """
        # Hardware e ambiente: preflight + resumo de telemetria persistida
        preflight_candidates = list(QUANT_ROOT.glob("*/fp32/preflight.json"))
        preflight = read_json(preflight_candidates[0], {}) if preflight_candidates else {}
        hardware = preflight.get("nvidia_smi", {}).get("gpus", [{}])[0]
        platform_info = preflight.get("platform", {})
        tf_info = preflight.get("tensorflow", {})

        environment_table = pd.DataFrame([
            {"item": "GPU", "value": hardware.get("name", "não registrado")},
            {"item": "VRAM", "value": f"{hardware.get('memory_total_mib', math.nan):,.0f} MiB"},
            {"item": "CPU lógico", "value": preflight.get("cpu", {}).get("logical_cores")},
            {"item": "RAM total", "value": f"{preflight.get('cpu', {}).get('memory_total_bytes', 0) / (1024**3):.2f} GiB"},
            {"item": "Sistema", "value": f"{platform_info.get('system')} / {platform_info.get('release')}"},
            {"item": "Python", "value": platform_info.get("python")},
            {"item": "TensorFlow", "value": preflight.get("packages", {}).get("tensorflow")},
            {"item": "CUDA", "value": tf_info.get("build_info", {}).get("cuda_version")},
            {"item": "cuDNN", "value": tf_info.get("build_info", {}).get("cudnn_version")},
        ])
        display(environment_table)

        def summary_for(dataset, variant):
            summary_path = QUANT_ROOT / dataset / variant / dataset / "summary.json"
            payload = read_json(summary_path, {})
            runs = payload.get("runs", [])
            return runs[0] if runs else {}

        hardware_rows = []
        for _, row in quant_df[quant_df["variant"].isin(["fp32", "fp16"])].iterrows():
            summary = summary_for(row["dataset"], row["variant"])
            hardware_rows.append({
                "dataset": row["dataset"], "variant": row["variant"],
                "mean_epoch_seconds": summary.get("mean_epoch_seconds", row.get("mean_epoch_seconds")),
                "training_total_seconds": summary.get("training_total_seconds"),
                "peak_gpu_utilization_pct": summary.get("peak_gpu_utilization_percent"),
                "peak_gpu_memory_mb": summary.get("peak_gpu_memory_used_bytes", np.nan) / (1024**2),
                "peak_ram_used_gb": summary.get("peak_ram_used_bytes", np.nan) / (1024**3),
                "telemetry_samples": summary.get("telemetry_samples"),
            })
        hardware_df = pd.DataFrame(hardware_rows)
        display(hardware_df.sort_values(["dataset", "variant"]).head(18))
        """
    ),
    markdown_cell("## Resultados da quantização"),
    code_cell(
        """
        quant_view = quant_df[["dataset", "variant", "accuracy_pct", "macro_f1_pct", "mean_epoch_seconds", "median_batch_latency_ms", "serialized_litert_kb", "serialized_model_mb", "status"]].copy()
        quant_view = quant_view.sort_values(["dataset", "variant"])
        display(quant_view.round({"accuracy_pct": 2, "macro_f1_pct": 2, "mean_epoch_seconds": 2, "median_batch_latency_ms": 2, "serialized_litert_kb": 1, "serialized_model_mb": 2}))

        quant_winners = quant_df.loc[quant_df.groupby("dataset")["macro_f1"].idxmax(), ["dataset", "variant", "accuracy_pct", "macro_f1_pct", "serialized_litert_kb", "median_batch_latency_ms"]].sort_values("dataset")
        display(Markdown("**Melhor macro-F1 por dataset:**"))
        display(quant_winners.round({"accuracy_pct": 2, "macro_f1_pct": 2, "serialized_litert_kb": 1, "median_batch_latency_ms": 2}))
        """
    ),
    code_cell(
        """
        # Acurácia e macro-F1 por dataset e variante
        datasets = quant_df["dataset"].drop_duplicates().tolist()
        x = np.arange(len(datasets))
        width = 0.25
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
        for index, variant in enumerate(["fp32", "fp16", "int8_ptq"]):
            current = quant_df[quant_df["variant"] == variant].set_index("dataset").reindex(datasets)
            axes[0].bar(x + (index - 1) * width, current["accuracy_pct"], width, label=variant, color=COLORS[variant])
            axes[1].bar(x + (index - 1) * width, current["macro_f1_pct"], width, label=variant, color=COLORS[variant])
        for ax, title, ylabel in zip(axes, ["Acurácia por variante", "Macro-F1 por variante"], ["Acurácia (%)", "Macro-F1 (%)"]):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=45, ha="right")
            ax.legend(title="Variante")
        fig.suptitle("Comparação de qualidade entre FP32, FP16 e INT8-PTQ")
        plt.tight_layout()
        fig.savefig(FIGURES_ROOT / "quantization_quality.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown_cell("**Figura salva:** `notebooks/figures/quantization_quality.png`\n\n![Qualidade por variante](figures/quantization_quality.png)"),
    code_cell(
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for variant in ["fp32", "fp16", "int8_ptq"]:
            current = quant_df[quant_df["variant"] == variant]
            axes[0].scatter(current["serialized_litert_kb"], current["median_batch_latency_ms"], s=65, alpha=0.85, label=variant, color=COLORS[variant])
            axes[1].bar(variant, current["serialized_litert_kb"].mean(), color=COLORS[variant], label=variant)
        for _, row in quant_df.iterrows():
            axes[0].annotate(row["dataset"], (row["serialized_litert_kb"], row["median_batch_latency_ms"]), fontsize=7, alpha=0.7)
        axes[0].set_title("Tamanho LiteRT versus latência")
        axes[0].set_xlabel("Modelo LiteRT (KiB)")
        axes[0].set_ylabel("Latência mediana por batch (ms)")
        axes[0].legend()
        axes[1].set_title("Tamanho LiteRT médio por variante")
        axes[1].set_ylabel("Tamanho médio (KiB)")
        plt.tight_layout()
        fig.savefig(FIGURES_ROOT / "quantization_size_latency.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown_cell("**Figura salva:** `notebooks/figures/quantization_size_latency.png`\n\n![Tamanho e latência](figures/quantization_size_latency.png)"),
    markdown_cell("## Resultados do teste de batch"),
    code_cell(
        """
        batch_view = batch_df[["dataset", "batch_size", "accuracy_pct", "macro_f1_pct", "mean_epoch_seconds", "status"]].sort_values(["dataset", "batch_size"])
        display(batch_view.round({"accuracy_pct": 2, "macro_f1_pct": 2, "mean_epoch_seconds": 2}))

        batch_winner_payload = pipeline_status.get("stages", {}).get("batch", {}).get("winners", {})
        batch_winners = pd.DataFrame([{"dataset": dataset, **winner} for dataset, winner in batch_winner_payload.items()])
        if not batch_winners.empty:
            display(Markdown("**Vencedores registrados pelo pipeline:**"))
            display(batch_winners[["dataset", "batch_size", "accuracy", "macro_f1", "mean_epoch_seconds"]].round({"accuracy": 4, "macro_f1": 4, "mean_epoch_seconds": 2}))
        """
    ),
    code_cell(
        """
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for dataset in sorted(batch_df["dataset"].unique()):
            current = batch_df[batch_df["dataset"] == dataset].sort_values("batch_size")
            ax.plot(current["batch_size"], current["macro_f1_pct"], marker="o", linewidth=1.5, label=dataset)
        ax.set_title("Macro-F1 em função do batch size")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_xticks(sorted(batch_df["batch_size"].unique()))
        ax.legend(ncol=3, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(FIGURES_ROOT / "batch_macro_f1.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown_cell("**Figura salva:** `notebooks/figures/batch_macro_f1.png`\n\n![Macro-F1 por batch](figures/batch_macro_f1.png)"),
    markdown_cell("## Resultados do teste de ativações"),
    code_cell(
        """
        activation_view = activation_df[["dataset", "hidden_activation", "accuracy_pct", "macro_f1_pct", "selected_quantization", "median_batch_latency_ms", "serialized_litert_bytes", "status"]].sort_values(["dataset", "hidden_activation"])
        display(activation_view.round({"accuracy_pct": 2, "macro_f1_pct": 2, "median_batch_latency_ms": 2, "serialized_litert_bytes": 0}))

        activation_winner_payload = pipeline_status.get("stages", {}).get("activation", {}).get("winners", {})
        activation_winners = pd.DataFrame([{"dataset": dataset, **winner} for dataset, winner in activation_winner_payload.items()])
        if not activation_winners.empty:
            display(Markdown("**Vencedores registrados pelo pipeline:**"))
            display(activation_winners[["dataset", "hidden_activation", "accuracy", "macro_f1", "selected_quantization"]].round({"accuracy": 4, "macro_f1": 4}))
        """
    ),
    code_cell(
        """
        activation_pivot = activation_df.pivot(index="dataset", columns="hidden_activation", values="macro_f1_pct")
        fig, ax = plt.subplots(figsize=(11, 6))
        activation_pivot[["relu", "sigmoid", "softmax"]].plot(kind="bar", ax=ax, color=[COLORS["relu"], COLORS["sigmoid"], COLORS["softmax"]])
        ax.set_title("Macro-F1 por ativação")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Macro-F1 (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Ativação")
        plt.tight_layout()
        fig.savefig(FIGURES_ROOT / "activation_macro_f1.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown_cell("**Figura salva:** `notebooks/figures/activation_macro_f1.png`\n\n![Macro-F1 por ativação](figures/activation_macro_f1.png)"),
    markdown_cell("## Uso de hardware e tempo"),
    code_cell(
        """
        hardware_agg = hardware_df.groupby("variant", as_index=False).agg({
            "mean_epoch_seconds": "mean",
            "training_total_seconds": "mean",
            "peak_gpu_utilization_pct": "mean",
            "peak_gpu_memory_mb": "mean",
            "peak_ram_used_gb": "mean",
        })
        display(hardware_agg.round({"mean_epoch_seconds": 2, "training_total_seconds": 0, "peak_gpu_utilization_pct": 1, "peak_gpu_memory_mb": 1, "peak_ram_used_gb": 2}))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].bar(hardware_agg["variant"], hardware_agg["mean_epoch_seconds"], color=[COLORS[v] for v in hardware_agg["variant"]])
        axes[0].set_title("Tempo médio por época")
        axes[0].set_ylabel("Segundos")
        axes[1].bar(hardware_agg["variant"], hardware_agg["peak_gpu_memory_mb"], color=[COLORS[v] for v in hardware_agg["variant"]])
        axes[1].set_title("VRAM de pico média — treino")
        axes[1].set_ylabel("MiB")
        plt.tight_layout()
        fig.savefig(FIGURES_ROOT / "hardware_summary.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    markdown_cell("**Figura salva:** `notebooks/figures/hardware_summary.png`\n\n![Resumo de hardware](figures/hardware_summary.png)"),
    markdown_cell("## Arquitetura registrada e rastreabilidade"),
    code_cell(
        """
        model_summary_path = QUANT_ROOT / "mnist" / "fp16" / "mnist" / "runs" / "mnist__unit_interval__all_raw__seed-42" / "logs" / "model_summary.txt"
        if model_summary_path.exists():
            print(model_summary_path.relative_to(REPO_ROOT))
            print(model_summary_path.read_text(encoding="utf-8", errors="replace")[:12000])
        else:
            print("model_summary.txt não encontrado")

        source_manifest = {
            "pipeline_status": str((OUTPUT_ROOT / "pipeline-status.json").relative_to(REPO_ROOT)),
            "batch_report": str((REPORTS_ROOT / "batch_comparison.csv").relative_to(REPO_ROOT)),
            "quantization_report": str((REPORTS_ROOT / "quantization_comparison.csv").relative_to(REPO_ROOT)),
            "activation_report": str((REPORTS_ROOT / "activation_comparison.csv").relative_to(REPO_ROOT)),
            "quantization_suites": str(quant_suite_root.relative_to(REPO_ROOT)),
        }
        display(pd.DataFrame({"artifact": list(source_manifest), "relative_path": list(source_manifest.values())}))
        """
    ),
    markdown_cell(
        """
        ## Conclusões

        1. O conjunto completo de quantização possui 27 resultados finalizados, incluindo INT8-PTQ para todos os nove datasets.
        2. A comparação de tamanho e latência deve ser lida junto com macro-F1/acurácia: redução de armazenamento não é, sozinha, critério suficiente para escolher a variante.
        3. A configuração de quantização usa `Swish`. Para responder à hipótese de ReLU, seria necessário gerar uma nova suíte de quantização com `hidden_activation: relu` e comparar os artefatos separadamente.
        4. Os gráficos de hardware são médias agregadas; os arquivos de telemetria por execução continuam sendo a fonte para auditoria detalhada.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10", "mimetype": "text/x-python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook written to {NOTEBOOK_PATH}")
