from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "outputs" / "training-analysis-2026-08-02"

MODE_LABELS = {
    "all_raw": "Sem balanceamento",
    "class_weight": "Pesos de classe",
    "oversample": "Oversampling",
    "undersample": "Undersampling",
}
DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "kmnist": "KMNIST",
    "emnist_balanced": "EMNIST Balanced",
    "cifar10": "CIFAR-10",
    "cifar100_coarse": "CIFAR-100 Coarse",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "fer2013": "FER2013",
}


def clean_number(value: object, digits: int = 6) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return round(value, digits) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def records(frame: pd.DataFrame, digits: int = 6) -> list[dict[str, object]]:
    return [
        {str(key): clean_number(value, digits) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def source(source_id: str, label: str, filename: str, description: str, metrics: list[str]) -> dict:
    path = ANALYSIS_DIR / filename
    wsl_path = "/mnt/c/source/repos/teste-modelos-tcc/outputs/training-analysis-2026-08-02/" + filename
    if path.suffix == ".json":
        query_code = f"SELECT * FROM read_json_auto('{wsl_path}')"
    else:
        query_code = f"SELECT * FROM read_csv_auto('{wsl_path}', header = true)"
    return {
        "id": source_id,
        "label": label,
        "path": str(path),
        "query": {
            "id": f"local-{source_id}",
            "description": description,
            "engine": "DuckDB",
            "language": "SQL",
            "executed_at": "2026-08-02T18:05:22.296524+00:00",
            "sql": query_code,
            "tables_used": [str(path)],
            "filters": [
                "36 runs canônicos: 9 datasets × 4 modos de balanceamento, seed 42, normalização z-score",
                "Tentativas interrompidas e datasets fora da suíte atual foram excluídos",
            ],
            "metric_definitions": metrics,
        },
    }


def main() -> None:
    overview = json.loads((ANALYSIS_DIR / "analysis_overview.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(ANALYSIS_DIR / "run_metrics.csv")
    epochs = pd.read_csv(ANALYSIS_DIR / "epoch_metrics_all.csv")
    classes = pd.read_csv(ANALYSIS_DIR / "per_class_metrics.csv")
    quality = pd.read_csv(ANALYSIS_DIR / "quality_checks.csv")
    comparability = pd.read_csv(ANALYSIS_DIR / "comparability_by_dataset.csv")

    runs["dataset_label"] = runs["dataset"].map(DATASET_LABELS)
    runs["mode_label"] = runs["balance_mode"].map(MODE_LABELS)
    runs["run_label"] = runs["dataset_label"] + " · " + runs["mode_label"]

    performance = runs[
        [
            "run_id",
            "dataset_label",
            "mode_label",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "delta_macro_f1_vs_all_raw",
            "best_epoch",
            "epochs_completed",
            "test_samples",
        ]
    ].rename(
        columns={
            "dataset_label": "dataset",
            "mode_label": "mode",
            "test_accuracy": "accuracy",
            "test_balanced_accuracy": "balanced_accuracy",
            "test_macro_f1": "macro_f1",
            "delta_macro_f1_vs_all_raw": "delta_vs_all_raw",
            "epochs_completed": "epochs",
        }
    )

    training = runs[
        [
            "run_id",
            "dataset_label",
            "mode_label",
            "epochs_completed",
            "best_epoch",
            "training_hours",
            "mean_epoch_seconds",
            "time_to_best_seconds",
            "weighted_train_examples_per_second",
            "total_examples_processed",
            "evaluation_seconds",
            "final_learning_rate",
        ]
    ].rename(
        columns={
            "dataset_label": "dataset",
            "mode_label": "mode",
            "epochs_completed": "epochs",
            "weighted_train_examples_per_second": "examples_per_second",
        }
    )
    training["time_to_best_hours"] = training.pop("time_to_best_seconds") / 3600

    hardware = runs[
        [
            "run_id",
            "run_label",
            "dataset_label",
            "mode_label",
            "gpu_util_mean_percent",
            "gpu_util_p95_percent",
            "gpu_memory_max_gib",
            "ram_mean_percent",
            "ram_max_percent",
            "process_rss_max_gib",
            "gpu_temperature_max_c",
            "cpu_mean_percent",
            "output_disk_free_min_gib",
            "weighted_train_examples_per_second",
            "training_hours",
        ]
    ].rename(
        columns={
            "dataset_label": "dataset",
            "mode_label": "mode",
            "weighted_train_examples_per_second": "examples_per_second",
        }
    )

    deltas = performance[performance["mode"] != MODE_LABELS["all_raw"]].copy()

    fer_curve = epochs[epochs["dataset"] == "fer2013"].copy()
    fer_curve["mode"] = fer_curve["balance_mode"].map(MODE_LABELS)
    fer_curve = fer_curve[
        ["run_id", "mode", "epoch", "val_macro_f1", "val_balanced_accuracy", "val_accuracy", "learning_rate"]
    ]

    class_extremes: list[dict[str, object]] = []
    for (run_id, dataset, balance_mode), group in classes.groupby(
        ["run_id", "dataset", "balance_mode"], sort=False
    ):
        worst = group.loc[group["f1"].idxmin()]
        best = group.loc[group["f1"].idxmax()]
        class_extremes.append(
            {
                "run_id": run_id,
                "dataset": DATASET_LABELS[dataset],
                "mode": MODE_LABELS[balance_mode],
                "worst_class": str(worst["class_name"]),
                "worst_class_f1": float(worst["f1"]),
                "worst_class_support": int(worst["support"]),
                "best_class": str(best["class_name"]),
                "best_class_f1": float(best["f1"]),
                "best_class_support": int(best["support"]),
            }
        )
    class_extremes_frame = pd.DataFrame(class_extremes)

    failed_checks = quality[~quality["passed"].astype(bool)].copy()
    quality_summary = (
        quality.assign(result=np.where(quality["passed"].astype(bool), "Aprovado", "Falhou"))
        .groupby(["severity_if_failed", "result"], dropna=False)
        .size()
        .reset_index(name="checks")
        .rename(columns={"severity_if_failed": "severity"})
    )
    comparability["dataset"] = comparability["dataset"].map(DATASET_LABELS)
    comparability = comparability.rename(
        columns={
            "within_dataset_comparable": "comparable",
        }
    )

    headline = pd.DataFrame(
        [
            {
                "completed_runs": overview["completed_runs"],
                "planned_runs": overview["canonical_runs"],
                "epochs": overview["epochs_recorded"],
                "training_hours": overview["training_hours"],
                "best_macro_f1": overview["best_overall_test_macro_f1"],
                "peak_ram_rate": overview["peak_ram_percent"] / 100,
                "peak_gpu_memory_gib": overview["peak_gpu_memory_gib"],
            }
        ]
    )

    report_sources = [
        source(
            "overview",
            "Resumo consolidado da auditoria",
            "analysis_overview.json",
            "Consolidação dos 36 runs, tempos e picos de hardware.",
            [
                "Horas de treinamento = soma de epoch_seconds de todos os epochs canônicos.",
                "Picos de hardware = máximos observados nas amostras de telemetria.",
            ],
        ),
        source(
            "runs",
            "Métricas consolidadas por run",
            "run_metrics.csv",
            "Uma linha por run com teste, tempo, throughput, hardware e proveniência.",
            [
                "Macro-F1 = média não ponderada do F1 por classe no conjunto de teste.",
                "Delta vs. sem balanceamento = macro-F1 do modo menos macro-F1 de all_raw no mesmo dataset.",
                "Throughput = exemplos efetivos de treino divididos pelo tempo de epoch, agregado pelo total.",
            ],
        ),
        source(
            "epochs",
            "Histórico completo por epoch",
            "epoch_metrics_all.csv",
            "Histórico de 2.486 epochs com métricas, taxa de aprendizado e duração.",
            [
                "Melhor epoch = epoch com o checkpoint best.keras selecionado pelo monitor configurado.",
                "Early stopping = interrupção após 10 epochs sem melhora do monitor.",
            ],
        ),
        source(
            "classes",
            "Métricas completas por classe",
            "per_class_metrics.csv",
            "Precisão, recall, F1 e suporte de todas as classes em todos os runs.",
            ["F1 por classe = média harmônica de precisão e recall para aquela classe."],
        ),
        source(
            "quality",
            "Controles de qualidade e comparabilidade",
            "quality_checks.csv",
            "396 verificações de arquivos, status, epochs, previsões, checkpoints, tempos e finitude.",
            ["Severidade crítica/alta indica falha que invalidaria métricas ou comparações."],
        ),
    ]

    cards = [
        {
            "id": "completion",
            "description": "Conclusão da suíte canônica",
            "dataset": "headline",
            "sourceId": "overview",
            "metrics": [
                {"label": "Runs concluídos", "field": "completed_runs", "format": "number"},
                {"label": "Planejados", "field": "planned_runs", "format": "number"},
            ],
        },
        {
            "id": "training_time",
            "description": "Custo total de treinamento",
            "dataset": "headline",
            "sourceId": "overview",
            "metrics": [{"label": "Horas", "field": "training_hours", "format": "number"}],
        },
        {
            "id": "epochs_recorded",
            "description": "Volume efetivamente registrado",
            "dataset": "headline",
            "sourceId": "overview",
            "metrics": [{"label": "Epochs", "field": "epochs", "format": "number"}],
        },
        {
            "id": "best_score",
            "description": "Melhor macro-F1 de teste",
            "dataset": "headline",
            "sourceId": "overview",
            "metrics": [{"label": "Macro-F1", "field": "best_macro_f1", "format": "percent"}],
        },
        {
            "id": "ram_peak",
            "description": "Maior uso de RAM observado",
            "dataset": "headline",
            "sourceId": "overview",
            "metrics": [{"label": "RAM", "field": "peak_ram_rate", "format": "percent"}],
        },
        {
            "id": "vram_peak",
            "description": "Maior alocação de VRAM observada",
            "dataset": "headline",
            "sourceId": "overview",
            "metrics": [{"label": "VRAM (GiB)", "field": "peak_gpu_memory_gib", "format": "number"}],
        },
    ]

    charts = [
        {
            "id": "macro_f1_by_run",
            "title": "Macro-F1 de teste por dataset e balanceamento",
            "subtitle": "A estratégia vencedora muda conforme o dataset; não há um modo universalmente superior.",
            "type": "bar",
            "dataset": "performance",
            "sourceId": "runs",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                "y": {"field": "macro_f1", "type": "quantitative", "label": "Macro-F1"},
                "color": {"field": "mode", "type": "nominal", "label": "Balanceamento"},
            },
            "options": {"orientation": "vertical", "grouping": "grouped", "legend": {"visible": True}},
            "layout": {"width": "full", "height": 430},
        },
        {
            "id": "delta_by_run",
            "title": "Variação de macro-F1 contra o baseline sem balanceamento",
            "subtitle": "Undersampling causou as maiores perdas no FER2013 e no GTSRB.",
            "type": "bar",
            "dataset": "deltas",
            "sourceId": "runs",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                "y": {"field": "delta_vs_all_raw", "type": "quantitative", "label": "Δ macro-F1"},
                "color": {"field": "mode", "type": "nominal", "label": "Balanceamento"},
            },
            "options": {"orientation": "vertical", "grouping": "grouped", "legend": {"visible": True}},
            "layout": {"width": "full", "height": 430},
        },
        {
            "id": "training_hours_by_run",
            "title": "Tempo de treinamento por dataset e balanceamento",
            "subtitle": "GTSRB e SVHN concentram a maior parte do custo computacional.",
            "type": "bar",
            "dataset": "training",
            "sourceId": "runs",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                "y": {"field": "training_hours", "type": "quantitative", "label": "Horas"},
                "color": {"field": "mode", "type": "nominal", "label": "Balanceamento"},
            },
            "options": {"orientation": "vertical", "grouping": "grouped", "legend": {"visible": True}},
            "layout": {"width": "full", "height": 430},
        },
        {
            "id": "hardware_scatter",
            "title": "Throughput e utilização média da GPU por run",
            "subtitle": "A GPU ficou subutilizada em média, sugerindo gargalo no pipeline ou na CPU em parte da suíte.",
            "type": "scatter",
            "dataset": "hardware",
            "sourceId": "runs",
            "encodings": {
                "x": {"field": "gpu_util_mean_percent", "type": "quantitative", "label": "GPU média (%)"},
                "y": {"field": "examples_per_second", "type": "quantitative", "label": "Exemplos/s"},
                "color": {"field": "dataset", "type": "nominal", "label": "Dataset"},
                "label": {"field": "run_label", "type": "nominal"},
            },
            "options": {"legend": {"visible": True}},
            "layout": {"width": "full", "height": 430},
        },
        {
            "id": "fer_curve",
            "title": "Val macro-F1 do FER2013 ao longo dos epochs",
            "subtitle": "O run com pesos de classe encerrou dez epochs após o melhor checkpoint, conforme a paciência configurada.",
            "type": "line",
            "dataset": "fer_curve",
            "sourceId": "epochs",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                "y": {"field": "val_macro_f1", "type": "quantitative", "label": "Val macro-F1"},
                "color": {"field": "mode", "type": "nominal", "label": "Balanceamento"},
            },
            "options": {"legend": {"visible": True}},
            "layout": {"width": "full", "height": 400},
        },
    ]

    tables = [
        {
            "id": "performance_table",
            "title": "Métricas de teste dos 36 runs",
            "subtitle": "Resultados no conjunto de teste e posição do melhor checkpoint.",
            "dataset": "performance",
            "sourceId": "runs",
            "defaultSort": {"field": "macro_f1", "direction": "desc"},
            "density": "compact",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"},
                {"field": "mode", "label": "Balanceamento", "type": "text"},
                {"field": "accuracy", "label": "Accuracy", "type": "number", "format": "percent"},
                {"field": "balanced_accuracy", "label": "Balanced acc.", "type": "number", "format": "percent"},
                {"field": "macro_f1", "label": "Macro-F1", "type": "number", "format": "percent"},
                {"field": "delta_vs_all_raw", "label": "Δ vs baseline", "type": "number", "format": "percent", "movement": True},
                {"field": "best_epoch", "label": "Melhor epoch", "type": "number", "format": "number"},
                {"field": "epochs", "label": "Epochs", "type": "number", "format": "number"},
                {"field": "test_samples", "label": "Amostras teste", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "training_table",
            "title": "Tempo e volume de treinamento dos 36 runs",
            "subtitle": "A duração é a soma de epoch_seconds; inclui trechos anteriores a retomadas.",
            "dataset": "training",
            "sourceId": "runs",
            "defaultSort": {"field": "training_hours", "direction": "desc"},
            "density": "compact",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"},
                {"field": "mode", "label": "Balanceamento", "type": "text"},
                {"field": "epochs", "label": "Epochs", "type": "number", "format": "number"},
                {"field": "best_epoch", "label": "Melhor", "type": "number", "format": "number"},
                {"field": "training_hours", "label": "Treino (h)", "type": "number", "format": "number"},
                {"field": "mean_epoch_seconds", "label": "Média/epoch (s)", "type": "number", "format": "number"},
                {"field": "time_to_best_hours", "label": "Até melhor (h)", "type": "number", "format": "number"},
                {"field": "examples_per_second", "label": "Exemplos/s", "type": "number", "format": "number"},
                {"field": "total_examples_processed", "label": "Exemplos processados", "type": "number", "format": "compact"},
                {"field": "evaluation_seconds", "label": "Avaliação (s)", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "hardware_table",
            "title": "Telemetria de hardware dos 36 runs",
            "subtitle": "Médias e picos observados; percentuais são relativos ao recurso do host.",
            "dataset": "hardware",
            "sourceId": "runs",
            "defaultSort": {"field": "gpu_memory_max_gib", "direction": "desc"},
            "density": "compact",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"},
                {"field": "mode", "label": "Balanceamento", "type": "text"},
                {"field": "gpu_util_mean_percent", "label": "GPU média (%)", "type": "number", "format": "number"},
                {"field": "gpu_util_p95_percent", "label": "GPU p95 (%)", "type": "number", "format": "number"},
                {"field": "gpu_memory_max_gib", "label": "VRAM máx. (GiB)", "type": "number", "format": "number"},
                {"field": "ram_mean_percent", "label": "RAM média (%)", "type": "number", "format": "number"},
                {"field": "ram_max_percent", "label": "RAM máx. (%)", "type": "number", "format": "number"},
                {"field": "process_rss_max_gib", "label": "RSS máx. (GiB)", "type": "number", "format": "number"},
                {"field": "gpu_temperature_max_c", "label": "GPU máx. (°C)", "type": "number", "format": "number"},
                {"field": "output_disk_free_min_gib", "label": "Disco livre mín. (GiB)", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "class_extremes_table",
            "title": "Classes de maior e menor F1 em cada run",
            "subtitle": "A tabela completa por classe está disponível no CSV de apoio.",
            "dataset": "class_extremes",
            "sourceId": "classes",
            "defaultSort": {"field": "worst_class_f1", "direction": "asc"},
            "density": "compact",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"},
                {"field": "mode", "label": "Balanceamento", "type": "text"},
                {"field": "worst_class", "label": "Pior classe", "type": "text"},
                {"field": "worst_class_f1", "label": "F1 pior", "type": "number", "format": "percent"},
                {"field": "worst_class_support", "label": "Suporte", "type": "number", "format": "number"},
                {"field": "best_class", "label": "Melhor classe", "type": "text"},
                {"field": "best_class_f1", "label": "F1 melhor", "type": "number", "format": "percent"},
                {"field": "best_class_support", "label": "Suporte", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "comparability_table",
            "title": "Comparabilidade interna por dataset",
            "subtitle": "Os quatro modos de cada dataset compartilham split, fonte e configuração relevante.",
            "dataset": "comparability",
            "sourceId": "quality",
            "defaultSort": {"field": "dataset", "direction": "asc"},
            "density": "compact",
            "columns": [
                {"field": "dataset", "label": "Dataset", "type": "text"},
                {"field": "distinct_split_fingerprints", "label": "Splits distintos", "type": "number", "format": "number"},
                {"field": "distinct_source_samples", "label": "Fontes distintas", "type": "number", "format": "number"},
                {"field": "distinct_batch_sizes", "label": "Batches distintos", "type": "number", "format": "number"},
                {"field": "distinct_tensorflow_versions", "label": "Versões TF", "type": "number", "format": "number"},
                {"field": "distinct_shuffle_caps", "label": "Limites shuffle", "type": "number", "format": "number"},
                {"field": "comparable", "label": "Comparável", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# Análise técnica dos 36 treinamentos do TCC"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "overview",
            "body": (
                "## Resumo técnico\n\n"
                "A suíte canônica terminou: **36 de 36 runs** estão com status `completed`, reunindo "
                "**2.486 epochs** e **88,86 horas** de treinamento efetivo. Não é necessário executar "
                "comando de retomada. Os avisos exibidos ao fim do FER2013 não invalidaram o run: o "
                "epoch 46 foi registrado integralmente, o early stopping ocorreu na janela prevista e "
                "checkpoint, avaliação e previsões finais estão presentes."
            ),
        },
        {
            "id": "metrics",
            "type": "metric-strip",
            "cardIds": ["completion", "training_time", "epochs_recorded", "best_score", "ram_peak", "vram_peak"],
        },
        {
            "id": "performance_text",
            "type": "markdown",
            "sourceId": "runs",
            "body": (
                "## Desempenho de teste\n\n"
                "O maior macro-F1 foi **99,6749%**, no **GTSRB com oversampling**. A melhor estratégia "
                "dependeu do dataset: undersampling venceu CIFAR-10, EMNIST Balanced e KMNIST; "
                "oversampling venceu CIFAR-100 Coarse, FER2013 e GTSRB; pesos de classe venceu MNIST; "
                "e o baseline sem balanceamento venceu Fashion-MNIST e SVHN. Em vários casos a diferença "
                "foi pequena, portanto a escolha deve considerar também custo e estabilidade por classe."
            ),
        },
        {"id": "performance_chart_block", "type": "chart", "chartId": "macro_f1_by_run"},
        {
            "id": "balance_text",
            "type": "markdown",
            "sourceId": "runs",
            "body": (
                "## Efeito do balanceamento\n\n"
                "Na média dos nove datasets, oversampling ficou praticamente neutro contra o baseline "
                "(**−0,009 ponto percentual** de macro-F1), enquanto undersampling perdeu **4,00 pontos "
                "percentuais** em média. Essa perda é concentrada no FER2013 (**−19,28 p.p.**) e no GTSRB "
                "(**−13,69 p.p.**), onde descartar exemplos prejudicou classes difíceis. Pesos de classe "
                "foi exatamente equivalente ao baseline nos datasets já balanceados e caiu nos mais "
                "desbalanceados, especialmente no FER2013."
            ),
        },
        {"id": "delta_chart_block", "type": "chart", "chartId": "delta_by_run"},
        {
            "id": "time_text",
            "type": "markdown",
            "sourceId": "runs",
            "body": (
                "## Tempo e eficiência\n\n"
                "O run mais demorado foi **GTSRB com oversampling**, com **9,90 h**; o mais curto foi "
                "FER2013 com undersampling, com **4,21 min**, porque o conjunto efetivo foi drasticamente "
                "reduzido. Os 36 runs somaram **88,86 h**. GTSRB e SVHN dominam o custo, enquanto os "
                "datasets 28×28 e 64×64 têm epochs muito mais curtos."
            ),
        },
        {"id": "time_chart_block", "type": "chart", "chartId": "training_hours_by_run"},
        {
            "id": "hardware_text",
            "type": "markdown",
            "sourceId": "runs",
            "body": (
                "## Uso de hardware\n\n"
                "A telemetria da **NVIDIA GeForce RTX 3050 de 8 GB** registrou pico de **7,64 GiB de "
                "VRAM**, RAM do sistema em **58,9%**, RSS do processo em **9,14 GiB** e GPU a no máximo "
                "**61 °C**. A utilização média ponderada da GPU foi apenas **25,0%**, sinal de que parte "
                "do tempo foi limitada pelo carregamento/preprocessamento de dados ou pela CPU. Não há "
                "evidência de OOM ou disco cheio nos runs concluídos. Os 12 runs finais (SVHN, GTSRB e "
                "FER2013) estão no disco E:, enquanto os 24 anteriores permanecem no C:; esta análise "
                "consolida os dois locais."
            ),
        },
        {"id": "hardware_chart_block", "type": "chart", "chartId": "hardware_scatter"},
        {
            "id": "run_tables_text",
            "type": "markdown",
            "sourceId": "runs",
            "body": (
                "## Detalhamento completo por run\n\n"
                "As três tabelas seguintes cobrem os 36 runs: métricas de teste, epochs e tempo, e "
                "telemetria de hardware. Para runs retomados, o tempo canônico é a soma dos registros "
                "por epoch, evitando subcontagem no resumo da tentativa final."
            ),
        },
        {"id": "performance_table_block", "type": "table", "tableId": "performance_table"},
        {"id": "training_table_block", "type": "table", "tableId": "training_table"},
        {"id": "hardware_table_block", "type": "table", "tableId": "hardware_table"},
        {
            "id": "fer_warning_text",
            "type": "markdown",
            "sourceId": "epochs",
            "body": (
                "## Interpretação dos avisos finais do FER2013\n\n"
                "No run `fer2013__zscore__class_weight__seed-42`, o melhor checkpoint ocorreu no epoch "
                "**36** e o treinamento encerrou no **46**, exatamente após os **10 epochs** de paciência "
                "do early stopping. O log mostra 148/148 batches no epoch 46. A mensagem `input ran out "
                "of data` apareceu durante o encerramento do iterador; `use_unbounded_threadpool` foi "
                "explicitamente ignorado pelo TensorFlow e os cancelamentos de rendezvous são compatíveis "
                "com a desmontagem do pipeline. Como os artefatos de teste foram gravados e o status final "
                "é `completed`, não há razão técnica para repetir esse run."
            ),
        },
        {"id": "fer_curve_block", "type": "chart", "chartId": "fer_curve"},
        {
            "id": "class_text",
            "type": "markdown",
            "sourceId": "classes",
            "body": (
                "## Comportamento por classe\n\n"
                "As métricas macro escondem diferenças importantes. No FER2013, `fear` permanece entre "
                "as classes mais difíceis; no CIFAR-10, `cat`; e no SVHN, os dígitos 8 e 6. O "
                "undersampling do GTSRB derrubou especialmente a classe 0. A tabela resume os extremos; "
                "o CSV de apoio contém precisão, recall, F1 e suporte de todas as classes."
            ),
        },
        {"id": "class_table_block", "type": "table", "tableId": "class_extremes_table"},
        {
            "id": "scope_text",
            "type": "markdown",
            "sourceId": "quality",
            "body": (
                "## Escopo, método e qualidade dos dados\n\n"
                "Foram incluídos somente os 36 runs canônicos da suíte atual: nove datasets, quatro "
                "modos, normalização z-score e seed 42. Tentativas interrompidas antigas e o CINIC-10, "
                "que não pertence mais à suíte, foram excluídos. A auditoria executou **396 verificações** "
                "de arquivos, status, sequência de epochs, previsões, suporte, checkpoints, tempos e "
                "valores finitos. Não houve falha crítica ou alta. A única falha média foi a subcontagem "
                "do resumo de tempo do SVHN com pesos de classe; o histórico completo por epoch resolve "
                "essa diferença sem afetar o modelo ou suas métricas."
            ),
        },
        {"id": "comparability_table_block", "type": "table", "tableId": "comparability_table"},
        {
            "id": "limitations_text",
            "type": "markdown",
            "sourceId": "runs",
            "body": (
                "## Limitações e robustez\n\n"
                "Cada configuração foi executada com uma única seed (**42**), então diferenças pequenas "
                "devem ser tratadas como descritivas, não como evidência estatística de superioridade. "
                "Os seis primeiros datasets não registraram explicitamente o limite de shuffle em MiB; "
                "SVHN, GTSRB e FER2013 usaram limite de 1.024 MiB. Isso não quebra as comparações entre "
                "os quatro modos do mesmo dataset, mas recomenda cautela ao comparar eficiência absoluta "
                "entre datasets de fases distintas."
            ),
        },
        {
            "id": "recommendations_text",
            "type": "markdown",
            "body": (
                "## Recomendações\n\n"
                "1. Use **sem balanceamento** como baseline padrão e escolha outra estratégia apenas quando "
                "o ganho por classe ou macro-F1 justificar o custo.\n"
                "2. Prefira **oversampling** a undersampling nos datasets fortemente desbalanceados; para "
                "FER2013, o ganho global é pequeno, mas merece análise específica da classe `fear`.\n"
                "3. Evite undersampling no GTSRB e FER2013.\n"
                "4. Para conclusões do TCC, repita as configurações candidatas com pelo menos 3–5 seeds e "
                "relate média e desvio-padrão/intervalo de confiança.\n"
                "5. Para reduzir tempo, faça profiling do pipeline `tf.data`: a GPU média de 25% mostra "
                "margem para prefetch, paralelismo e cache mais eficientes."
            ),
        },
        {
            "id": "questions_text",
            "type": "markdown",
            "body": (
                "## Próximas perguntas\n\n"
                "Vale confirmar se os ganhos de CIFAR-100, GTSRB e KMNIST persistem em outras seeds; medir "
                "o custo energético por run; e testar se focal loss ou augmentação direcionada melhora as "
                "classes raras do FER2013 sem o custo do oversampling integral."
            ),
        },
    ]

    # The hosted artifact validator requires widget provenance inline even when
    # sourceId is also present. Markdown blocks keep sourceId because file-backed
    # narrative provenance does not require an inline query object.
    source_by_id = {item["id"]: item for item in report_sources}
    for widget in [*cards, *charts, *tables]:
        source_id = widget["sourceId"]
        widget["source"] = source_by_id[source_id]

    snapshot = {
        "version": 1,
        "generatedAt": overview["generated_at"],
        "status": "ready",
        "datasets": {
            "headline": records(headline),
            "performance": records(performance),
            "deltas": records(deltas),
            "training": records(training),
            "hardware": records(hardware),
            "fer_curve": records(fer_curve),
            "class_extremes": records(class_extremes_frame),
            "quality_summary": records(quality_summary),
            "failed_checks": records(failed_checks),
            "comparability": records(comparability),
        },
    }
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Análise técnica dos 36 treinamentos do TCC",
        "description": "Auditoria consolidada de desempenho, tempo, hardware, qualidade e comparabilidade.",
        "generatedAt": overview["generated_at"],
        "sources": report_sources,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
    }
    payload = {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": report_sources}
    output_path = ANALYSIS_DIR / "training_report_artifact.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    size_mib = output_path.stat().st_size / (1024**2)
    print(f"Wrote {output_path}")
    print(f"Snapshot datasets: {len(snapshot['datasets'])}; payload size: {size_mib:.3f} MiB")


if __name__ == "__main__":
    main()
