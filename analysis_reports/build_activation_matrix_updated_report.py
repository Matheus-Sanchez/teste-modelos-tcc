"""Build the updated activation investigation report and audit companion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "analysis_reports" / "investigacao_ativacoes_atualizada.json"
OUTPUT_ARTIFACT = ROOT / "analysis_reports" / "investigacao_ativacoes_atualizada.artifact.json"
OUTPUT_MARKDOWN = ROOT / "analysis_reports" / "relatorio_investigacao_ativacoes_atualizada.md"
OUTPUT_SNAPSHOT = ROOT / "analysis_reports" / "investigacao_ativacoes_atualizada_snapshot.sql"

ACTIVATION_LABELS = {"relu": "ReLU", "sigmoid": "Sigmoid", "softmax": "Softmax"}
DATASET_LABELS = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "kmnist": "KMNIST"}


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%".replace(".", ",")


def num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def sci(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}e}".replace(".", ",")


def pp(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):+.{digits}f} pp".replace(".", ",")


def source_catalog(generated_at: str) -> list[dict[str, Any]]:
    local = [
        ("analysis_script", "Script reprodutível da investigação atualizada", "analysis_reports/investigate_activation_matrix_updated.py", "Reenumera a matriz, recalcula concentração/variação e sonda camadas intermediárias."),
        ("model_code", "Implementação da arquitetura e da loss", "src/tcc_benchmark/model.py", "Verificação de GroupNormalization, ativações ocultas repetidas, camada final linear e from_logits=True."),
        ("data_code", "Implementação do protocolo de dados", "src/tcc_benchmark/data.py", "Verificação da semântica de extra_fraction e da construção determinística dos datasets."),
        ("metrics_code", "Métricas do benchmark", "src/tcc_benchmark/metrics.py", "Implementação das métricas independentes de classificação."),
        ("benchmark_docs", "Documentação do protocolo controlado", "docs/BENCHMARK_CONTROLADO.md", "Definições e controles do benchmark de ativações."),
        ("technical_guide", "Guia técnico do projeto", "docs/GUIA_TECNICO.md", "Descrição técnica de hidden_activation, logits e treinamento."),
    ]
    sources: list[dict[str, Any]] = [
        {
            "id": "run_artifacts",
            "label": "Runs de ativação: lote anterior e lote novo",
            "path": "analysis_reports/investigacao_ativacoes_atualizada_snapshot.sql",
            "query": {
                "language": "Python/JSON/CSV/NumPy",
                "description": "Manifestos, históricos, test_metrics.json, predictions.csv e logits.npy das execuções completas; o snapshot SQL registra as linhas-chave usadas no relatório.",
                "executed_at": generated_at,
                "metric_definitions": [
                    "Acurácia, balanced accuracy e F1 macro são as métricas independentes salvas em classification.",
                    "Participação dominante é a fração atribuída à classe prevista mais frequente.",
                    "Variação dos logits é o desvio-padrão médio por componente entre linhas do teste.",
                    "Variação intermediária usa 256 exemplos do primeiro lote de teste e best.keras.",
                ],
            },
        }
    ]
    for source_id, label, path, description in local:
        sources.append({"id": source_id, "label": label, "path": path, "query": {"language": "Python/Markdown", "description": description, "executed_at": generated_at}})
    external = [
        ("keras_softmax", "Keras — Softmax layer", "https://keras.io/api/layers/activation_layers/softmax/", "Definição do softmax, eixo de normalização e soma unitária."),
        ("keras_activations", "Keras — Layer activation functions", "https://keras.io/api/layers/activations/", "Uso das ativações e propriedades de ReLU, Sigmoid e Softmax."),
        ("keras_dense", "Keras — Dense layer", "https://keras.io/api/layers/core_layers/dense/", "Operação Dense e aplicação da activation."),
        ("tensorflow_loss", "TensorFlow — SparseCategoricalCrossentropy", "https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy", "Contrato de from_logits=True para saídas em logits."),
        ("keras_group_norm", "Keras — GroupNormalization", "https://keras.io/2/api/layers/normalization_layers/group_normalization/", "Normalização por grupos e eixo de canais/features."),
        ("mnist_source", "MNIST database — Yann LeCun", "https://yann.lecun.com/exdb/mnist/", "Referência da base MNIST."),
    ]
    for source_id, label, href, description in external:
        sources.append({"id": source_id, "label": label, "href": href, "query": {"language": "documentation", "description": description}})
    return sources


def write_snapshot(evidence: dict[str, Any]) -> None:
    rows = []
    for row in evidence["records"]:
        values = [
            repr(row["root_id"]), repr(row["dataset"]), repr(row["activation"]),
            str(row["extra_fraction"]), str(row["test_accuracy"]), str(row["test_macro_f1"]),
            str(row["test_loss"]), str(row["predicted_class_count"]), str(row["predicted_mode_share"]),
            str(row["logit_component_std_across_rows"]),
        ]
        rows.append("  (" + ", ".join(values) + ")")
    sql = (
        "-- Audit companion generated from investigacao_ativacoes_atualizada.json.\n"
        "-- Raw manifests/CSV/NumPy files remain the primary evidence.\n"
        "CREATE TABLE activation_runs (\n"
        "  root_id TEXT, dataset TEXT, activation TEXT, extra_fraction REAL,\n"
        "  test_accuracy REAL, test_macro_f1 REAL, test_loss REAL,\n"
        "  predicted_class_count INTEGER, predicted_mode_share REAL,\n"
        "  logit_component_std REAL\n"
        ");\nINSERT INTO activation_runs VALUES\n"
        + ",\n".join(rows)
        + ";\n"
    )
    OUTPUT_SNAPSHOT.write_text(sql, encoding="utf-8")


def build_artifact(evidence: dict[str, Any]) -> dict[str, Any]:
    generated_at = str(evidence["generated_at"])
    records = evidence["records"]
    new_records = [row for row in records if row["root_id"] == "augmentation_0_5"]
    softmax_records = [row for row in records if row["activation"] == "softmax"]
    new_mnist = next(row for row in new_records if row["dataset"] == "mnist" and row["activation"] == "softmax")
    new_fashion = next(row for row in new_records if row["dataset"] == "fashion_mnist" and row["activation"] == "softmax")
    sources = source_catalog(generated_at)

    performance_rows = [{
        "dataset": DATASET_LABELS[row["dataset"]], "activation": ACTIVATION_LABELS[row["activation"]],
        "test_accuracy": row["test_accuracy"], "test_macro_f1": row["test_macro_f1"],
        "test_loss": row["test_loss"], "root": row["root_label"],
    } for row in new_records]
    history_rows = [{
        "dataset": DATASET_LABELS[row["dataset"]], "activation": ACTIVATION_LABELS[row["activation"]],
        "series": f"{DATASET_LABELS[row['dataset']]} — {ACTIVATION_LABELS[row['activation']]}",
        "epoch": item["epoch"], "val_macro_f1": item["val_macro_f1"], "val_accuracy": item["val_accuracy"],
    } for row in new_records if row["dataset"] == "mnist" for item in row["history"]]
    robustness_rows = [{
        "scenario": f"{DATASET_LABELS[row['dataset']]} — {row['root_id'].replace('augmentation_', 'extra_fraction=')}",
        "dataset": DATASET_LABELS[row["dataset"]], "root": row["root_label"],
        "test_accuracy": row["test_accuracy"], "test_macro_f1": row["test_macro_f1"],
        "predicted_mode_share": row["predicted_mode_share"], "predicted_class": row["predicted_class_mode"],
        "logit_component_std": row["logit_component_std_across_rows"],
    } for row in softmax_records]
    metric_rows = [{
        "lote": row["root_label"], "dataset": DATASET_LABELS[row["dataset"]], "activation": ACTIVATION_LABELS[row["activation"]],
        "test_accuracy": row["test_accuracy"], "test_balanced_accuracy": row["test_balanced_accuracy"],
        "test_macro_f1": row["test_macro_f1"], "test_loss": row["test_loss"], "epochs": row["epochs"],
        "predicted_class_count": row["predicted_class_count"], "predicted_mode_share": row["predicted_mode_share"],
        "logit_component_std": row["logit_component_std_across_rows"],
    } for row in records]
    paired_rows = [{
        "dataset": DATASET_LABELS[row["dataset"]], "activation": ACTIVATION_LABELS[row["activation"]],
        "old_f1": row["old_test_macro_f1"], "new_f1": row["new_test_macro_f1"], "delta_f1": row["delta_test_macro_f1"],
        "old_accuracy": row["old_test_accuracy"], "new_accuracy": row["new_test_accuracy"],
        "delta_accuracy": row["delta_test_accuracy"], "old_training_hours": row["old_training_hours"],
        "new_training_hours": row["new_training_hours"], "delta_training_hours": row["delta_training_hours"],
    } for row in evidence["paired_comparisons"]]
    hidden_rows = [{
        "dataset": DATASET_LABELS[row["dataset"]], "activation": ACTIVATION_LABELS[row["activation"]], "layer": row["layer"],
        "sample_count": row["sample_count"], "mean_feature_std_across_samples": row["mean_feature_std_across_samples"],
        "mean_l2_distance_from_sample_mean": row["mean_l2_distance_from_sample_mean"], "max_abs_difference_from_first": row["max_abs_difference_from_first"],
        "softmax_last_axis_sum_mean": row.get("softmax_last_axis_sum_mean"), "softmax_last_axis_entropy_mean_nats": row.get("softmax_last_axis_entropy_mean_nats"),
    } for row in evidence["intermediate"]]
    status_rows = [{
        "lote": row["root_label"], "dataset": DATASET_LABELS[row["dataset"]], "activation": ACTIVATION_LABELS[row["activation"]],
        "status": "Completo" if row["status"] == "completed" else "Ausente/incompleto", "run_id": row["run_id"] or "—",
    } for row in evidence["status_matrix"]]
    headline = [{
        "new_completed_runs": len(new_records), "new_expected_runs": 9,
        "new_mnist_softmax_accuracy": new_mnist["test_accuracy"], "new_mnist_softmax_macro_f1": new_mnist["test_macro_f1"],
        "softmax_completed_collapsed_runs": sum(row["predicted_mode_share"] == 1.0 for row in softmax_records),
        "softmax_completed_runs": len(softmax_records),
    }]

    title = "Investigação atualizada: por que o Softmax colapsa no teste de ativação"
    executive = (
        "## Resumo executivo\n\n"
        f"Os novos treinos não corrigiram o problema do Softmax: nos dois casos completos do lote novo, MNIST e Fashion-MNIST, ele prevê uma única classe em 100% do teste. "
        f"No MNIST novo, a acurácia é {pct(new_mnist['test_accuracy'])} e o F1 macro {pct(new_mnist['test_macro_f1'])}; no Fashion-MNIST, {pct(new_fashion['test_accuracy'])} e {pct(new_fashion['test_macro_f1'])}. "
        f"No probe novo de MNIST, o desvio médio entre exemplos cai de {sci(next(x for x in evidence['intermediate'] if x['dataset']=='mnist' and x['activation']=='softmax' and x['layer']=='block1_activation1')['mean_feature_std_across_samples'])} no primeiro bloco para zero numérico em `dense1`/`dense2`. "
        "A causa melhor sustentada é arquitetural/dinâmica: Softmax foi usado como ativação oculta em 10 pontos convolucionais e 2 camadas densas. A camada final continua linear e a loss usa `from_logits=True`, portanto a evidência não aponta para dupla aplicação na saída."
    )
    diagnosis = (
        "## O que está acontecendo\n\n"
        "1. `hidden_activation` é aplicado após cada GroupNormalization nos 5 blocos convolucionais e em `Dense(256, activation=...)` duas vezes.\n"
        "2. Com `softmax` e eixo padrão `-1`, cada posição espacial convolucional é normalizada entre canais; nas camadas densas, cada vetor de 256 unidades vira uma distribuição que soma aproximadamente 1.\n"
        "3. A aplicação repetida restringe escala e geometria da representação. Sem residual/atalho para preservar o sinal pré-Softmax, os blocos seguintes recebem vetores progressivamente menos dependentes da imagem.\n"
        "4. O vetor final de logits fica quase constante; a argmax escolhe sempre uma classe e a acurácia se aproxima da prevalência dela, enquanto balanced accuracy e F1 macro caem porque as demais classes têm recall zero.\n\n"
        "Esta é uma inferência causal apoiada conjuntamente pelo código, pela concentração das predições, pelas curvas de validação e pelo probe intermediário; não é uma prova de que nenhuma outra implementação poderia produzir o mesmo sintoma."
    )
    protocol = (
        "## O que mudou nos novos treinos\n\n"
        "O lote novo mantém seed 42, split estratificado 70/15/15, batch 256, learning rate 0,0003, mixed_float16, arquitetura, normalização e avaliação sem augmentation. A mudança principal é `extra_fraction`: 2,0 no lote anterior versus 0,5 no novo, isto é, base + aproximadamente duas cópias aumentadas versus base + meia cópia no treino.\n\n"
        "A redução de dados aumentados cortou aproximadamente pela metade o tempo de treino, mas não alterou o colapso do Softmax. As diferenças de ReLU/Sigmoid entre lotes são observacionais; com uma única seed por célula, não são intervalos de confiança."
    )
    demonstration = (
        "## Demonstração e interpretação das métricas\n\n"
        f"- **Concentração:** os {len(softmax_records)} runs Softmax completos disponíveis têm participação da classe dominante igual a 100%.\n"
        f"- **MNIST:** o Softmax novo fica em {pct(new_mnist['test_accuracy'])} de acurácia, próximo da prevalência da classe escolhida, mas com F1 macro de {pct(new_mnist['test_macro_f1'])}.\n"
        f"- **Logits:** a variação média por componente no MNIST novo é {sci(new_mnist['logit_component_std_across_rows'])}; no lote anterior, {sci(next(row for row in softmax_records if row['dataset']=='mnist' and row['root_id']=='augmentation_2_0')['logit_component_std_across_rows'])}.\n"
        "- **Camadas:** no MNIST novo, a variação média do Softmax cai de 8,02e-3 no primeiro bloco para 2,18e-5 no bloco 3 e zero numérico nas densas; ReLU e Sigmoid mantêm variação não nula.\n\n"
        "Por isso, acurácia isolada é insuficiente: balanced accuracy e F1 macro revelam que nove classes não estão sendo reconhecidas."
    )
    limitations = (
        "## Limitações, qualidade e próximos testes\n\n"
        f"A matriz esperada tem 18 células; {len(records)} execuções estão completas e 5 estão ausentes/incompletas. O lote novo tem {len(new_records)}/9 células completas; KMNIST–Softmax possui apenas preflight e é inconclusivo. Todos os runs usam seed 42.\n\n"
        "O diagnóstico do Softmax é robusto ao que foi repetido: colapso em MNIST com dois valores de extra_fraction e em Fashion-MNIST no lote novo. As variações de ReLU/Sigmoid entre lotes não devem ser atribuídas exclusivamente a augmentation sem replicação por seed.\n\n"
        "Recomendação: retirar Softmax das ativações ocultas ou registrá-lo como ablação negativa. Se o objetivo for testar Softmax como saída, separar `hidden_activation` de `output_activation`, treinar com logits lineares e `from_logits=True` e aplicar Softmax somente na inferência. Um teste de 5–10 épocas pode confirmar o colapso antes de repetir 100 épocas."
    )
    sources_text = (
        "## Fontes utilizadas\n\n"
        "**Evidência local:** manifests, históricos, métricas, previsões e logits dos diretórios `outputs/controlled-augmentation2-activations-mac2/` e `outputs/controlled-augmentation05-activations-mac2/`; `src/tcc_benchmark/model.py`, `src/tcc_benchmark/data.py`, `src/tcc_benchmark/metrics.py`, `docs/BENCHMARK_CONTROLADO.md` e `docs/GUIA_TECNICO.md`.\n\n"
        "**Referências técnicas:** [Keras Softmax layer](https://keras.io/api/layers/activation_layers/softmax/), [Keras activation functions](https://keras.io/api/layers/activations/), [Keras Dense layer](https://keras.io/api/layers/core_layers/dense/), [TensorFlow SparseCategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy), [Keras GroupNormalization](https://keras.io/2/api/layers/normalization_layers/group_normalization/) e [MNIST database](https://yann.lecun.com/exdb/mnist/)."
    )

    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": f"# {title}"},
        {"id": "executive", "type": "markdown", "layout": "full", "body": executive, "sourceId": "run_artifacts"},
        {"id": "headline", "type": "metric-strip", "layout": "full", "cardIds": ["new_runs", "mnist_softmax_accuracy", "mnist_softmax_f1", "collapsed_runs"]},
        {"id": "performance_chart", "type": "chart", "layout": "full", "chartId": "performance_chart"},
        {"id": "performance_finding", "type": "markdown", "layout": "full", "body": "## Desempenho no lote novo\n\nO Softmax é o único que falha de forma catastrófica nas duas bases com resultado completo. ReLU e Sigmoid preservam classificação em MNIST, Fashion-MNIST e, nas células disponíveis, KMNIST; Softmax fica no regime de uma classe.", "sourceId": "run_artifacts"},
        {"id": "robustness_chart", "type": "chart", "layout": "full", "chartId": "robustness_chart"},
        {"id": "robustness_finding", "type": "markdown", "layout": "full", "body": "## Robustez do padrão de colapso\n\nA concentração de 100% não depende de um único lote de augmentation: ela aparece no MNIST com `extra_fraction=2,0` e `0,5`, e no Fashion-MNIST com `0,5`. A classe escolhida muda entre bases, mas o mecanismo é o mesmo: a saída deixa de depender do exemplo.", "sourceId": "run_artifacts"},
        {"id": "history_chart", "type": "chart", "layout": "full", "chartId": "history_chart"},
        {"id": "hidden_variation_table", "type": "table", "layout": "full", "tableId": "hidden_variation_table"},
        {"id": "diagnosis", "type": "markdown", "layout": "full", "body": diagnosis},
        {"id": "protocol", "type": "markdown", "layout": "full", "body": protocol, "sourceId": "run_artifacts"},
        {"id": "paired_table", "type": "table", "layout": "full", "tableId": "paired_table"},
        {"id": "demonstration", "type": "markdown", "layout": "full", "body": demonstration, "sourceId": "run_artifacts"},
        {"id": "metrics_table", "type": "table", "layout": "full", "tableId": "metrics_table"},
        {"id": "status_table", "type": "table", "layout": "full", "tableId": "status_table"},
        {"id": "limitations", "type": "markdown", "layout": "full", "body": limitations, "sourceId": "run_artifacts"},
        {"id": "sources", "type": "markdown", "layout": "full", "body": sources_text},
    ]
    cards = [
        {"id": "new_runs", "dataset": "headline", "description": "Execuções completas do lote novo", "sourceId": "run_artifacts", "metrics": [{"label": "Completas", "field": "new_completed_runs", "format": "number"}, {"label": "Esperadas", "field": "new_expected_runs", "format": "number"}]},
        {"id": "mnist_softmax_accuracy", "dataset": "headline", "description": "Softmax no MNIST novo", "sourceId": "run_artifacts", "metrics": [{"label": "Acurácia", "field": "new_mnist_softmax_accuracy", "format": "percent"}]},
        {"id": "mnist_softmax_f1", "dataset": "headline", "description": "Softmax no MNIST novo", "sourceId": "run_artifacts", "metrics": [{"label": "F1 macro", "field": "new_mnist_softmax_macro_f1", "format": "percent"}]},
        {"id": "collapsed_runs", "dataset": "headline", "description": "Runs Softmax completos com uma única classe prevista", "sourceId": "run_artifacts", "metrics": [{"label": "Colapsados", "field": "softmax_completed_collapsed_runs", "format": "number"}, {"label": "Softmax completos", "field": "softmax_completed_runs", "format": "number"}]},
    ]
    charts = [
        {"id": "performance_chart", "dataset": "performance_chart", "type": "bar", "intent": "comparison", "title": "Métricas de teste por ativação — lote novo", "subtitle": "Softmax cai para o regime de uma única classe; ReLU/Sigmoid permanecem altos.", "showDescription": True, "encodings": {"x": {"field": "dataset", "type": "nominal"}, "y": {"field": "test_macro_f1", "type": "quantitative"}, "color": {"field": "activation", "type": "nominal"}}, "sourceId": "run_artifacts"},
        {"id": "robustness_chart", "dataset": "robustness_chart", "type": "bar", "intent": "comparison", "title": "Concentração das predições nos Softmax completos", "subtitle": "Todos os runs Softmax completos atribuem 100% das amostras à classe dominante.", "showDescription": True, "encodings": {"x": {"field": "scenario", "type": "nominal"}, "y": {"field": "predicted_mode_share", "type": "quantitative"}, "color": {"field": "dataset", "type": "nominal"}}, "sourceId": "run_artifacts"},
        {"id": "history_chart", "dataset": "history_chart", "type": "line", "intent": "trend", "title": "F1 macro de validação por época — MNIST novo", "subtitle": "ReLU e Sigmoid aprendem; Softmax permanece próximo de zero.", "showDescription": True, "encodings": {"x": {"field": "epoch", "type": "quantitative"}, "y": {"field": "val_macro_f1", "type": "quantitative"}, "color": {"field": "activation", "type": "nominal"}}, "sourceId": "run_artifacts"},
    ]
    tables = [
        {"id": "hidden_variation_table", "dataset": "hidden_variation", "title": "Variação entre exemplos nas camadas internas", "subtitle": "Probe de 256 exemplos do primeiro lote de teste nos checkpoints do lote novo.", "showDescription": True, "columns": [{"field": "dataset", "label": "Dataset"}, {"field": "activation", "label": "Ativação"}, {"field": "layer", "label": "Camada"}, {"field": "sample_count", "label": "Amostras", "type": "number", "format": "number"}, {"field": "mean_feature_std_across_samples", "label": "Desvio médio entre exemplos", "type": "number", "format": "number"}, {"field": "mean_l2_distance_from_sample_mean", "label": "Distância L2 média", "type": "number", "format": "number"}, {"field": "max_abs_difference_from_first", "label": "Máx. diferença vs. primeira", "type": "number", "format": "number"}, {"field": "softmax_last_axis_sum_mean", "label": "Soma eixo Softmax", "type": "number", "format": "number"}], "defaultSort": {"field": "dataset", "direction": "asc"}, "sourceId": "run_artifacts"},
        {"id": "paired_table", "dataset": "paired_table", "title": "Efeito observado da redução de extra_fraction", "subtitle": "Pares disponíveis; deltas = novo menos anterior.", "showDescription": True, "columns": [{"field": "dataset", "label": "Dataset"}, {"field": "activation", "label": "Ativação"}, {"field": "old_f1", "label": "F1 anterior", "type": "percent", "format": "percent"}, {"field": "new_f1", "label": "F1 novo", "type": "percent", "format": "percent"}, {"field": "delta_f1", "label": "Delta F1", "type": "percent", "format": "percent"}, {"field": "old_accuracy", "label": "Acc. anterior", "type": "percent", "format": "percent"}, {"field": "new_accuracy", "label": "Acc. nova", "type": "percent", "format": "percent"}, {"field": "delta_training_hours", "label": "Delta horas", "type": "number", "format": "number"}], "defaultSort": {"field": "dataset", "direction": "asc"}, "sourceId": "run_artifacts"},
        {"id": "metrics_table", "dataset": "metrics_table", "title": "Tabela completa das execuções disponíveis", "subtitle": "Uma linha por execução completa; valores de teste e sinais de concentração.", "showDescription": True, "columns": [{"field": "lote", "label": "Lote"}, {"field": "dataset", "label": "Dataset"}, {"field": "activation", "label": "Ativação"}, {"field": "test_accuracy", "label": "Acurácia", "type": "percent", "format": "percent"}, {"field": "test_balanced_accuracy", "label": "Balanced acc.", "type": "percent", "format": "percent"}, {"field": "test_macro_f1", "label": "F1 macro", "type": "percent", "format": "percent"}, {"field": "test_loss", "label": "Loss", "type": "number", "format": "number"}, {"field": "epochs", "label": "Épocas", "type": "number", "format": "number"}, {"field": "predicted_class_count", "label": "Classes previstas", "type": "number", "format": "number"}, {"field": "predicted_mode_share", "label": "Dominante", "type": "percent", "format": "percent"}], "defaultSort": {"field": "test_macro_f1", "direction": "desc"}, "sourceId": "run_artifacts"},
        {"id": "status_table", "dataset": "status_table", "title": "Matriz de completude", "subtitle": "Células ausentes/incompletas não foram interpretadas como resultados.", "showDescription": True, "columns": [{"field": "lote", "label": "Lote"}, {"field": "dataset", "label": "Dataset"}, {"field": "activation", "label": "Ativação"}, {"field": "status", "label": "Status"}, {"field": "run_id", "label": "Run ID"}], "defaultSort": {"field": "status", "direction": "asc"}, "sourceId": "run_artifacts"},
    ]
    return {"surface": "report", "manifest": {"version": 1, "surface": "report", "title": title, "description": "Investigação atualizada e demonstrativa do colapso do Softmax em ativações internas.", "generatedAt": generated_at, "sources": sources, "blocks": blocks, "cards": cards, "charts": charts, "tables": tables}, "snapshot": {"version": 1, "status": "ready", "generatedAt": generated_at, "datasets": {"headline": headline, "performance_chart": performance_rows, "robustness_chart": robustness_rows, "history_chart": history_rows, "hidden_variation": hidden_rows, "paired_table": paired_rows, "metrics_table": metric_rows, "status_table": status_rows}}, "sources": sources}


def build_markdown(evidence: dict[str, Any]) -> str:
    records = evidence["records"]
    new_records = [row for row in records if row["root_id"] == "augmentation_0_5"]
    new_mnist_softmax = next(row for row in new_records if row["dataset"] == "mnist" and row["activation"] == "softmax")
    new_fashion_softmax = next(row for row in new_records if row["dataset"] == "fashion_mnist" and row["activation"] == "softmax")
    old_mnist_softmax = next(row for row in records if row["root_id"] == "augmentation_2_0" and row["dataset"] == "mnist" and row["activation"] == "softmax")
    lines = [
        "# Investigação atualizada: por que o Softmax colapsa no teste de ativação", "",
        f"Data da análise: {evidence['generated_at']}.", "", "## Conclusão", "",
        f"Os novos treinos confirmam o diagnóstico: o Softmax não está apenas pior; ele colapsa. No MNIST novo, obteve {pct(new_mnist_softmax['test_accuracy'])} de acurácia e {pct(new_mnist_softmax['test_macro_f1'])} de F1 macro, prevendo a classe {new_mnist_softmax['predicted_class_mode']} em {pct(new_mnist_softmax['predicted_mode_share'])} das amostras. No Fashion-MNIST novo, obteve {pct(new_fashion_softmax['test_accuracy'])} de acurácia e {pct(new_fashion_softmax['test_macro_f1'])} de F1 macro, prevendo a classe {new_fashion_softmax['predicted_class_mode']} em 100% das amostras.", "",
        f"O padrão é robusto à mudança de augmentation no MNIST: com extra_fraction=2,0, o Softmax também teve {pct(old_mnist_softmax['test_accuracy'])} de acurácia, {pct(old_mnist_softmax['test_macro_f1'])} de F1 e 100% de concentração. A causa melhor sustentada é o uso de Softmax como ativação oculta repetida, não um erro na loss final.", "",
        "## Evidências principais", "",
        "| Lote | Dataset | Ativação | Acurácia | F1 macro | Classes previstas | Classe dominante |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(f"| {row['root_label']} | {DATASET_LABELS[row['dataset']]} | {ACTIVATION_LABELS[row['activation']]} | {pct(row['test_accuracy'])} | {pct(row['test_macro_f1'])} | {row['predicted_class_count']} | {pct(row['predicted_mode_share'])} |")
    lines.extend([
        "", "## O que e por que está acontecendo", "",
        "No código, `hidden_activation` é aplicado após cada GroupNormalization nos 5 blocos convolucionais e nas duas camadas densas de 256 unidades. Com `softmax`, cada vetor ao longo do último eixo vira uma distribuição não negativa cuja soma é aproximadamente 1. Em convoluções isso ocorre entre canais de cada posição espacial; nas densas, entre as 256 unidades.", "",
        "A aplicação repetida restringe a representação e reduz sua escala. Sem conexão residual preservando o sinal pré-Softmax, os blocos seguintes recebem vetores cada vez menos informativos. O probe do lote novo mostra a progressão: no MNIST Softmax, a variação média entre exemplos é 8,02e-3 em `block1_activation1`, 2,18e-5 em `block3_activation1`, cerca de 9,77e-9 em `block5_activation2` e zero numérico em `dense1`/`dense2`.", "",
        "Isso explica o classificador quase constante: o último vetor de logits muda muito pouco entre exemplos, a argmax escolhe sempre uma classe e a acurácia se aproxima da prevalência dela. A camada final é linear e o compilador usa `SparseCategoricalCrossentropy(from_logits=True)`, combinação coerente com logits; portanto, o problema não é um Softmax duplicado na saída.", "",
        "## Novos dados e efeito de `extra_fraction`", "",
        "O lote novo reduziu `extra_fraction` de 2,0 para 0,5, mantendo seed, split, arquitetura, batch, learning rate, dtype e avaliação. Isso reduziu o tempo de treino aproximadamente pela metade, mas não alterou o colapso do Softmax. Para ReLU e Sigmoid, houve variação entre lotes; como existe uma única seed por célula, esses deltas são observacionais e não representam intervalo de confiança.", "",
        "| Dataset | Ativação | F1 anterior | F1 novo | Delta F1 | Horas anteriores | Horas novas |", "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in evidence["paired_comparisons"]:
        lines.append(f"| {DATASET_LABELS[row['dataset']]} | {ACTIVATION_LABELS[row['activation']]} | {pct(row['old_test_macro_f1'])} | {pct(row['new_test_macro_f1'])} | {pp(row['delta_test_macro_f1'])} | {num(row['old_training_hours'], 2)} | {num(row['new_training_hours'], 2)} |")
    lines.extend([
        "", "## Qualidade, limites e recomendação", "",
        f"A matriz esperada tem 18 células; {len(records)} estão completas. No lote novo, 8 de 9 estão completas. KMNIST–Softmax está incompleto e foi mantido como inconclusivo. Todos os runs disponíveis usam seed 42, então não há estimativa de variabilidade entre seeds.", "",
        "Recomendo retirar Softmax do conjunto de ativações ocultas ou registrá-lo como ablação negativa. Se o objetivo for testar Softmax como saída, separar `hidden_activation` de `output_activation`: treinar com logits lineares e `from_logits=True`, aplicando Softmax somente na inferência; ou usar saída probabilística com `from_logits=False`. Um teste curto de 5–10 épocas pode confirmar o colapso sem repetir o custo de 100 épocas.", "",
        "## Fontes", "",
        "- Evidência local: `outputs/controlled-augmentation2-activations-mac2/`, `outputs/controlled-augmentation05-activations-mac2/`, `src/tcc_benchmark/model.py`, `src/tcc_benchmark/data.py`, `src/tcc_benchmark/metrics.py`, `docs/BENCHMARK_CONTROLADO.md` e `docs/GUIA_TECNICO.md`.",
        "- [Keras Softmax layer](https://keras.io/api/layers/activation_layers/softmax/)", "- [Keras activation functions](https://keras.io/api/layers/activations/)", "- [Keras Dense layer](https://keras.io/api/layers/core_layers/dense/)", "- [TensorFlow SparseCategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy)", "- [Keras GroupNormalization](https://keras.io/2/api/layers/normalization_layers/group_normalization/)", "- [MNIST database — Yann LeCun](https://yann.lecun.com/exdb/mnist/)",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    write_snapshot(evidence)
    artifact = build_artifact(evidence)
    OUTPUT_ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_MARKDOWN.write_text(build_markdown(evidence), encoding="utf-8")
    print(OUTPUT_ARTIFACT)
    print(OUTPUT_MARKDOWN)
    print(OUTPUT_SNAPSHOT)


if __name__ == "__main__":
    main()
