"""Build the canonical report artifact from the softmax investigation JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "analysis_reports" / "investigacao_mnist_softmax.json"
OUTPUT = ROOT / "analysis_reports" / "investigacao_mnist_softmax.artifact.json"

ACTIVATION_LABELS = {"relu": "ReLU", "sigmoid": "Sigmoid", "softmax": "Softmax"}
SELECTED_LAYERS = (
    "block1_activation1",
    "block3_activation1",
    "block5_activation2",
    "dense1",
    "dense2",
    "logits",
)


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%".replace(".", ",")


def number(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def pp(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):+.{digits}f} pp".replace(".", ",")


def source_catalog(generated_at: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "run_artifacts",
            "label": "Artefatos dos três runs de ativação MNIST",
            "path": "analysis_reports/investigacao_mnist_softmax_snapshot.sql",
            "query": {
                "language": "Python/JSON/CSV/NumPy",
                "description": "Leitura dos manifestos, históricos por época, test_metrics.json, predictions.csv e logits.npy dos runs ReLU, Sigmoid e Softmax.",
                "executed_at": generated_at,
                "metric_definitions": [
                    "Acurácia, balanced accuracy e F1 macro são as métricas de classificação recomputadas a partir dos rótulos previstos.",
                    "O conjunto de teste tem 10.498 amostras no split estratificado 70/15/15, seed 42.",
                    "Variação dos logits = desvio-padrão médio por componente entre as 10.498 linhas de logits.",
                    "Variação intermediária = estatísticas entre 256 amostras do primeiro lote do teste, usando os checkpoints best.keras.",
                ],
            },
        },
        {
            "id": "model_code",
            "label": "Implementação da arquitetura e da loss",
            "path": "src/tcc_benchmark/model.py",
            "query": {
                "language": "Python",
                "description": "Inspeção de build_legacy_cnn e compile_legacy_cnn para verificar onde a ativação é aplicada e como os logits são treinados.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "analysis_script",
            "label": "Script reprodutível da investigação",
            "path": "analysis_reports/investigate_mnist_softmax.py",
            "query": {
                "language": "Python",
                "description": "Reproduz a comparação, as contagens de previsão, a variação dos logits e as estatísticas das ativações intermediárias.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "keras_softmax",
            "label": "Keras — Softmax layer",
            "href": "https://keras.io/api/layers/activation_layers/softmax/",
            "query": {
                "language": "documentation",
                "description": "Definição da operação softmax, do eixo de normalização e da forma do resultado.",
            },
        },
        {
            "id": "keras_activations",
            "label": "Keras — Layer activation functions",
            "href": "https://keras.io/api/layers/activations/",
            "query": {
                "language": "documentation",
                "description": "Uso de ativações como argumento de camadas e propriedades das ativações sigmoid e softmax.",
            },
        },
        {
            "id": "keras_dense",
            "label": "Keras — Dense layer",
            "href": "https://keras.io/api/layers/core_layers/dense/",
            "query": {
                "language": "documentation",
                "description": "Operação Dense: activation(dot(input, kernel) + bias).",
            },
        },
        {
            "id": "tensorflow_loss",
            "label": "TensorFlow — SparseCategoricalCrossentropy",
            "href": "https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy",
            "query": {
                "language": "documentation",
                "description": "Contrato de from_logits=True para entradas que são logits.",
            },
        },
        {
            "id": "keras_group_norm",
            "label": "Keras — GroupNormalization",
            "href": "https://keras.io/2/api/layers/normalization_layers/group_normalization/",
            "query": {
                "language": "documentation",
                "description": "Normalização por grupos de canais e uso do eixo de features.",
            },
        },
        {
            "id": "mnist_source",
            "label": "MNIST database — Yann LeCun",
            "href": "https://yann.lecun.com/exdb/mnist/",
            "query": {
                "language": "dataset documentation",
                "description": "Fonte de referência do MNIST; o experimento local juntou os splits oficiais e refez o split estratificado conforme o protocolo do repositório.",
            },
        },
    ]


def build_artifact(evidence: dict[str, Any]) -> dict[str, Any]:
    generated_at = str(evidence["generated_at"])
    comparison = evidence["comparison"]
    by_activation = {row["activation"]: row for row in comparison}
    relu = by_activation["relu"]
    sigmoid = by_activation["sigmoid"]
    softmax = by_activation["softmax"]
    best_other = max(relu["test_macro_f1"], sigmoid["test_macro_f1"])
    sources = source_catalog(generated_at)

    performance_chart = []
    for row in comparison:
        for metric, field in (
            ("Acurácia", "test_accuracy"),
            ("Balanced accuracy", "test_balanced_accuracy"),
            ("F1 macro", "test_macro_f1"),
        ):
            performance_chart.append(
                {
                    "activation": ACTIVATION_LABELS[row["activation"]],
                    "metric": metric,
                    "value": row[field],
                    "test_loss": row["test_loss"],
                }
            )

    history_chart = [
        {
            "activation": ACTIVATION_LABELS[row["activation"]],
            "epoch": row["epoch"],
            "val_macro_f1": row["val_macro_f1"],
            "val_accuracy": row["val_accuracy"],
            "val_loss": row["val_loss"],
        }
        for row in evidence["history"]
    ]

    prediction_chart = []
    for row in evidence["prediction_distribution"]["softmax"]:
        prediction_chart.extend(
            [
                {"label": str(row["label"]), "kind": "Rótulo real", "count": row["true_count"]},
                {"label": str(row["label"]), "kind": "Predição softmax", "count": row["predicted_count"]},
            ]
        )

    hidden_rows = []
    for row in evidence["intermediate"]:
        if row.get("layer") not in SELECTED_LAYERS:
            continue
        hidden_rows.append(
            {
                "activation": ACTIVATION_LABELS.get(row["activation"], row["activation"]),
                "layer": row["layer"],
                "sample_count": row.get("sample_count"),
                "mean_feature_std_across_samples": row.get("mean_feature_std_across_samples"),
                "mean_l2_distance_from_sample_mean": row.get("mean_l2_distance_from_sample_mean"),
                "max_abs_difference_from_first": row.get("max_abs_difference_from_first"),
                "softmax_last_axis_sum_mean": row.get("softmax_last_axis_sum_mean"),
                "softmax_last_axis_entropy_mean_nats": row.get("softmax_last_axis_entropy_mean_nats"),
            }
        )

    metrics_table = [
        {
            "activation": ACTIVATION_LABELS[row["activation"]],
            "test_accuracy": row["test_accuracy"],
            "test_balanced_accuracy": row["test_balanced_accuracy"],
            "test_macro_f1": row["test_macro_f1"],
            "test_loss": row["test_loss"],
            "initial_val_macro_f1": row["initial_val_macro_f1"],
            "best_val_macro_f1": row["best_val_macro_f1"],
            "best_val_epoch": row["best_val_epoch"],
            "final_val_macro_f1": row["final_val_macro_f1"],
            "predicted_class_count": row["predicted_class_count"],
            "predicted_mode_share": row["predicted_mode_share"],
            "logit_max_abs_difference_from_first": row["logit_max_abs_difference_from_first"],
        }
        for row in comparison
    ]

    headline = [
        {
            "softmax_test_accuracy": softmax["test_accuracy"],
            "softmax_test_macro_f1": softmax["test_macro_f1"],
            "best_other_test_macro_f1": best_other,
            "softmax_predicted_mode_share": softmax["predicted_mode_share"],
            "softmax_predicted_class_count": softmax["predicted_class_count"],
            "softmax_logit_row_l2_from_mean": softmax["logit_row_l2_from_mean"],
            "softmax_test_samples": softmax["test_samples"],
        }
    ]

    experiment = evidence["experiment"]
    title = "Por que o softmax colapsou no teste de ativação do MNIST"
    summary_body = (
        "## Resumo técnico\n\n"
        f"O resultado muito inferior do softmax não é explicado por um erro de loss, por diferença no conjunto de teste ou por falha de hardware. "
        f"O código aplica softmax como ativação interna em todos os 5 blocos convolucionais e nas 2 camadas densas; no run MNIST, essa escolha fez a representação perder dependência da imagem. "
        f"Ao final, as {softmax['test_samples']:,} amostras receberam o mesmo vetor de logits, com diferença absoluta máxima de {softmax['logit_max_abs_difference_from_first']:.2e} entre qualquer linha e a primeira.\n\n"
        f"No teste, o softmax obteve acurácia de {pct(softmax['test_accuracy'])}, balanced accuracy de {pct(softmax['test_balanced_accuracy'])} e F1 macro de {pct(softmax['test_macro_f1'])}. "
        f"ReLU e Sigmoid ficaram em {pct(relu['test_macro_f1'])} e {pct(sigmoid['test_macro_f1'])} de F1 macro. "
        f"A evidência aponta para colapso arquitetural/dinâmico causado pelo uso do softmax em camadas ocultas; a loss final permaneceu corretamente configurada para receber logits."
    )
    summary_body = summary_body.replace(f"{softmax['test_samples']:,}", "10.498")

    blocks = [
        {"id": "title", "type": "markdown", "layout": "full", "body": f"# {title}"},
        {"id": "technical_summary", "type": "markdown", "layout": "full", "body": summary_body},
        {"id": "headline_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["softmax_accuracy", "softmax_f1", "prediction_concentration"]},
        {"id": "performance_chart", "type": "chart", "layout": "full", "chartId": "performance_chart"},
        {
            "id": "performance_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "run_artifacts",
            "body": (
                "## A queda é incompatível com uma simples variação aleatória entre ativações\n\n"
                f"O softmax perdeu {abs(100 * (softmax['test_accuracy'] - sigmoid['test_accuracy'])):.2f} pontos percentuais de acurácia e {abs(100 * (softmax['test_macro_f1'] - sigmoid['test_macro_f1'])):.2f} pontos percentuais de F1 macro em relação ao Sigmoid, que foi a melhor das outras duas variantes. "
                f"A balanced accuracy de {pct(softmax['test_balanced_accuracy'])} e o F1 macro de {pct(softmax['test_macro_f1'])} são compatíveis com uma regra que praticamente não distingue classes. "
                f"Como o conjunto tem 10 classes, a referência de acaso uniforme é 10%; o softmax ficou em {pct(softmax['test_accuracy'])} porque escolheu a classe 1, que representa {pct(1181 / softmax['test_samples'])} do teste."
            ),
        },
        {"id": "history_chart", "type": "chart", "layout": "full", "chartId": "history_chart"},
        {
            "id": "history_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "run_artifacts",
            "body": (
                "## O treinamento do softmax não sai do regime de chance\n\n"
                f"A curva de validação mostra a diferença de dinâmica: ReLU e Sigmoid crescem ao longo das 100 épocas, enquanto o Softmax começa em F1 macro de {pct(softmax['initial_val_macro_f1'])}, atinge o máximo na época {softmax['best_val_epoch']} e permanece em {pct(softmax['final_val_macro_f1'])}. "
                f"A loss de validação fica próxima de {number(softmax['test_loss'], 3)} e a acurácia de treino termina em {pct(softmax['final_train_accuracy'])}; portanto, não há sinal de que o modelo tenha aprendido e apenas generalizado mal."
            ),
        },
        {"id": "prediction_distribution_chart", "type": "chart", "layout": "full", "chartId": "prediction_distribution_chart"},
        {
            "id": "prediction_finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "run_artifacts",
            "body": (
                "## A matriz de confusão confirma o colapso em uma única classe\n\n"
                f"A distribuição real contém as dez classes, mas `predictions.csv` registra {softmax['predicted_class_count']} classe prevista: o dígito 1 em 100% das {softmax['test_samples']:,} amostras. "
                f"A matriz de confusão tem uma única coluna preenchida; a classe 1 tem recall 100%, as outras nove têm recall 0%. Isso explica o F1 macro de {pct(softmax['test_macro_f1'])} e separa o problema de um conjunto de classes desbalanceado: o desbalanceamento existe, mas é pequeno e a validação usa o mesmo split estratificado."
            ).replace(f"{softmax['test_samples']:,}", "10.498"),
        },
        {"id": "metrics_table", "type": "table", "layout": "full", "tableId": "metrics_table"},
        {
            "id": "scope_and_definitions",
            "type": "markdown",
            "layout": "full",
            "sourceId": "run_artifacts",
            "body": (
                "## O protocolo foi controlado; a ativação foi a variável principal\n\n"
                f"Os três runs usam seed {experiment['seed']}, o mesmo fingerprint de split `{experiment['split_fingerprint']}`, as mesmas frações 70/15/15, batch {experiment['batch_size']}, taxa de aprendizado {experiment['learning_rate']}, política `{experiment['dtype_policy']}`, normalização `{experiment['normalization']}` e balanceamento `{experiment['balance_mode']}`. "
                "A avaliação não aplica augmentação. Cada run treinou 100 épocas e o checkpoint usado foi o `best.keras`, selecionado pelo F1 macro de validação.\n\n"
                "A comparação principal usa métricas derivadas dos rótulos e logits salvos: acurácia é a fração de rótulos corretos; balanced accuracy é a média dos recalls das dez classes; F1 macro é a média não ponderada dos dez F1 por classe."
            ),
        },
        {"id": "hidden_variation_table", "type": "table", "layout": "full", "tableId": "hidden_variation_table"},
        {
            "id": "architecture_diagnosis",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## A causa está no lugar em que o softmax foi aplicado\n\n"
                "No `build_legacy_cnn`, `hidden_activation` é reutilizado em 10 ativações após GroupNormalization nos blocos convolucionais e em `Dense(256, activation=hidden_activation)` duas vezes. Com `softmax` e eixo padrão `-1`, cada posição espacial vira uma distribuição entre canais e cada vetor denso vira uma distribuição entre 256 unidades: valores não negativos, soma aproximadamente 1 e perda da escala comum do vetor.\n\n"
                "Essa restrição não é equivalente a usar softmax na saída de um classificador. Ela é aplicada repetidamente antes de convoluções, pooling e dropout, sem conexões residuais para preservar a informação. O probe do primeiro lote mostra a variação média entre amostras caindo de aproximadamente 8,02e-3 em `block1_activation1` para 5,62e-9 em `block5_activation2` e 2,43e-9 nas camadas densas; os logits finais ficam praticamente constantes."
            ),
        },
        {"id": "methodology", "type": "markdown", "layout": "full", "body":
         "## Como o diagnóstico foi demonstrado\n\n"
         "1. Comparei os manifestos para confirmar que os três runs têm o mesmo dataset, split, seed, batch, normalização, balanceamento, dtype, resolução e número de épocas.\n"
         "2. Recalculei a distribuição de previsões a partir de `predictions.csv`, a variação das linhas em `logits.npy` e as métricas de classificação presentes em `test_metrics.json`.\n"
         "3. Carreguei os checkpoints `best.keras` e medi a variação entre 256 exemplos do mesmo primeiro lote de teste em camadas intermediárias.\n"
         "4. Tracei a arquitetura no código para distinguir softmax oculto de softmax de saída e conferi o contrato `from_logits=True` da loss.\n\n"
         "A fórmula usada na interpretação é a própria definição do softmax: `softmax(z_i) = exp(z_i) / Σ_j exp(z_j)`, ao longo do eixo escolhido. A inferência adicional — que a aplicação repetida, combinada com GroupNorm e ausência de atalhos, colapsou a informação dependente da imagem — é sustentada conjuntamente pelas curvas, pelos logits e pelo probe de ativações; não é uma prova formal de unicidade da causa."
        },
        {"id": "limitations", "type": "markdown", "layout": "full", "sourceId": "run_artifacts", "body":
         "## Limitações e verificações de robustez\n\n"
         f"A matriz de ativações disponível contém um run por variante, todos com seed 42; portanto, a magnitude exata do desempenho não deve ser tratada como intervalo estatístico entre seeds. O padrão de colapso, porém, é muito forte: validação e teste exibem a mesma classe dominante, 100 épocas permanecem planas e os logits são quase invariantes entre imagens.\n\n"
         "Há uma pequena inconsistência operacional nos artefatos legados: no run ReLU, o campo `keras_metrics.accuracy` registra 98,83%, enquanto a acurácia independente derivada dos logits/predictions é 98,74%; Sigmoid e Softmax coincidem essencialmente. A comparação deste relatório usa as métricas independentes salvas em `classification`, e a discrepância do wrapper de métrica Keras deve ser revalidada antes da publicação final do TCC."
        },
        {"id": "next_steps", "type": "markdown", "layout": "full", "body":
         "## Próximos passos recomendados\n\n"
         "- Retirar `softmax` do conjunto de ativações internas do benchmark ou mantê-lo explicitamente como um resultado negativo esperado, sem misturá-lo com uma comparação de ativações ocultas convencionais.\n"
         "- Se a pergunta for testar softmax na saída, separar `hidden_activation` de `output_activation`: manter a última camada linear com `SparseCategoricalCrossentropy(from_logits=True)` para treino e aplicar softmax apenas na decodificação/inferência, ou usar a combinação probabilística equivalente com `from_logits=False`.\n"
         "- Repetir ReLU e Sigmoid com pelo menos 3 seeds se a conclusão precisar de incerteza estatística; para Softmax, uma ablação curta de poucas épocas já deve reproduzir o colapso e economizar horas.\n"
         "- Corrigir ou documentar a diferença entre a métrica Keras e a métrica independente antes de consolidar a tabela final."
        },
        {"id": "further_questions", "type": "markdown", "layout": "full", "body":
         "## Questões em aberto\n\n"
         "- O colapso desaparece se o softmax for usado somente em uma camada, com temperatura ajustável ou com uma conexão residual?\n"
         "- Qual das normalizações internas — GroupNorm, eixo de softmax ou repetição em profundidade — produz a maior perda de variação quando alterada isoladamente?\n"
         "- A discrepância de 9 exemplos no campo `keras_metrics.accuracy` do ReLU vem do wrapper de métricas do Keras 3, da avaliação em GPU Metal ou de algum detalhe de cardinalidade do dataset?"
        },
        {"id": "sources_used", "type": "markdown", "layout": "full", "body":
         "## Fontes utilizadas\n\n"
         "**Artefatos e código do experimento:** `outputs/controlled-augmentation2-activations-mac2/activations/mnist/`, `outputs/controlled-augmentation2-activations-mac2/generated-suites/activations/`, `src/tcc_benchmark/model.py`, `src/tcc_benchmark/metrics.py` e `analysis_reports/investigate_mnist_softmax.py`.\n\n"
         "**Referências técnicas:** [Keras Softmax layer](https://keras.io/api/layers/activation_layers/softmax/), [Keras activation functions](https://keras.io/api/layers/activations/), [Keras Dense layer](https://keras.io/api/layers/core_layers/dense/), [TensorFlow SparseCategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy), [Keras GroupNormalization](https://keras.io/2/api/layers/normalization_layers/group_normalization/) e [documentação do MNIST](https://yann.lecun.com/exdb/mnist/)."
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Investigação técnica e demonstrativa do colapso observado quando softmax é usado em todas as ativações internas do CNN no MNIST.",
            "generatedAt": generated_at,
            "sources": sources,
            "blocks": blocks,
            "cards": [
                {
                    "id": "softmax_accuracy",
                    "dataset": "headline",
                    "description": "Acurácia de teste do run Softmax.",
                    "sourceId": "run_artifacts",
                    "metrics": [
                        {"label": "Acurácia Softmax", "field": "softmax_test_accuracy", "format": "percent"},
                        {"label": "Amostras", "field": "softmax_test_samples", "format": "number"},
                    ],
                },
                {
                    "id": "softmax_f1",
                    "dataset": "headline",
                    "description": "F1 macro do run Softmax em comparação com a melhor outra ativação.",
                    "sourceId": "run_artifacts",
                    "metrics": [
                        {"label": "F1 macro Softmax", "field": "softmax_test_macro_f1", "format": "percent"},
                        {"label": "Melhor outra ativação", "field": "best_other_test_macro_f1", "format": "percent"},
                    ],
                },
                {
                    "id": "prediction_concentration",
                    "dataset": "headline",
                    "description": "Concentração das previsões do Softmax.",
                    "sourceId": "run_artifacts",
                    "metrics": [
                        {"label": "Classe dominante", "field": "softmax_predicted_class_count", "format": "number"},
                        {"label": "Participação da dominante", "field": "softmax_predicted_mode_share", "format": "percent"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "performance_chart",
                    "dataset": "performance_chart",
                    "type": "bar",
                    "intent": "comparison",
                    "title": "Métricas de teste por ativação",
                    "subtitle": "ReLU e Sigmoid ficam próximas de 99%; Softmax cai para 11,25% de acurácia e 2,02% de F1 macro.",
                    "showDescription": True,
                    "encodings": {
                        "x": {"field": "activation", "type": "nominal"},
                        "y": {"field": "value", "type": "quantitative"},
                        "color": {"field": "metric", "type": "nominal"},
                    },
                    "sourceId": "run_artifacts",
                },
                {
                    "id": "history_chart",
                    "dataset": "history_chart",
                    "type": "line",
                    "intent": "trend",
                    "title": "F1 macro de validação por época",
                    "subtitle": "ReLU e Sigmoid aprendem ao longo das épocas; Softmax permanece em aproximadamente 2,02%.",
                    "showDescription": True,
                    "encodings": {
                        "x": {"field": "epoch", "type": "quantitative"},
                        "y": {"field": "val_macro_f1", "type": "quantitative"},
                        "color": {"field": "activation", "type": "nominal"},
                    },
                    "sourceId": "run_artifacts",
                },
                {
                    "id": "prediction_distribution_chart",
                    "dataset": "prediction_distribution_chart",
                    "type": "bar",
                    "intent": "comparison",
                    "title": "Rótulos reais e predições do Softmax",
                    "subtitle": "O teste contém as dez classes; todas as 10.498 predições foram atribuídas à classe 1.",
                    "showDescription": True,
                    "encodings": {
                        "x": {"field": "label", "type": "nominal"},
                        "y": {"field": "count", "type": "quantitative"},
                        "color": {"field": "kind", "type": "nominal"},
                    },
                    "sourceId": "run_artifacts",
                },
            ],
            "tables": [
                {
                    "id": "metrics_table",
                    "dataset": "metrics_table",
                    "title": "Tabela comparativa dos runs",
                    "subtitle": "Métricas de teste e sinais de colapso; uma linha por ativação, seed 42.",
                    "showDescription": True,
                    "columns": [
                        {"field": "activation", "label": "Ativação"},
                        {"field": "test_accuracy", "label": "Acurácia teste", "type": "percent", "format": "percent"},
                        {"field": "test_balanced_accuracy", "label": "Balanced accuracy", "type": "percent", "format": "percent"},
                        {"field": "test_macro_f1", "label": "F1 macro teste", "type": "percent", "format": "percent"},
                        {"field": "test_loss", "label": "Loss teste", "type": "number", "format": "number"},
                        {"field": "best_val_macro_f1", "label": "Melhor F1 validação", "type": "percent", "format": "percent"},
                        {"field": "best_val_epoch", "label": "Época melhor", "type": "number", "format": "number"},
                        {"field": "predicted_class_count", "label": "Classes previstas", "type": "number", "format": "number"},
                        {"field": "predicted_mode_share", "label": "Participação dominante", "type": "percent", "format": "percent"},
                    ],
                    "defaultSort": {"field": "test_macro_f1", "direction": "desc"},
                    "sourceId": "run_artifacts",
                },
                {
                    "id": "hidden_variation_table",
                    "dataset": "hidden_variation",
                    "title": "Variação entre exemplos nas ativações",
                    "subtitle": "Estatísticas do primeiro lote de teste (256 amostras); valores menores indicam representação mais invariável entre imagens.",
                    "showDescription": True,
                    "columns": [
                        {"field": "activation", "label": "Ativação"},
                        {"field": "layer", "label": "Camada"},
                        {"field": "sample_count", "label": "Amostras", "type": "number", "format": "number"},
                        {"field": "mean_feature_std_across_samples", "label": "Desvio médio entre amostras", "type": "number", "format": "number"},
                        {"field": "mean_l2_distance_from_sample_mean", "label": "Distância L2 média", "type": "number", "format": "number"},
                        {"field": "max_abs_difference_from_first", "label": "Máx. diferença vs. 1ª", "type": "number", "format": "number"},
                        {"field": "softmax_last_axis_sum_mean", "label": "Soma do eixo softmax", "type": "number", "format": "number"},
                    ],
                    "defaultSort": {"field": "activation", "direction": "asc"},
                    "sourceId": "run_artifacts",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "headline": headline,
                "performance_chart": performance_chart,
                "history_chart": history_chart,
                "prediction_distribution_chart": prediction_chart,
                "metrics_table": metrics_table,
                "hidden_variation": hidden_rows,
            },
        },
        "sources": sources,
    }


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    artifact = build_artifact(evidence)
    OUTPUT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
