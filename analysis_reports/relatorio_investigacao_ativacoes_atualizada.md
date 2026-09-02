# Investigação atualizada: por que o Softmax colapsa no teste de ativação

Data da análise: 2026-09-01T00:17:44.392718+00:00.

## Conclusão

Os novos treinos confirmam o diagnóstico: o Softmax não está apenas pior; ele colapsa. No MNIST novo, obteve 11,25% de acurácia e 2,02% de F1 macro, prevendo a classe 1 em 100,00% das amostras. No Fashion-MNIST novo, obteve 10,00% de acurácia e 1,82% de F1 macro, prevendo a classe 5 em 100% das amostras.

O padrão é robusto à mudança de augmentation no MNIST: com extra_fraction=2,0, o Softmax também teve 11,25% de acurácia, 2,02% de F1 e 100% de concentração. A causa melhor sustentada é o uso de Softmax como ativação oculta repetida, não um erro na loss final.

## Evidências principais

| Lote | Dataset | Ativação | Acurácia | F1 macro | Classes previstas | Classe dominante |
|---|---|---:|---:|---:|---:|---:|
| Lote anterior — extra_fraction=2,0 | Fashion-MNIST | ReLU | 87,60% | 87,07% | 10 | 12,70% |
| Lote anterior — extra_fraction=2,0 | Fashion-MNIST | Sigmoid | 91,81% | 91,79% | 10 | 10,58% |
| Lote anterior — extra_fraction=2,0 | MNIST | ReLU | 98,74% | 98,72% | 10 | 11,25% |
| Lote anterior — extra_fraction=2,0 | MNIST | Sigmoid | 98,92% | 98,92% | 10 | 11,26% |
| Lote anterior — extra_fraction=2,0 | MNIST | Softmax | 11,25% | 2,02% | 1 | 100,00% |
| Lote novo — extra_fraction=0,5 | Fashion-MNIST | ReLU | 87,82% | 87,50% | 10 | 13,21% |
| Lote novo — extra_fraction=0,5 | Fashion-MNIST | Sigmoid | 90,05% | 90,04% | 10 | 11,52% |
| Lote novo — extra_fraction=0,5 | Fashion-MNIST | Softmax | 10,00% | 1,82% | 1 | 100,00% |
| Lote novo — extra_fraction=0,5 | KMNIST | ReLU | 97,02% | 97,02% | 10 | 10,42% |
| Lote novo — extra_fraction=0,5 | KMNIST | Sigmoid | 97,11% | 97,11% | 10 | 10,25% |
| Lote novo — extra_fraction=0,5 | MNIST | ReLU | 98,37% | 98,36% | 10 | 11,44% |
| Lote novo — extra_fraction=0,5 | MNIST | Sigmoid | 98,46% | 98,44% | 10 | 11,28% |
| Lote novo — extra_fraction=0,5 | MNIST | Softmax | 11,25% | 2,02% | 1 | 100,00% |

## O que e por que está acontecendo

No código, `hidden_activation` é aplicado após cada GroupNormalization nos 5 blocos convolucionais e nas duas camadas densas de 256 unidades. Com `softmax`, cada vetor ao longo do último eixo vira uma distribuição não negativa cuja soma é aproximadamente 1. Em convoluções isso ocorre entre canais de cada posição espacial; nas densas, entre as 256 unidades.

A aplicação repetida restringe a representação e reduz sua escala. Sem conexão residual preservando o sinal pré-Softmax, os blocos seguintes recebem vetores cada vez menos informativos. O probe do lote novo mostra a progressão: no MNIST Softmax, a variação média entre exemplos é 8,02e-3 em `block1_activation1`, 2,18e-5 em `block3_activation1`, cerca de 9,77e-9 em `block5_activation2` e zero numérico em `dense1`/`dense2`.

Isso explica o classificador quase constante: o último vetor de logits muda muito pouco entre exemplos, a argmax escolhe sempre uma classe e a acurácia se aproxima da prevalência dela. A camada final é linear e o compilador usa `SparseCategoricalCrossentropy(from_logits=True)`, combinação coerente com logits; portanto, o problema não é um Softmax duplicado na saída.

## Novos dados e efeito de `extra_fraction`

O lote novo reduziu `extra_fraction` de 2,0 para 0,5, mantendo seed, split, arquitetura, batch, learning rate, dtype e avaliação. Isso reduziu o tempo de treino aproximadamente pela metade, mas não alterou o colapso do Softmax. Para ReLU e Sigmoid, houve variação entre lotes; como existe uma única seed por célula, esses deltas são observacionais e não representam intervalo de confiança.

| Dataset | Ativação | F1 anterior | F1 novo | Delta F1 | Horas anteriores | Horas novas |
|---|---|---:|---:|---:|---:|---:|
| MNIST | ReLU | 98,72% | 98,36% | -0,36 pp | 6,29 | 3,20 |
| MNIST | Sigmoid | 98,92% | 98,44% | -0,48 pp | 8,19 | 4,23 |
| MNIST | Softmax | 2,02% | 2,02% | +0,00 pp | 6,42 | 3,27 |
| Fashion-MNIST | ReLU | 87,07% | 87,50% | +0,43 pp | 5,72 | 3,27 |
| Fashion-MNIST | Sigmoid | 91,79% | 90,04% | -1,75 pp | 8,28 | 4,18 |

## Qualidade, limites e recomendação

A matriz esperada tem 18 células; 13 estão completas. No lote novo, 8 de 9 estão completas. KMNIST–Softmax está incompleto e foi mantido como inconclusivo. Todos os runs disponíveis usam seed 42, então não há estimativa de variabilidade entre seeds.

Recomendo retirar Softmax do conjunto de ativações ocultas ou registrá-lo como ablação negativa. Se o objetivo for testar Softmax como saída, separar `hidden_activation` de `output_activation`: treinar com logits lineares e `from_logits=True`, aplicando Softmax somente na inferência; ou usar saída probabilística com `from_logits=False`. Um teste curto de 5–10 épocas pode confirmar o colapso sem repetir o custo de 100 épocas.

## Fontes

- Evidência local: `outputs/controlled-augmentation2-activations-mac2/`, `outputs/controlled-augmentation05-activations-mac2/`, `src/tcc_benchmark/model.py`, `src/tcc_benchmark/data.py`, `src/tcc_benchmark/metrics.py`, `docs/BENCHMARK_CONTROLADO.md` e `docs/GUIA_TECNICO.md`.
- [Keras Softmax layer](https://keras.io/api/layers/activation_layers/softmax/)
- [Keras activation functions](https://keras.io/api/layers/activations/)
- [Keras Dense layer](https://keras.io/api/layers/core_layers/dense/)
- [TensorFlow SparseCategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy)
- [Keras GroupNormalization](https://keras.io/2/api/layers/normalization_layers/group_normalization/)
- [MNIST database — Yann LeCun](https://yann.lecun.com/exdb/mnist/)
