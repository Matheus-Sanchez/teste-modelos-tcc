# O Softmax interno colapsou a representação do MNIST

**Relatório técnico reproduzível — evidência gerada em 28/08/2026**

## Resumo executivo

O resultado ruim não foi causado pela função de perda nem por um simples problema de generalização. No modelo avaliado, `hidden_activation="softmax"` foi aplicado repetidamente dentro das 10 ativações convolucionais e das 2 camadas densas. Isso transformou cada vetor de canais/unidades em uma distribuição que soma aproximadamente 1, removendo escala e impondo competição entre características.

O efeito observável foi um colapso progressivo da informação dependente da imagem:

```text
F1 macro no teste
ReLU     98,72% |██████████████████████████████████████████████████|
Sigmoid  98,92% |██████████████████████████████████████████████████|
Softmax   2,02% |█                                                 |
```

O Softmax terminou prevendo somente o dígito **1** para os 10.498 exemplos de teste. Por isso, a acurácia ficou em **11,25%**, exatamente a participação real da classe 1 no teste. A balanced accuracy foi **10,00%**, indicando desempenho nulo nas demais classes.

## 1. Resposta direta à pergunta

O Softmax foi usado como ativação **interna**, e não apenas como conversor final para probabilidades. Em uma camada convolucional com formato `(batch, altura, largura, canais)`, o eixo final é o eixo dos canais; portanto, o Softmax normaliza os canais em cada posição espacial. Em uma camada densa, normaliza as 256 unidades.

Essa escolha é inadequada como substituta genérica de ReLU ou Sigmoid porque:

1. todas as saídas ficam não negativas;
2. cada vetor local passa a somar aproximadamente 1;
3. as unidades competem entre si, em vez de poderem responder independentemente;
4. a escala absoluta da ativação é perdida;
5. a aplicação repetida, após `GroupNormalization`, reduz drasticamente as diferenças entre imagens;
6. sem uma conexão residual que preserve o sinal anterior, o colapso se propaga até as camadas densas e os logits.

Essa conclusão é sustentada pelo experimento controlado: split, seed, arquitetura, otimização, normalização, aumento de dados, número de épocas e hardware foram mantidos iguais; a ativação foi a variável comparada.

## 2. Comparação controlada

| Ativação | Acurácia | Balanced accuracy | F1 macro | Loss | Classes previstas | Classe dominante / participação |
|---|---:|---:|---:|---:|---:|---:|
| ReLU | 98,74% | 98,72% | 98,72% | 0,052 | 10 | 1 / 11,25% |
| Sigmoid | 98,92% | 98,91% | 98,92% | 0,037 | 10 | 1 / 11,26% |
| Softmax | 11,25% | 10,00% | 2,02% | 2,301 | 1 | 1 / 100,00% |

### Sanidade da acurácia do Softmax

O conjunto contém 1.181 exemplos verdadeiros da classe 1 em 10.498 exemplos:

```text
1.181 / 10.498 = 0,1124976 = 11,2498%
```

Como o modelo previu 1 em todos os casos, a acurácia obtida é exatamente essa fração. Isso explica a acurácia baixa, mas não é a causa primária: a causa é que o modelo perdeu a capacidade de produzir representações distintas por imagem.

## 3. Demonstração do colapso

### 3.1 Distribuição das previsões

| Dígito | Verdadeiro | Previsto pelo Softmax |
|---:|---:|---:|
| 0 | 1.035 | 0 |
| 1 | 1.181 | 10.498 |
| 2 | 1.048 | 0 |
| 3 | 1.071 | 0 |
| 4 | 1.023 | 0 |
| 5 | 947 | 0 |
| 6 | 1.031 | 0 |
| 7 | 1.094 | 0 |
| 8 | 1.024 | 0 |
| 9 | 1.044 | 0 |

A entropia da distribuição das classes previstas pelo Softmax foi **0 nats**: não há diversidade de classes na saída. ReLU e Sigmoid previram as 10 classes e tiveram entropia de aproximadamente **2,301 nats**, próxima de `ln(10) = 2,3026`.

### 3.2 A variação entre imagens desaparece

O probe foi feito no primeiro lote de teste, com 256 imagens. A métrica `mean_feature_std_across_samples` é o desvio-padrão médio de cada característica entre as imagens do lote.

| Camada | ReLU | Sigmoid | Softmax |
|---|---:|---:|---:|
| `block1_activation1` | 0,3353 | 0,0952 | 0,008017 |
| `block3_activation1` | 0,3171 | 0,1278 | 0,00002178 |
| `block5_activation2` | 0,5208 | 0,2573 | 0,000000005623 |
| `dense2` | 13,0418 | 0,3541 | 0,000000002431 |
| `logits` | 6,6068 | 5,1898 | 0,000000038404 |

No Softmax, o desvio médio cai de `8,017 × 10⁻³` no primeiro bloco para `5,623 × 10⁻⁹` em `block5_activation2` e chega a `2,431 × 10⁻⁹` em `dense2`. Nos logits completos, a maior diferença absoluta entre qualquer imagem e a primeira foi somente `2,27243 × 10⁻⁷`.

Em contraste, os logits de ReLU e Sigmoid variaram em aproximadamente 29,35 e 24,64 unidades, respectivamente, quando comparados à primeira imagem. Portanto, o Softmax não apenas classificou mal: produziu praticamente o mesmo vetor de saída para imagens diferentes.

### 3.3 O treinamento já nasce degenerado

| Ativação | Melhor época de F1 macro de validação | F1 macro inicial | F1 macro final | Acurácia final de treino |
|---|---:|---:|---:|---:|
| ReLU | 72 | — | 98,78% | 98,45% |
| Sigmoid | 83 | — | 99,04% | 98,85% |
| Softmax | 1 | 2,02% | 2,02% | 11,25% |

Para o Softmax, a validação permaneceu essencialmente plana durante as 100 épocas. Isso descarta a interpretação de que ele aprendeu e depois sofreu apenas overfitting.

## 4. Por que a normalização interna é destrutiva neste caso

O Softmax calcula, para um vetor `z`,

```text
softmax(z_i) = exp(z_i) / soma_j exp(z_j)
```

Assim, cada posição recebe valores positivos cuja soma é 1. Em uma convolução, isso faz os canais disputarem massa de probabilidade em cada pixel/posição. A representação deixa de dizer apenas “quanto cada filtro detectou” e passa a dizer “como distribuir um orçamento fixo entre filtros”.

Depois, o mesmo mecanismo é reaplicado em blocos sucessivos. No probe, a entropia do Softmax ficou próxima da distribuição uniforme no eixo normalizado:

- `block5_activation2`, 128 canais: **4,8517 nats**, próximo de `ln(128) = 4,8520`;
- `dense2`, 256 unidades: **5,5449 nats**, próximo de `ln(256) = 5,5452`.

Isso é compatível com vetores quase uniformes e pouco dependentes do exemplo. A consequência é uma entrada quase constante para a etapa final.

## 5. A função de perda estava pareada corretamente

O modelo termina em uma camada linear de 10 saídas, produzindo **logits**. A compilação usa:

```text
SparseCategoricalCrossentropy(from_logits=True)
```

Esse pareamento é correto: a própria loss aplica internamente a transformação apropriada dos logits. Logo, o diagnóstico não é “Softmax duplicado na saída”. O Softmax problemático está dentro das camadas ocultas.

Como verificação de escala, a loss do Softmax foi **2,301**, muito próxima de `ln(10) = 2,3026`, o valor esperado quando o modelo não separa as dez classes e fica próximo de uma previsão uniforme antes do `argmax`.

## 6. Protocolo e rastreabilidade

- Dataset: MNIST.
- Split local estratificado: 70% treino, 15% validação, 15% teste.
- Amostra de teste: 10.498 imagens.
- Seed: 42.
- Fingerprint do split: `e318c7442df3e0a1d1f67bd6fa1eac1602b6440bf135240aca0008293af1e6c8`.
- Batch size: 256.
- Learning rate: `0,0003`.
- Épocas máximas: 100.
- Política numérica: `mixed_float16`.
- Normalização: `unit_interval`.
- Balanceamento: `all_raw`.
- Aumento de dados: mesma política para as três variantes; avaliação sem aumento.
- Checkpoint analisado: `best.keras` de cada run.

Artefatos locais:

- Evidência consolidada: `analysis_reports/investigacao_mnist_softmax.json`.
- Script de investigação: `analysis_reports/investigate_mnist_softmax.py`.
- Código do modelo: `src/tcc_benchmark/model.py`.
- Métricas: `src/tcc_benchmark/metrics.py`.
- Runs brutos: `outputs/controlled-augmentation2-activations-mac2/activations/mnist/`.

## 7. Validação e limites

**Avaliação:** pronta para análise técnica e compartilhamento com as ressalvas abaixo.

### Checagens realizadas

- Acurácia do Softmax reconciliada com a participação real da classe 1: **11,2498%**.
- Contagens previstas somam exatamente os 10.498 exemplos de teste.
- As três variantes usam o mesmo fingerprint de split e o mesmo protocolo.
- O eixo Softmax é o eixo final, coerente com os formatos convolucionais e densos.
- A loss `from_logits=True` é coerente com a saída linear.
- O probe intermediário confirma a queda de variação antes do colapso dos logits.

### Limites

- A evidência identifica fortemente o Softmax interno como o mecanismo do colapso sob este protocolo, mas não separa cada possível interação entre Softmax, `GroupNormalization`, precisão mista e ausência de conexões residuais.
- A comparação de magnitudes absolutas entre ReLU, Sigmoid e Softmax deve ser interpretada com cuidado, pois as funções têm escalas diferentes. A tendência dentro da cadeia Softmax e a invariância dos logits são as evidências mais fortes.
- O dataset é MNIST; a conclusão deve ser validada novamente antes de generalizar para arquiteturas, tarefas ou datasets muito diferentes.

## 8. Conclusão prática

Para esta arquitetura, a recomendação é manter ReLU ou Sigmoid nas camadas ocultas e reservar Softmax para a saída probabilística — caso a saída final seja configurada com `from_logits=False`. Mantendo a saída linear e `from_logits=True`, não é necessário adicionar Softmax durante o treinamento.

Se o objetivo for estudar normalizações alternativas, faça comparações isoladas com: ativação interna sem Softmax, Softmax apenas em uma camada, `axis` explicitamente documentado, conexão residual e precisão `float32`. O critério principal deve incluir diversidade de previsões e variação dos logits, além de acurácia/F1.

## Fontes utilizadas

As fontes abaixo são documentação primária e foram consultadas para confirmar o comportamento matemático e a configuração das camadas:

1. [Keras — Softmax layer](https://keras.io/api/layers/activation_layers/softmax/): normalização no eixo especificado, por padrão o último eixo.
2. [Keras — Activation functions](https://keras.io/api/layers/activations/): definições e uso das ativações.
3. [Keras — Dense layer](https://keras.io/api/layers/core_layers/dense/): aplicação da transformação afim e da ativação na camada densa.
4. [Keras — GroupNormalization](https://keras.io/2/api/layers/normalization_layers/group_normalization/): comportamento da normalização usada antes das ativações nos blocos.
5. [TensorFlow — SparseCategoricalCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy): significado de `from_logits=True`.
6. [Yann LeCun — MNIST database](https://yann.lecun.com/exdb/mnist/): descrição e origem do dataset MNIST.
