# Relatório técnico — CNN SVHN INT8 para sistemas embarcados

## Resumo

Este documento descreve o modelo `svhn_embedded_int8.tflite`, treinado para reconhecer os dez dígitos do conjunto **SVHN** (*Street View House Numbers*). Ele foi projetado especificamente para execução embarcada com aritmética quantizada: a exportação final possui entrada e saída `int8`, não contém tensores `float32` e usa acumuladores `int32` nas convoluções e nas camadas densas.

O modelo alcançou **80,93% de acurácia** no conjunto oficial de teste SVHN (26.032 imagens). A versão de treinamento em ponto flutuante alcançou 81,12%; a quantização INT8 reduziu a acurácia em somente **0,19 ponto percentual**.

> Escopo: este é um modelo novo e compacto, criado para substituir a CNN SVHN legada que usava `GroupNormalization` e `swish`. A arquitetura legada não pôde ser convertida por LiteRT para uma execução inteiramente inteira.

## Dados

Foram usados os arquivos originais do SVHN no formato MATLAB:

| Divisão | Origem | Uso | Quantidade |
|---|---|---:|---:|
| Treino | `train_32x32.mat` | ajuste dos pesos | 65.931 imagens |
| Validação | `train_32x32.mat` | acompanhamento durante o treinamento | 7.326 imagens |
| Teste | `test_32x32.mat` | avaliação final, sem ajuste | 26.032 imagens |

O arquivo de treino contém 73.257 imagens. Ele foi separado em 90% para treino e 10% para validação por uma divisão estratificada por classe, com semente 42. As imagens foram reorganizadas para o formato `N × 32 × 32 × 3`, convertidas para `float32` e normalizadas para o intervalo `[0, 1]` pela divisão por 255. No SVHN, o dígito zero é codificado como rótulo 10; o pré-processamento aplica `rótulo % 10` para convertê-lo em classe 0.

Não há aumento de dados (*data augmentation*) implementado no script de treinamento.

## Arquitetura

A rede usa convoluções separáveis para reduzir parâmetros e custo de processamento. Cada bloco convolucional é seguido de normalização por lote (*Batch Normalization*) e ReLU6; as normalizações são absorvidas/convertidas na exportação para o grafo quantizado.

| Etapa | Operação | Saída | Parâmetros |
|---|---|---:|---:|
| Entrada | imagem RGB | `32 × 32 × 3` | 0 |
| 1 | Conv2D 3×3, 16 filtros, sem viés + BatchNorm + ReLU6 | `32 × 32 × 16` | 496 |
| 2 | MaxPool 2×2 | `16 × 16 × 16` | 0 |
| 3 | SeparableConv2D 3×3, 24 filtros + BatchNorm + ReLU6 | `16 × 16 × 24` | 624 |
| 4 | MaxPool 2×2 | `8 × 8 × 24` | 0 |
| 5 | SeparableConv2D 3×3, 32 filtros + BatchNorm + ReLU6 | `8 × 8 × 32` | 1.112 |
| 6 | MaxPool 2×2 | `4 × 4 × 32` | 0 |
| 7 | SeparableConv2D 3×3, 48 filtros + BatchNorm + ReLU6 | `4 × 4 × 48` | 2.016 |
| 8 | GlobalAveragePooling2D | `48` | 0 |
| 9 | Densa, 64 unidades, ReLU | `64` | 3.136 |
| Saída | Densa, 10 unidades, Softmax | `10` | 650 |

O total é de **8.034 parâmetros**, dos quais 7.794 são treináveis e 240 são estatísticas não treináveis das camadas BatchNorm. A estimativa analítica é de **673.408 MACs** por imagem (uma MAC equivale a uma multiplicação seguida de uma acumulação), ou aproximadamente 1,35 milhão de operações se multiplicação e soma forem contadas separadamente.

## Procedimento de treinamento

O treinamento foi executado com TensorFlow 2.21.0, em WSL2, apenas com CPU (nenhuma GPU física foi detectada). A política numérica durante o ajuste foi `float32`; a quantização aconteceu após o treinamento.

| Item | Configuração |
|---|---|
| Semente | 42 |
| Otimizador | Adam |
| Taxa de aprendizado inicial | `1e-3` |
| Função de perda | Sparse Categorical Crossentropy |
| Métrica | Sparse Categorical Accuracy |
| Lote | 256 imagens |
| Máximo de épocas | 16 |
| Épocas concluídas | 16 |
| Early stopping | `val_accuracy`, paciência 4, restaura melhores pesos |
| Redução de taxa | `val_accuracy`, paciência 2, fator 0,5, mínimo `1e-5` |

O treinamento atingiu 85,21% de acurácia no subconjunto de treino e 80,13% na validação na última época. A rotina completa — carregamento, treinamento, avaliação, exportação e validação INT8 — levou 118,32 s; esse número não deve ser interpretado como tempo de treinamento isolado.

## Exportação e validação INT8

A conversão foi feita por *post-training quantization* do LiteRT/TensorFlow Lite. Foram apresentadas as primeiras 256 imagens de treino como conjunto representativo de calibração. O conversor foi restringido a operações nativas INT8 e a interface foi fixada em `int8` tanto na entrada quanto na saída.

Validações feitas após a conversão:

- 24 tensores `int8` e 10 tensores `int32`;
- nenhum tensor em ponto flutuante;
- operadores: `CONV_2D`, `DEPTHWISE_CONV_2D`, `MAX_POOL_2D`, `MEAN`, `FULLY_CONNECTED` e `SOFTMAX`;
- a acurácia foi recalculada em todas as 26.032 imagens do conjunto de teste, usando a entrada quantizada conforme a escala e o ponto zero do modelo.

| Versão | Perda no teste | Acurácia no teste |
|---|---:|---:|
| Keras FP32 | 0,6182 | 81,12% |
| LiteRT INT8 | — | 80,93% |
| Diferença INT8 − FP32 | — | **−0,19 p.p.** |

## Uso de memória e desempenho medidos no host

| Medida | Resultado | Interpretação |
|---|---:|---|
| Arquivo LiteRT | 23.224 bytes (22,7 KiB) | armazenamento do modelo; normalmente fica na flash do microcontrolador |
| Tensores de execução declarados | 40.836 bytes (39,9 KiB) | soma dos tensores que não são constantes |
| Todos os tensores declarados | 49.388 bytes (48,2 KiB) | inclui tensores constantes e de execução |
| Latência mediana no host | 0,0165 ms/imagem | LiteRT, CPU x86, uma thread; não é uma medição de ESP32 |
| Percentil 95 no host | 0,0301 ms/imagem | mesma limitação acima |

A soma de 40.836 bytes não é o tamanho definitivo do *tensor arena* do TensorFlow Lite Micro: alocação, alinhamento e buffers temporários do runtime podem mudar esse valor, enquanto a reutilização de buffers pode reduzi-lo. Por isso, para uma implantação no ESP32, a memória da arena e a latência devem ser medidas após compilar o firmware na placa. A métrica de RSS do processo Python não representa o consumo de memória do microcontrolador e não deve ser usada para esse fim.

## Limitações e próximos passos

- Não foi feito teste de latência, arena de memória ou energia em uma placa ESP32 física; os números de tempo são apenas do host x86.
- Não há modelo INT4 inteiramente inteiro neste experimento. A rota padrão de conversão empregada suporta o modelo INT8 validado; um INT4 implantável exigiria kernels e runtime específicos.
- Para uma comparação justa com a CNN SVHN legada, seria necessário avaliar as duas arquiteturas com o mesmo particionamento e protocolo de teste.

## Reprodutibilidade e artefatos

- Script de treinamento e conversão: [`scripts/train_svhn_embedded_int8.py`](../scripts/train_svhn_embedded_int8.py)
- Resumo completo do experimento: [`outputs/svhn-embedded-int8-2026-09-16/summary.json`](../outputs/svhn-embedded-int8-2026-09-16/summary.json)
- Perfil de tensores e latência: [`outputs/svhn-embedded-int8-2026-09-16/litert_profile.json`](../outputs/svhn-embedded-int8-2026-09-16/litert_profile.json)
- Modelo Keras FP32: [`outputs/svhn-embedded-int8-2026-09-16/svhn_embedded_float32.keras`](../outputs/svhn-embedded-int8-2026-09-16/svhn_embedded_float32.keras)
- Modelo LiteRT INT8: [`outputs/svhn-embedded-int8-2026-09-16/svhn_embedded_int8.tflite`](../outputs/svhn-embedded-int8-2026-09-16/svhn_embedded_int8.tflite)
