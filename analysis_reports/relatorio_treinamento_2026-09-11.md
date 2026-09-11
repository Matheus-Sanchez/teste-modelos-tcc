# Relatório técnico de progresso do treinamento — 11/09/2026

## Resumo técnico

O treinamento continua ativo na fase de batch. No snapshot de **11/09/2026 às 20:55 BRT**, o processo real era **CIFAR-100 coarse, batch 32**, com **37 epochs persistidos** e o epoch seguinte quase concluído. O supervisor e o processo TensorFlow continuam vivos; não houve interrupção durante esta verificação.

Já existem **16 resultados finais de batch em 36 células**. A etapa CIFAR-10 foi concluída integralmente. Ainda há quatro execuções antigas parciais que não estão sendo retomadas pelo comando atual, pois ele não usa `--resume`, além de 15 células ainda não iniciadas.

A quantização ainda não começou: não há artefatos em `quantization/`. Ela está habilitada no comando atual, que usa `--skip-activation` mas não usa `--skip-quantization`, e será executada após a etapa batch. As ativações continuam separadas para o outro Mac.

## Estado da matriz batch

O desenho experimental é de 9 datasets × 4 batches (`32, 64, 128, 256`), totalizando 36 células.

| Estado | Quantidade | Situação |
|---|---:|---|
| Concluídas com métricas finais | 16 | Possuem métricas de teste, checkpoints, artefatos e telemetria |
| Em execução real | 1 | CIFAR-100 coarse, batch 32 |
| Parciais antigas | 4 | Checkpoints existem, mas sem avaliação final |
| Não iniciadas | 15 | Sem checkpoint ou telemetria de treino |
| **Total** | **36** | |

Os 16 resultados finais correspondem a 44,4% da matriz. A célula ativa não deve ser contada como concluída até terminar as 100 epochs e a avaliação de teste.

![Estado das 36 células de treinamento](charts/treinamento_matriz_status.png)

## Processo ativo e métricas atuais

- Supervisor: PID 39140, iniciado em **08/09/2026 13:34:05 BRT**.
- Processo TensorFlow: PID 73243, iniciado em **11/09/2026 16:16:22 BRT**.
- Célula ativa: `cifar100_coarse/batch-032`.
- Último epoch persistido: **37/100**.
- Últimas métricas persistidas: `val_accuracy = 0,4686` e `val_macro_f1 = 0,4615`.
- Últimos epochs: aproximadamente **413 s por epoch**.
- Log ao vivo: o epoch seguinte avançava em aproximadamente **1872/1969 steps**.

O `val_macro_f1` atual é apenas uma métrica de validação. Ainda não há `test_metrics.json` para CIFAR-100 coarse batch 32, portanto não é possível compará-lo aos resultados finais dos outros datasets.

![Curva de validação do lote ativo](charts/treinamento_curva_validacao_lote_ativo.png)

## Hardware e persistência dos dados de uso

A telemetria está sendo capturada e salva em `telemetry/samples.csv`. Na amostra mais recente, às **20:55:20 BRT**, foram registrados:

| Indicador | Valor |
|---|---:|
| CPU total do sistema | 20,8% |
| CPU do processo | 108,9% |
| RAM utilizada | 79,1% |
| Utilização GPU Apple M4 | 64% |
| Memória alocada pelo driver Metal | 2,21 GiB |
| Pressão térmica | `nominal` |

A CPU do processo pode superar 100% porque o monitor agrega o uso de múltiplos núcleos. Não há evidência de OOM, throttling térmico ou perda de telemetria no lote ativo.

![Telemetria do lote ativo](charts/treinamento_hardware_lote_ativo.png)

## Resultados finais disponíveis

| Dataset | Batches concluídos | Melhor Macro-F1 de teste | Melhor batch |
|---|---|---:|---:|
| MNIST | 32, 64, 256 | 0,9908 | 64 |
| Fashion-MNIST | 128, 256 | 0,9112 | 128 |
| KMNIST | 32, 64, 128, 256 | 0,9863 | 32 |
| EMNIST balanced | 64, 128, 256 | 0,8780 | 64 |
| CIFAR-10 | 32, 64, 128, 256 | 0,7235 | 32 |

Os tempos confirmam o ganho operacional de batches maiores, mas a melhor precisão não é sempre obtida no batch mais rápido. Exemplos: MNIST teve melhor Macro-F1 no batch 64, KMNIST no batch 32 e EMNIST no batch 64. A seleção final deve ponderar precisão, latência e tempo de treino por dataset.

![Tempo médio por época nas células concluídas](charts/treinamento_tempo_por_epoca.png)

## Células parciais que exigem retomada

Quatro células antigas possuem checkpoints, mas não possuem resultado final:

| Dataset | Batch | Último checkpoint |
|---|---:|---:|
| MNIST | 128 | 37 epochs |
| Fashion-MNIST | 32 | 2 epochs |
| Fashion-MNIST | 64 | 58 epochs |
| EMNIST balanced | 32 | 66 epochs |

O `pipeline.log` registra `skip_needs_resume` para essas células. Como a execução ativa foi iniciada sem `--resume`, ela passa adiante em vez de retomar os checkpoints. Isso permite que o supervisor continue, mas deixa a matriz batch incompleta caso todos os 36 resultados sejam obrigatórios.

## Células ainda não iniciadas

As 15 células sem execução final são:

- CIFAR-100 coarse: batches 64, 128 e 256;
- SVHN: batches 32, 64, 128 e 256;
- GTSRB: batches 32, 64, 128 e 256;
- FER2013: batches 32, 64, 128 e 256.

Esses datasets concentram a maior incerteza da previsão, porque ainda não há duração de 100 epochs medida neste output root.

## Quantização e ativações

### Quantização

A fase ainda não foi iniciada. O código do pipeline confirma que, sem `--skip-quantization`, a sequência prevista após a matriz batch é:

1. Treino FP32 e FP16 para cada um dos 9 datasets, totalizando 18 treinos.
2. Conversão e benchmark INT8 PTQ para cada dataset, totalizando 9 conversões/benchmarks.
3. Seleção da variante dentro da tolerância de Macro-F1.

### Ativações

As ativações não serão executadas neste Mac. O comando contém `--skip-activation`, a pasta `activations/` não existe e os testes devem ser feitos no outro Mac.

## Qualidade dos datasets

Os 9 datasets passaram pela auditoria de leitura e estrutura com zero erros:

| Dataset | Exemplos | Erros | Alertas |
|---|---:|---:|---:|
| MNIST | 70.000 | 0 | 0 |
| Fashion-MNIST | 70.000 | 0 | 0 |
| KMNIST | 70.000 | 0 | 0 |
| EMNIST balanced | 131.600 | 0 | 1 |
| CIFAR-10 | 60.000 | 0 | 0 |
| CIFAR-100 coarse | 60.000 | 0 | 2 |
| SVHN | 99.289 | 0 | 2 |
| GTSRB | 39.270 | 0 | 0 |
| FER2013 | 35.887 | 0 | 2 |

Os alertas são duplicatas detectadas pelo pipeline. CIFAR-100 coarse, SVHN e FER2013 também apresentam duplicatas com rótulos conflitantes; isso não bloqueou o treino, mas deve ser considerado na interpretação das métricas finais.

## Interrupção anterior e estado dos logs

O processo atual está ativo. A interrupção anterior fica delimitada entre a última telemetria de **04/09/2026 12:35:07 BRT** e o reinício do supervisor em **08/09/2026 13:34:05 BRT**. A causa exata não foi registrada.

Não foram encontrados no `pipeline.log` os marcadores `KeyboardInterrupt`, `SIGTERM`, `SIGKILL`, `Killed`, `Traceback`, `OutOfMemory` ou falha fatal equivalente. `OUT_OF_RANGE: End of sequence` é uma mensagem normal do TensorFlow ao terminar a leitura de um dataset.

O arquivo `pipeline-status.json` ainda está com `stages: {}` e não é uma fonte confiável de progresso em tempo real. O acompanhamento correto usa processos, logs, `status.json`, checkpoints e telemetria.

## Previsão de término

As estimativas assumem execução serial, ausência de nova interrupção e desempenho semelhante ao observado até agora.

| Marco | Previsão | Confiança |
|---|---|---|
| CIFAR-100 coarse batch 32 | 12/09, aproximadamente 03:50–04:40 BRT | Média/baixa |
| Matriz batch visitada pelo supervisor, sem retomar os 4 parciais | 18–28/09 | Baixa |
| Matriz batch incluindo a retomada dos 4 parciais | posterior ao intervalo acima | Baixa |
| Batch + quantização, sem ativações | 24/09–12/10 | Baixa |
| Experimento completo com ativações | não estimável neste Mac | — |

O intervalo é amplo porque SVHN, GTSRB e FER2013 ainda não têm tempos reais de execução para esta configuração. A previsão poderá ser recalculada com mais precisão após a conclusão do primeiro batch de cada dataset.

## O que falta para fechar o experimento

1. Finalizar CIFAR-100 coarse batch 32 e os três batches seguintes.
2. Executar os quatro batches de SVHN, GTSRB e FER2013.
3. Retomar as quatro células parciais com `--resume` se todos os 36 batches forem obrigatórios.
4. Consolidar os vencedores de batch por dataset.
5. Executar a quantização, que está habilitada no pipeline atual.
6. Executar e consolidar as ativações no outro Mac.
7. Atualizar o estado consolidado após o término das fases.

## Evidências consultadas

- `outputs/controlled-augmentation2-mac-m4-aug05/logs/pipeline.log`
- `outputs/controlled-augmentation2-mac-m4-aug05/pipeline-status.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/status.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/checkpoints/epoch_metrics.csv`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/telemetry/samples.csv`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/logs/training_summary.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/audit/*/audit/audit.json`
- `scripts/run_controlled_pipeline.py`
- `src/tcc_benchmark/controlled.py`
