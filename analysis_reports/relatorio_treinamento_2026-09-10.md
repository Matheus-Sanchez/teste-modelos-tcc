# Relatório técnico de progresso do treinamento — 10/09/2026

## Resumo técnico

O supervisor continua rodando normalmente e ainda está na fase de testes de batch. No snapshot de **10/09/2026 às 11:10 BRT**, o processo ativo era o **CIFAR-10 com batch 32**, no epoch 27/100 já persistido e avançando no epoch seguinte. Não há evidência de interrupção do processo atual.

O progresso confirmado da matriz batch é de **12 células concluídas de 36**. Existe uma célula em execução real, quatro células antigas incompletas e 19 células ainda não inicializadas. As quatro células antigas estão marcadas como `running`, mas o log mostra `skip_needs_resume`; portanto, elas não estão sendo retomadas pelo comando atual, que foi iniciado sem `--resume`.

A quantização ainda não começou: a pasta `quantization/` não existe neste output root. Ela está habilitada no pipeline porque o comando não usa `--skip-quantization` e será chamada depois que a etapa batch terminar. As ativações estão deliberadamente desabilitadas por `--skip-activation` e devem ser executadas no outro Mac.

## Escopo e contagem da matriz

O experimento tem 9 datasets × 4 tamanhos de batch, totalizando **36 células**. Cada célula usa seed 42, normalização `unit_interval`, 100 epochs, `extra_fraction: 0.5`, augmentation configurada na suite e `mixed_float16`.

| Estado confirmado | Quantidade | Interpretação |
|---|---:|---|
| Concluídas com métricas finais | 12 | Possuem `test_metrics.json`, artefatos e resumo de telemetria |
| Em execução real | 1 | CIFAR-10, batch 32 |
| Parciais antigas que exigem retomada | 4 | Checkpoints existem, mas não há resultado final |
| Ainda não inicializadas | 19 | Sem checkpoint ou telemetria de execução |
| **Total da matriz** | **36** | |

Os 12 resultados concluídos representam **33,3% da matriz**. Se a célula atualmente em execução for incluída como trabalho já iniciado, o avanço operacional é de 13/36 células.

## Estado atual do processo

### Processo ativo

- Supervisor: PID 39140, iniciado em **08/09/2026 13:34:05 BRT**.
- Processo TensorFlow: PID 82989, iniciado em **10/09/2026 07:20:36 BRT**.
- Dataset/célula: `cifar10/batch-032`.
- Último epoch persistido: **27/100**.
- Métricas do último epoch persistido: `val_accuracy = 0,7030` e `val_macro_f1 = 0,7020`.
- Tempo médio recente por epoch: aproximadamente **421 s**.
- Log ao vivo: o epoch seguinte estava avançando por volta do step **508/1969**.

### Telemetria salva

Na amostra mais recente, às **11:10:26 BRT**, a telemetria registrou:

| Indicador | Valor aproximado |
|---|---:|
| CPU total do sistema | 26,1% |
| CPU do processo | 91,2% |
| RAM utilizada | 78,9% |
| RSS do processo | 1,62 GiB |
| Utilização da GPU Apple M4 | 80% |
| Memória alocada pelo driver Metal | 2,36 GiB |
| Pressão térmica | `nominal` |

A telemetria está sendo gravada em `batch/batch-032/cifar10/.../telemetry/samples.csv`. Não há sinal de pressão térmica, falha de memória ou processo parado.

## Resultados finais de batch já disponíveis

| Dataset | Batch | Macro-F1 teste | Accuracy teste | Média s/epoch | Treino total |
|---|---:|---:|---:|---:|---:|
| MNIST | 32 | 0,9882 | 0,9884 | 513,6 s | 14,27 h |
| MNIST | 64 | 0,9908 | 0,9909 | 297,5 s | 8,26 h |
| MNIST | 256 | 0,9901 | 0,9902 | 140,4 s | 3,90 h |
| Fashion-MNIST | 128 | 0,9112 | 0,9115 | 233,2 s | 6,48 h |
| Fashion-MNIST | 256 | 0,9097 | 0,9099 | 134,2 s | 3,73 h |
| KMNIST | 32 | 0,9863 | 0,9863 | 616,0 s | 17,11 h |
| KMNIST | 64 | 0,9859 | 0,9859 | 360,8 s | 10,02 h |
| KMNIST | 128 | 0,9825 | 0,9825 | 239,0 s | 6,64 h |
| KMNIST | 256 | 0,9846 | 0,9846 | 134,9 s | 3,75 h |
| EMNIST balanced | 64 | 0,8780 | 0,8794 | 672,3 s | 18,68 h |
| EMNIST balanced | 128 | 0,8729 | 0,8743 | 452,0 s | 12,56 h |
| EMNIST balanced | 256 | 0,8704 | 0,8717 | 278,9 s | 7,75 h |

Os resultados sugerem que batches maiores reduzem substancialmente o tempo por epoch. Porém, a escolha final não deve considerar apenas velocidade: em MNIST, por exemplo, batch 64 teve Macro-F1 ligeiramente superior ao batch 256; em EMNIST, o batch 64 também foi o melhor entre os três resultados concluídos.

O CIFAR-10 ativo ainda não possui Macro-F1 de teste final. O valor `val_macro_f1 = 0,7020` é de validação e não deve ser tratado como resultado final.

## Células parciais que não foram concluídas

Estas células possuem checkpoints, mas não possuem métricas de teste finais:

| Dataset | Batch | Último checkpoint observado | Situação |
|---|---:|---:|---|
| MNIST | 128 | 37 epochs | `running` antigo; log atual registra `skip_needs_resume` |
| Fashion-MNIST | 32 | 2 epochs | `running` antigo; não retomado pelo comando atual |
| Fashion-MNIST | 64 | 58 epochs | `running` antigo; não retomado pelo comando atual |
| EMNIST balanced | 32 | 66 epochs | `running` antigo; não retomado pelo comando atual |

Esses quatro lotes não podem ser considerados concluídos. Como o comando atual não contém `--resume`, o runtime os identifica como execuções que precisam de retomada e passa adiante. A seleção de batch também ignora linhas sem Macro-F1 final, mas a matriz completa permanece incompleta.

## Células ainda não inicializadas

Depois do CIFAR-10 batch 32 atual, ainda não há execução final registrada para:

- CIFAR-10: batch 64, 128 e 256.
- CIFAR-100 coarse: batches 32, 64, 128 e 256.
- SVHN: batches 32, 64, 128 e 256.
- GTSRB: batches 32, 64, 128 e 256.
- FER2013: batches 32, 64, 128 e 256.

Esses datasets ainda são a principal fonte de incerteza da previsão, pois seus tempos reais de execução neste output root ainda não foram medidos.

## Quantização e ativações

### Quantização

A quantização ainda não começou. Não existem artefatos em `quantization/`. O código do pipeline confirma que, sem `--skip-quantization`, a sequência após a matriz batch será:

1. Treinos de quantização FP32 e FP16 para os 9 datasets.
2. Conversão e benchmark INT8 PTQ para cada dataset.
3. Seleção da variante de quantização dentro da tolerância de Macro-F1.

Isso representa 18 treinos de quantização, além de 9 conversões/benchmarks INT8-PTQ.

### Ativações

As ativações não serão executadas neste Mac. A execução contém `--skip-activation`, e não há arquivos em `activations/`. Elas continuam pendentes para o outro Mac.

## Qualidade dos dados

Os 9 datasets passaram pela auditoria de entrada com **0 erros de leitura ou estrutura**:

| Dataset | Exemplos auditados | Erros | Alertas |
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

Os alertas correspondem a duplicatas documentadas pelo pipeline. CIFAR-100 coarse, SVHN e FER2013 também têm duplicatas com rótulos conflitantes; isso não bloqueou o treino, mas deve ser mencionado na interpretação das métricas finais.

## Interrupções anteriores e confiabilidade dos logs

O processo atual está ativo. A interrupção anterior pode ser delimitada entre:

- última telemetria do processo anterior: **04/09/2026 12:35:07 BRT**;
- nova execução do supervisor: **08/09/2026 13:34:05 BRT**.

O motivo exato não foi registrado. Não foram encontrados `KeyboardInterrupt`, `SIGTERM`, `SIGKILL`, `Killed`, `Traceback`, `OutOfMemory` ou mensagens fatais equivalentes no `pipeline.log`. O texto `OUT_OF_RANGE: End of sequence` é uma mensagem normal de encerramento da leitura de um dataset, não uma indicação de falha.

O `pipeline-status.json` ainda não contém as fases em tempo real (`stages: {}`), portanto o acompanhamento deve usar os processos, os checkpoints, os `status.json`, o `pipeline.log` e as amostras de telemetria. Isso é uma limitação de observabilidade, não uma evidência de que o treinamento parou.

## Previsão de término

As previsões abaixo são estimativas operacionais, não garantias. Elas assumem execução serial, ausência de nova interrupção e desempenho semelhante ao medido no CIFAR-10 e nos batches já concluídos.

| Fase | Previsão atual | Confiança | Base |
|---|---|---|---|
| CIFAR-10 batch 32 atual | 10/09, aproximadamente 20:00–22:00 BRT | Média/baixa | 73 epochs restantes e 421 s/epoch recentes, com margem para avaliação |
| Matriz batch das células que o supervisor ainda vai visitar | 18–28/09 | Baixa | 19 células não iniciadas, incluindo datasets ainda sem tempo real |
| Retomada dos 4 lotes parciais | Adicional; depende de executar com `--resume` | Baixa | Checkpoints existem, mas os tempos variam por dataset |
| Quantização FP32/FP16 + INT8 PTQ | 24/09–12/10 | Baixa | 18 treinos + 9 conversões após a fase batch |
| Ativações no outro Mac | Não estimável neste Mac | — | Execução separada, ainda sem artefatos nesta pasta |

O intervalo para a matriz batch é largo porque CIFAR-100, SVHN, GTSRB e FER2013 ainda não produziram tempos de 100 epochs neste output root. Quando o primeiro deles concluir, a previsão poderá ser recalculada com muito mais precisão.

## O que falta para fechar o experimento

1. Finalizar CIFAR-10 batch 32.
2. Executar os demais batches de CIFAR-10, CIFAR-100 coarse, SVHN, GTSRB e FER2013.
3. Retomar os quatro lotes parciais com `--resume` se todos os 36 resultados forem obrigatórios.
4. Gerar os vencedores de batch por dataset com métricas finais comparáveis.
5. Executar a quantização, que está habilitada no comando atual.
6. Executar e consolidar as ativações no outro Mac.
7. Corrigir ou atualizar o `pipeline-status.json` ao término para que o estado consolidado reflita as fases executadas.

## Conclusão

O treinamento **continua ativo**, está salvando checkpoints, métricas por epoch e telemetria, e já concluiu 12 das 36 células de batch. A fase atual é CIFAR-10 batch 32. A quantização ainda não começou, mas está programada para depois da fase batch; as ativações estão explicitamente separadas para outro Mac.

O principal ponto de atenção é que quatro células antigas estão incompletas e são puladas pelo comando atual por falta de `--resume`. Assim, o pipeline pode avançar para a quantização sem produzir uma matriz batch completamente preenchida, embora consiga selecionar vencedores usando apenas as células concluídas. Para considerar todos os 36 batches válidos, esses quatro lotes precisam ser retomados posteriormente.

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
