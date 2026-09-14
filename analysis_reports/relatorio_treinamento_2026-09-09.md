# Relatório técnico de auditoria do treinamento — 09/09/2026

## Resumo técnico

O treinamento **continua rodando**. O EMNIST balanced com batch 64 terminou às **09:40:16 BRT** e o supervisor avançou para o EMNIST balanced com batch 128. No snapshot de **09/09/2026 21:07:58 BRT**, o batch 128 estava no **epoch 86 de 100**, com o epoch seguinte quase concluído e telemetria sendo atualizada.

A execução anterior foi interrompida ou perdeu o processo entre **04/09/2026 12:35:07 BRT** e **08/09/2026 13:34:05 BRT**. O horário exato e a causa não estão registrados no `pipeline.log` nem nos eventos de estado. Não foram encontrados `KeyboardInterrupt`, `SIGTERM`, `SIGKILL`, `Killed`, `Traceback` ou erro fatal.

Os dados e a telemetria dos runs concluídos estão sendo salvos. O principal problema atual é de **estado persistido**: quatro runs antigos continuam marcados como `running`, embora não tenham processo ativo, e não possuem resultado final.

## Escopo, definição e contagem

O experimento avalia 9 datasets × 4 tamanhos de batch, totalizando **36 células de treinamento**. Cada célula usa uma seed, normalização `unit_interval`, 100 epochs, augmentation com `extra_fraction: 0.5` e política `mixed_float16`.

| Estado da matriz batch | Quantidade | Evidência/interpretação |
|---|---:|---|
| Concluídas | 10 | Têm métricas de teste e resumo de hardware |
| Em execução real | 1 | EMNIST balanced, batch 128 |
| `running` órfãs | 4 | Sem processo correspondente e sem resultado final |
| Não iniciadas | 21 | Ainda sem execução real |
| **Total** | **36** | |

Os quatro estados órfãos são `emnist_balanced/batch-032`, `fashion_mnist/batch-032`, `fashion_mnist/batch-064` e `mnist/batch-128`.

## O avanço confirmado desde a auditoria anterior

O EMNIST batch 64 agora está concluído:

| Dataset | Batch | Macro-F1 teste | Accuracy teste | Média s/epoch | Treino total |
|---|---:|---:|---:|---:|---:|
| EMNIST balanced | 64 | 0,8780 | 0,8794 | 672,3 s | 18,68 h |

O EMNIST batch 128 acumulou 86 epochs. O último epoch persistido levou **479,4 s**, e o melhor `val_macro_f1` observado até o momento foi **0,8720 no epoch 71**. Ainda não há métrica de teste final para esse batch; portanto, não é correto projetar o vencedor do EMNIST antes da avaliação.

## Resultados de batch disponíveis

| Dataset | Batch | Macro-F1 de teste | Média s/epoch | Treino total |
|---|---:|---:|---:|---:|
| MNIST | 32 | 0,9882 | 513,6 s | 14,27 h |
| MNIST | 64 | 0,9908 | 297,5 s | 8,26 h |
| MNIST | 256 | 0,9901 | 140,4 s | 3,90 h |
| Fashion-MNIST | 128 | 0,9112 | 233,2 s | 6,48 h |
| Fashion-MNIST | 256 | 0,9097 | 134,2 s | 3,73 h |
| KMNIST | 32 | 0,9863 | 616,0 s | 17,11 h |
| KMNIST | 64 | 0,9859 | 360,8 s | 10,02 h |
| KMNIST | 128 | 0,9825 | 239,0 s | 6,64 h |
| KMNIST | 256 | 0,9846 | 134,9 s | 3,75 h |
| EMNIST balanced | 64 | 0,8780 | 672,3 s | 18,68 h |

As métricas acima são descritivas dos runs concluídos. A seleção final de batch ainda não está disponível para todos os datasets.

## Situação do processo ativo e hardware

O processo ativo confirmado é:

- Supervisor iniciado em **08/09/2026 13:34:05 BRT**.
- Processo TensorFlow do EMNIST batch 128 iniciado em **09/09/2026 09:40:31 BRT**.
- Epoch persistido mais recente: **86/100**.
- Telemetria mais recente: **09/09/2026 21:07:58 BRT**.

Na última amostra, foram registrados aproximadamente:

| Indicador | Valor |
|---|---:|
| CPU total do sistema | 24,6% |
| CPU do processo | 82,5% |
| RAM utilizada | 83,3% |
| RSS do processo | 1,20 GiB |
| GPU Apple M4 | 84,0% |
| Memória alocada pelo driver Metal | 2,76 GiB |
| Memória de GPU/sistema reportada | 388 MiB |
| Pressão térmica | `nominal` |

Os runs concluídos registraram utilização máxima de GPU entre 94% e 100%, com pressão térmica `nominal`. Não há evidência de saturação térmica ou de ausência de telemetry no run ativo.

## Qualidade dos dados de entrada

Os 9 datasets passaram pela auditoria com **0 erros de leitura ou estrutura**:

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

Os alertas são duplicatas exatas documentadas pelo pipeline. CIFAR-100 coarse, SVHN e FER2013 também apresentam duplicatas com rótulos conflitantes; isso não impediu a execução, mas pode afetar a interpretação da precisão final nesses datasets.

## Quando o treinamento foi interrompido

### O que os logs comprovam

| Evento | Data/hora BRT | Significado |
|---|---|---|
| Último checkpoint do EMNIST batch 32 | 04/09 12:22:52 | Epoch 66; sem conclusão |
| Última amostra de telemetry do mesmo processo | 04/09 12:35:07 | Última evidência de execução anterior |
| Nova execução do supervisor | 08/09 13:34:05 | Novo processo foi iniciado |
| Novo processo EMNIST batch 64 | 08/09 13:35:34 | Retomada da fila, não do checkpoint antigo |

O `state_events.jsonl` do run antigo termina em `pending → running`, sem transição para `interrupted`, `failed` ou `completed`. O `pipeline-status.json` também não contém horário de término ou interrupção.

Assim, o horário exato da interrupção **não pode ser recuperado dos logs da aplicação**. A conclusão mais forte suportada pelos dados é que o processo anterior deixou de produzir dados após **04/09 12:35:07 BRT** e uma nova execução começou em **08/09 13:34:05 BRT**.

O texto `OUT_OF_RANGE: End of sequence` encontrado no TensorFlow é o encerramento normal da leitura de um dataset, não uma interrupção do treinamento. O aviso sobre `use_unbounded_threadpool` é de compatibilidade e também não aparece acompanhado de falha fatal.

## Previsão de término

### Batch 128 do EMNIST atual

Com 86 epochs persistidos e média recente de aproximadamente 480 s/epoch, restam cerca de 14 epochs, além da avaliação final. A previsão para o término do EMNIST batch 128 é **09/09/2026 entre 23:00 e 23:45 BRT**, incluindo margem para avaliação e gravação dos artefatos.

### Matriz completa de batch

Após o batch atual, ainda há o EMNIST batch 256 e 20 células dos datasets que ainda não têm duração medida neste output root. Como as durações variam bastante entre datasets, a previsão deve ser tratada como faixa:

| Fase | Previsão atual | Confiança |
|---|---|---|
| Término do EMNIST batch 128 | 09/09, 23:00–23:45 | Média |
| Término de toda a matriz batch | 18–26/09 | Baixa/média |
| Quantização do comando atual | +6–14 dias após batch | Baixa |
| **Batch + quantização, sem ativações** | **24/09–10/10** | **Baixa** |

O comando em execução usa `--skip-activation`, mas **não usa `--skip-quantization`**. Portanto, depois da matriz batch o supervisor planeja mais 18 treinos de quantização (FP32 e FP16 para os 9 datasets), além de 9 conversões/benchmarks INT8-PTQ. Até este snapshot, não existe saída em `quantization/` nem em `activations/`.

As ativações não entram na previsão deste Mac porque estão explicitamente desativadas e devem ser executadas no outro Mac.

## O que falta para considerar o experimento fechado

1. Finalizar o EMNIST batch 128 atual.
2. Executar o EMNIST batch 256 e as 20 células restantes sem medição completa.
3. Resolver os quatro estados órfãos antes da consolidação: retomar explicitamente com checkpoint ou classificar como descartados.
4. Gerar a comparação global e o vencedor de batch por dataset.
5. Executar a quantização, que não foi desativada no comando atual.
6. Executar e consolidar as ativações no outro Mac.
7. Atualizar o `pipeline-status.json` e os relatórios finais após o supervisor concluir.

## Conclusão

O treinamento está ativo e avançando; não foi interrompido durante o snapshot atual. A interrupção anterior ocorreu em uma janela entre 04/09 e 08/09, sem causa explícita nos logs. Os dados de entrada foram auditados sem erros, a telemetria está sendo capturada e salva, e os resultados concluídos são consistentes com execução acelerada no Apple M4. A previsão mais realista para finalizar tudo que este comando executa — batch e quantização, sem ativações — é **entre 24/09 e 10/10/2026**, com baixa confiança enquanto CIFAR, SVHN, GTSRB e FER2013 ainda não tiverem tempos reais.

## Evidências consultadas

- `outputs/controlled-augmentation2-mac-m4-aug05/logs/pipeline.log`
- `outputs/controlled-augmentation2-mac-m4-aug05/pipeline-status.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/status.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/logs/state_events.jsonl`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/checkpoints/epoch_metrics.csv`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/telemetry/samples.csv`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/summary.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/audit/*/audit/audit.json`
