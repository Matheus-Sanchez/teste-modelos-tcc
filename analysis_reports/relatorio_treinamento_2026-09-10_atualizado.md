# Relatório de andamento do treinamento — 11/09/2026

**Snapshot:** 11/09/2026, aproximadamente 20:12 BRT  
**Escopo:** 9 datasets × 4 tamanhos de batch = 36 células, 100 épocas por célula.

## Resumo executivo

- O treinamento está **ativo** na célula **CIFAR-100 coarse, batch 32**.
- Há **32 épocas persistidas** e a **33ª época em andamento**, em aproximadamente **128/1969 steps** no momento do snapshot. O último `val_macro_f1` persistido é **0,4529** e ainda não há métrica final de teste para essa célula.
- A matriz tem **16/36 células concluídas (44,4%)**, uma célula ativa, quatro células parciais antigas e **15 ainda não iniciadas**.
- O hardware segue estável no Apple M4/Metal, sem alerta térmico ou evidência de falta de memória.

## O que já foi concluído

As 16 células com métricas finais disponíveis são:

| Dataset | Batches concluídos | Melhor Macro-F1 de teste observado |
|---|---|---:|
| CIFAR-10 | 32, 64, 128, 256 | 0,7235 (batch 32) |
| MNIST | 32, 64, 256 | 0,9908 (batch 64) |
| Fashion-MNIST | 128, 256 | 0,9112 (batch 128) |
| KMNIST | 32, 64, 128, 256 | 0,9863 (batch 32) |
| EMNIST balanced | 64, 128, 256 | 0,8780 (batch 64) |

Também já foram concluídos os gates de preflight, auditoria e smoke. Os nove datasets passaram pela auditoria de leitura/estrutura sem erros; permanecem apenas alertas de duplicatas em alguns datasets.

## Situação atual

O CIFAR-100 coarse batch 32 iniciou às 16:16 BRT. O último checkpoint registra a época 32; a 33ª está em andamento, com aproximadamente 128/1969 steps. As cinco épocas persistidas mais recentes levaram, em média, **413,2 s por época**. A última validação registrada foi `val_macro_f1 = 0,4529` e `val_accuracy = 0,4591`.

Desde o relatório anterior, foram finalizados CIFAR-10 batch 128, com Macro-F1 de teste **0,6948**, e batch 256, com **0,7018**. Os gráficos abaixo usam os artefatos salvos no output root e separam métricas observadas de previsões.

## Gráficos

![Estado das 36 células de treinamento](charts/treinamento_matriz_status.png)

*A matriz mostra 16 células concluídas, uma ativa, quatro parciais com checkpoint antigo e 15 pendentes.*

![Tempo médio por época nas células concluídas](charts/treinamento_tempo_por_epoca.png)

*Comparação do tempo médio por época nas 16 células que chegaram a 100 épocas e avaliação final.*

![Telemetria do lote ativo](charts/treinamento_hardware_lote_ativo.png)

*Uso de GPU Metal, CPU do processo e RAM reportada durante o lote CIFAR-100 coarse batch 32.*

![Curva de validação do lote ativo](charts/treinamento_curva_validacao_lote_ativo.png)

*Evolução do Macro-F1 e da acurácia de validação; a métrica de teste só será produzida ao final das 100 épocas.*

## O que falta

1. Finalizar CIFAR-100 coarse batch 32.
2. Executar os outros três batches de CIFAR-100 coarse e os quatro batches de SVHN, GTSRB e FER2013 — **15 células ainda não iniciadas**.
3. Tratar quatro células antigas que têm checkpoint, mas não resultado final: MNIST batch 128 (37 épocas), Fashion-MNIST batch 32 (2), Fashion-MNIST batch 64 (58) e EMNIST balanced batch 32 (66). Elas exigem uma execução explícita com `--resume` para entrar na matriz completa.
4. Executar a quantização: 18 treinos FP32/FP16 e 9 conversões/benchmarks INT8-PTQ. Essa fase ainda não começou.
5. Executar as ativações no outro Mac. Esta execução foi iniciada com as ativações desabilitadas e não possui artefatos dessa fase.

## Uso de hardware

O treinamento usa **Apple M4 com GPU de 10 núcleos**, TensorFlow 2.18.1 + tensorflow-metal, política `mixed_float16` e **16 GiB de memória unificada**.

Na janela das 120 amostras mais recentes da telemetria do lote ativo:

| Indicador | Observado |
|---|---:|
| CPU total do sistema | média 23,8% |
| CPU do processo | média 104,5%; pico 178,6% |
| RAM reportada pelo monitor | média 76,2%; última 77,0% |
| GPU Metal | média 62,0%; pico 80%; última 63% |
| Memória alocada pelo driver Metal | cerca de 2,30 GB |
| Memória do sistema em uso pelo GPU | cerca de 0,35 GB em média |
| Pressão térmica | `nominal` |

Como a memória é unificada no Apple Silicon, a alocação do driver não representa uma VRAM dedicada separada. Não há sinal atual de throttling térmico, OOM ou perda da telemetria. A CPU do processo pode superar 100% porque o monitor soma o uso em múltiplos núcleos.

## Previsão de término

As estimativas assumem execução serial, configuração inalterada e ausência de nova interrupção. A confiança é baixa para a matriz porque CIFAR-100, SVHN, GTSRB e FER2013 ainda não têm duração de 100 épocas medida neste output root.

| Marco | Previsão | Confiança |
|---|---|---|
| CIFAR-100 coarse batch 32 atual | aproximadamente 03:50–04:30 BRT em 12/09 | média/baixa |
| Matriz de batches, sem retomar as quatro células parciais | 18–26/09 | baixa |
| Batch + quantização, sem ativações | aproximadamente 24/09–12/10 | baixa |
| Experimento completo incluindo ativações no outro Mac | não estimável a partir desta execução | — |

Se as quatro células parciais forem obrigatórias, o prazo deve ser estendido além da estimativa mínima. A previsão do lote atual é separada da previsão da matriz completa.

## Evidências principais

- `outputs/controlled-augmentation2-mac-m4-aug05/logs/pipeline.log`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/status.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/checkpoints/epoch_metrics.csv`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/batch-032/cifar100_coarse/**/telemetry/samples.csv`
- `outputs/controlled-augmentation2-mac-m4-aug05/audit/*/audit/audit.json`
- `scripts/run_controlled_pipeline.py`
- `scripts/build_training_progress_charts.py`
- `analysis_reports/charts/`
