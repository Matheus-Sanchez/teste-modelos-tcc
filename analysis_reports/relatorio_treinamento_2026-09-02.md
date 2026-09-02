# Relatório de status do treinamento — 02/09/2026

## Avaliação geral

**O treinamento continua rodando**, sem evidência de interrupção, erro térmico ou falta de memória. O processo supervisor está ativo há aproximadamente 1 dia e 6 horas, e o filho TensorFlow está treinando **KMNIST com batch 64**, na época 13 de 100.

O treinamento está tecnicamente saudável, mas a matriz ainda **não está cientificamente completa**. O diretório de saída foi reutilizado entre tentativas e contém três registros antigos com status `running`, porém sem processo correspondente. Como o comando atual não usa `--resume`, esses três registros não devem ser tratados como concluídos.

**Classificação da análise: compartilhar com ressalvas.** Os resultados individuais completos são utilizáveis; a conclusão da matriz inteira ainda depende de corrigir os três registros antigos e executar os datasets restantes.

## Configuração observada

- Suite: `controlled-augmentation2-mac-m4-aug05`
- `extra_fraction`: `0.5`
- Precisão: `mixed_float16` (FP16 misto)
- Seed: `42`
- Épocas máximas: `100`
- Ativação oculta: `swish`
- Batches: `32, 64, 128, 256`
- Comando atual: `--skip-activation`
- Quantização: **não foi pulada**; portanto, será executada após a fase batch
- Execução serial: um job TensorFlow por vez

## Estado da matriz batch

São 9 datasets × 4 batches = **36 células planejadas**.

| Estado | Quantidade | Detalhe |
|---|---:|---|
| Concluídas | 6 | MNIST b32/b64/b256; Fashion-MNIST b128/b256; KMNIST b32 |
| Em execução real | 1 | KMNIST b64, época 13/100 |
| `running` antigo sem processo | 3 | Fashion-MNIST b32/b64; MNIST b128 |
| Ainda sem diretório de execução | 26 | Inclui EMNIST, CIFAR10, CIFAR100 coarse, SVHN, GTSRB e FER2013 |
| Células ainda não finalizadas | 30 | 1 ativa + 3 antigas + 26 não iniciadas |

### Registros antigos que precisam de atenção

| Célula | Última época persistida | Melhor validação | Situação |
|---|---:|---:|---|
| Fashion-MNIST b32 | 2 | 0,862709 | Sem processo correspondente |
| Fashion-MNIST b64 | 58 | 0,918544 | Sem processo correspondente |
| MNIST b128 | 37 | 0,989001 | Sem processo correspondente |

Esses registros têm checkpoints parciais, mas não possuem métricas finais de teste. O comando atual tende a ignorá-los, pois não foi iniciado com `--resume`.

## Resultados completos já obtidos

| Dataset | Batch | Macro-F1 teste | Tempo de treinamento | Telemetria |
|---|---:|---:|---:|---:|
| MNIST | 32 | 0,988215 | 14,3 h | 10.763 amostras |
| MNIST | 64 | **0,990785** | 8,3 h | 6.362 amostras |
| MNIST | 256 | 0,990069 | 3,9 h | 3.027 amostras |
| Fashion-MNIST | 128 | **0,911170** | 6,5 h | 4.956 amostras |
| Fashion-MNIST | 256 | 0,909732 | 3,7 h | 2.928 amostras |
| KMNIST | 32 | 0,986290 | 17,1 h | 12.857 amostras |

### Interpretação

- **MNIST:** batch 64 foi o melhor resultado medido; batch 256 foi aproximadamente 2,1 vezes mais rápido que b64, com queda pequena de macro-F1.
- **Fashion-MNIST:** b128 apresentou o melhor resultado entre as células concluídas, enquanto b256 foi mais rápido, com diferença de aproximadamente 0,14 ponto percentual de macro-F1.
- **KMNIST:** b32 já atingiu macro-F1 de 0,986290. O b64 ainda não possui teste final.
- A redução de `extra_fraction` para 0,5 reduziu substancialmente o tempo observado, aproximadamente pela metade em comparação com a configuração anterior, sem queda generalizada de qualidade nos datasets já medidos.
- Não é válido extrapolar esses resultados para os datasets RGB, GTSRB, SVHN, EMNIST ou FER2013 antes de medir suas primeiras execuções.

## Célula atualmente ativa

**KMNIST b64**

- Época atual: **13/100**
- Exemplos efetivos por época: **73.500**
- Steps por época: **1.149**
- Tempo médio atual por época: **aproximadamente 369 s**
- Melhor macro-F1 de validação: **0,974053**, na época 11
- Macro-F1 da última época registrada: **0,969936**
- Teste final: ainda não executado

O comportamento é compatível com uma execução normal: checkpoints e métricas por época estão sendo atualizados, e a telemetria continua crescendo.

## Captura de hardware e armazenamento

A captura está funcionando e os dados estão sendo salvos.

- Telemetria da célula ativa: **1.022 amostras** no último snapshot.
- Intervalo configurado: **5 segundos**.
- Backend: `apple-metal-ioreg`.
- Último ponto observado: GPU em aproximadamente 82%, CPU do processo em 75,3% e RAM do sistema em 65,7%.
- Memória GPU: compartilhada com a memória unificada; último ponto em aproximadamente 1,17 GiB.
- Pressão térmica: `nominal`.
- Espaço livre no volume: aproximadamente **277 GiB**.
- Os seis runs completos possuem checkpoints, métricas finais, manifestos e resumos de telemetria.

Limitações conhecidas: o backend atual não fornece de forma consistente percentual de memória GPU, temperatura e potência. Isso limita a granularidade do diagnóstico, mas não indica ausência de captura.

## Gates e qualidade dos dados

As três etapas iniciais estão concluídas:

- Auditoria dos datasets: concluída.
- Preflight: concluído.
- Smoke tests: concluídos.

As auditorias registraram zero erros. Permanecem avisos nos dados de EMNIST, CIFAR100 coarse, SVHN e FER2013, que devem ser considerados na interpretação final, mas não estão impedindo o treinamento atual.

## Previsão de término

As previsões abaixo usam os tempos observados na configuração `extra_fraction=0,5` e um fator central de aproximadamente 0,50 em relação à previsão anterior. Datasets ainda não medidos introduzem incerteza relevante.

| Escopo | Tempo restante estimado | Previsão | Confiança |
|---|---:|---|---|
| KMNIST b64 ativo | 8–10 h | 03/09, aproximadamente 04:45 BRT | Média |
| Fase batch no caminho formal atual, ignorando os 3 registros antigos | 310–380 h | 15–18/09 | Média-baixa |
| Fase batch com matriz cientificamente completa | 340–410 h | 17–19/09 | Baixa |
| Este Mac: batch + quantização, caminho formal | 680–820 h | 01–07/10 | Baixa |
| Este Mac: batch + correção dos 3 registros + quantização | 710–850 h | 02–08/10 | Baixa |
| Ativações no segundo Mac, se iniciadas agora | 120–170 h | 07–10/09 | Baixa |

### O que está incluído nessa previsão

O comando atual inclui, depois da fase batch:

- 18 treinos de quantização: 9 datasets × FP32/FP16.
- 9 conversões/testes PTQ.
- 0 ativações neste Mac, pois `--skip-activation` está ativo.

A quantização é o principal gargalo projetado. As ativações do segundo Mac não são observáveis neste workspace; portanto, a previsão delas é apenas uma extrapolação operacional.

## Verificações de cálculo

- **Contagem da matriz:** 9 datasets × 4 batches = 36 células; 6 concluídas + 1 ativa + 3 antigas + 26 não materializadas = 36.
- **Célula ativa:** 100 − 13 = 87 épocas restantes; 87 × aproximadamente 369 s = aproximadamente 8,9 h.
- **Métrica:** macro-F1 foi lida dos resumos e manifestos dos runs concluídos; não foi tratada como zero quando ausente.
- **Hardware:** telemetria foi separada da métrica de qualidade; utilização da GPU não foi interpretada como percentual de memória GPU.
- **Previsão:** tempos futuros são estimativas, não resultados observados; o intervalo foi ampliado para os seis datasets ainda não medidos.

## Conclusão

O treinamento não parou e está conseguindo treinar corretamente no Mac. Os dados de uso de hardware estão sendo capturados e salvos. O risco atual não é operacional, e sim de completude: se o processo terminar sem uma retomada dos três registros antigos, o pipeline poderá produzir um término formal com uma matriz batch incompleta.

Para considerar o experimento cientificamente fechado, é necessário concluir ou refazer Fashion-MNIST b32/b64 e MNIST b128, executar os 26 jobs ainda não iniciados, finalizar a quantização e obter os resultados de ativação do segundo Mac.

## Evidências locais

- [Configuração publicada](../configs/controlled-augmentation2-mac-m4-aug05.yaml)
- [Estado dos gates](../outputs/controlled-augmentation2-mac-m4-aug05/pipeline-status.json)
- [Resumo de evidências do snapshot](status_treinamento_2026-09-02.sources.json)
- [Resultados batch completos](../outputs/controlled-augmentation2-mac-m4-aug05/batch/)
- [Smoke tests completos](../outputs/controlled-augmentation2-mac-m4-aug05/smoke/)

Os resultados publicados incluem métricas, históricos, previsões, matrizes de confusão e telemetria (`summary.json`, `environment.json` e `samples.csv`) das 15 execuções concluídas. O log vivo do pipeline, as três execuções batch sem resultado final e os modelos/checkpoints binários não foram publicados: o treinamento continua rodando e esses arquivos ainda podem mudar ou são desnecessariamente pesados para versionamento.

*Relatório gerado em Markdown a partir de leitura somente dos processos, checkpoints, métricas, manifestos e telemetria locais. Nenhum processo de treinamento foi interrompido ou alterado.*
