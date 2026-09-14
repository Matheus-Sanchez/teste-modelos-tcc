# Relatório de auditoria do treinamento — 08/09/2026

## Resumo executivo

**Estado no momento da auditoria:** o treinamento está rodando. Há um supervisor ativo desde **08/09/2026 13:34:05 BRT**, com um processo TensorFlow executando `emnist_balanced`, batch 64.

Houve uma interrupção/parada anterior, mas ela **não foi registrada como evento formal** pelo pipeline. A última evidência do processo antigo foi o telemetry de **04/09/2026 12:35:07 BRT**, no treino incompleto `emnist_balanced/batch-032`; a execução atual começou em **08/09/2026 13:34:05 BRT**. Portanto, a janela comprovável da parada é:

> **04/09/2026 12:35:07 BRT — 08/09/2026 13:34:05 BRT**

Os logs não permitem afirmar se a causa foi encerramento manual, suspensão/reinício do macOS, queda do processo ou outro evento externo. Não há `KeyboardInterrupt`, `SIGTERM`, `SIGKILL`, `Killed`, `Traceback` ou mensagem de falha fatal no `pipeline.log`.

## Situação da matriz de batch

O experimento possui 9 datasets × 4 batches = **36 células**.

| Situação | Quantidade | Interpretação |
|---|---:|---|
| Concluídas | 9 | Possuem métricas de teste e resumo de hardware |
| Em execução real | 1 | EMNIST balanced, batch 64 |
| Marcadas `running`, mas órfãs | 4 | Não há processo correspondente; não possuem resultado final |
| Ainda não iniciadas | 22 | Sem `status.json` de execução concluída |
| **Total** | **36** | |

Os quatro estados órfãos são `emnist_balanced/batch-032`, `fashion_mnist/batch-032`, `fashion_mnist/batch-064` e `mnist/batch-128`. Eles permanecem com `status: running`, mas seus arquivos não são atualizados desde 30/08–04/09. Isso é um problema de estado persistido, não evidência de que ainda estejam treinando.

## Processo ativo

Comprovantes observados:

- Supervisor: `scripts/run_controlled_pipeline.py`, iniciado às **13:34:05 BRT**.
- Filho TensorFlow: `tcc_benchmark run --dataset emnist_balanced`, batch 64, iniciado às **13:35:34 BRT**.
- Progresso no `pipeline.log` durante a auditoria: aproximadamente **1059/2160 steps** do primeiro epoch, a cerca de **249 ms/step**.
- Estimativa do primeiro epoch: aproximadamente 9 minutos; a configuração prevê 100 epochs.
- O comando atual contém `--skip-activation`, mas **não contém `--skip-quantization`**. Assim, depois da matriz de batch o supervisor tentará executar a quantização.
- O comando atual não contém `--resume`; portanto, os quatro estados órfãos não estão sendo retomados por checkpoint nesta execução. Eles estão sendo ignorados como `skip_needs_resume`, enquanto as células pendentes são executadas.

## Resultados de batch já concluídos

| Dataset | Batch | Macro-F1 de teste | Média s/epoch | Treino |
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

Esses resultados são válidos como resultados individuais já finalizados. A comparação global ainda não está pronta porque os datasets restantes não terminaram e a seleção final de batch só é escrita ao final da fase.

## O que foi capturado do hardware

A coleta está funcionando:

- Os runs concluídos têm `telemetry/summary.json` e milhares de amostras.
- O run ativo possui `telemetry/samples.csv` sendo atualizado durante o treino.
- A amostra atual do EMNIST batch 64 registrou: CPU do processo em torno de 101%, RAM do sistema em 68,7%, RSS do processo em aproximadamente 1,59 GiB, GPU Apple M4 em 77% e pressão térmica `nominal`.
- Nos runs concluídos, o pico de utilização da GPU ficou entre 94% e 100%, com pressão térmica `nominal`.
- Os campos coletados incluem CPU, RAM, RSS, utilização da GPU, memória do driver Metal, memória de GPU/sistema, temperatura quando disponível, potência quando disponível e pressão térmica.

Não há indicação de perda de telemetry nos runs concluídos. Nos quatro runs órfãos, há telemetry parcial, mas não há resumo final nem métrica de teste; por isso eles não podem ser tratados como resultados válidos.

## Auditoria dos dados

Os 9 datasets passaram pela auditoria com **0 erros de leitura/estrutura**:

| Dataset | Exemplos auditados | Erros | Observações |
|---|---:|---:|---|
| MNIST | 70.000 | 0 | Sem alerta |
| Fashion-MNIST | 70.000 | 0 | Sem alerta |
| KMNIST | 70.000 | 0 | Sem alerta |
| EMNIST balanced | 131.600 | 0 | Duplicatas exatas dentro do conjunto |
| CIFAR-10 | 60.000 | 0 | Sem alerta |
| CIFAR-100 coarse | 60.000 | 0 | Duplicatas; há duplicata com rótulos conflitantes |
| SVHN | 99.289 | 0 | Duplicata com rótulos conflitantes |
| GTSRB | 39.270 | 0 | Sem alerta |
| FER2013 | 35.887 | 0 | Muitas duplicatas; há rótulos conflitantes |

Os alertas de duplicatas foram aceitos pelo próprio pipeline como avisos documentados. Eles não interromperam o treino, mas devem ser mencionados na análise final de qualidade dos dados, principalmente para CIFAR-100 coarse, SVHN e FER2013.

## Evidência da interrupção anterior

| Evidência | Horário BRT | Leitura |
|---|---|---|
| Último checkpoint persistido do EMNIST batch 32 | 04/09 12:22:52 | Epoch 66; o treino não chegou ao resultado final |
| Última amostra de hardware do mesmo run | 04/09 12:35:07 | Última evidência de processo ativo daquele run |
| Novo supervisor | 08/09 13:34:05 | Nova execução começou |
| Novo filho EMNIST batch 64 | 08/09 13:35:34 | Execução atual confirmada |

O `status.json` antigo continua em `running` e o `state_events.jsonl` termina em `pending → running`, sem transição para `interrupted`, `failed` ou `completed`. O `pipeline-status.json` também ainda não possui status final; isso é esperado enquanto o supervisor está em execução.

O `OUT_OF_RANGE: End of sequence` encontrado no log é a mensagem normal do TensorFlow ao alcançar o fim de um dataset. Também há um aviso de compatibilidade sobre `use_unbounded_threadpool`. Nenhum dos dois é, isoladamente, evidência de interrupção.

## Previsão de término

### Fase de batch

Para o EMNIST batch 64, a velocidade observada indica aproximadamente **15 horas para os 100 epochs**, antes da avaliação final. A fase completa ainda precisa executar a célula atual e cerca de 22 células não iniciadas, incluindo os datasets que ainda não possuem nenhuma medição de tempo neste Mac.

Estimativa de baixa a média confiança, mantendo execução serial e sem novas paradas:

| Fase | Previsão | Confiança | Motivo |
|---|---|---|---|
| EMNIST batch 64 atual | 09/09/2026, aproximadamente 04:00–06:00 | Média | Baseada na velocidade live de 249 ms/step |
| Toda a matriz de batch | 19–27/09/2026 | Baixa/média | 22 células ainda não têm duração medida |
| Quantização prevista no comando atual | +8–15 dias após batch | Baixa | São 18 novos treinos (FP32/FP16) + 9 conversões/benchmarks INT8 |
| **Batch + quantização, sem ativações** | **27/09–12/10/2026** | **Baixa** | Depende principalmente de CIFAR, SVHN, GTSRB, FER2013 e dos estados órfãos |

Essa previsão assume que o processo atual continue ligado, com `caffeinate`, sem mudança de configuração e sem falha externa. A faixa é deliberadamente ampla porque os datasets de imagem colorida e o GTSRB ainda não tiveram um run completo neste output root.

## O que falta

1. Finalizar o EMNIST batch 64 atual.
2. Executar as 22 células de batch ainda pendentes.
3. Decidir o tratamento dos quatro runs órfãos: retomá-los explicitamente com `--resume` ou marcá-los como descartados antes de consolidar a matriz. Eles não têm resultado final.
4. Gerar a seleção global de batches e o relatório final da fase batch.
5. Executar a quantização, porque ela não foi desativada no comando atual.
6. Executar as ativações no outro Mac, caso essa fase ainda faça parte do experimento; nesta execução local elas estão explicitamente desativadas.

## Arquivos auditados

- `outputs/controlled-augmentation2-mac-m4-aug05/logs/pipeline.log`
- `outputs/controlled-augmentation2-mac-m4-aug05/pipeline-status.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/status.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/logs/state_events.jsonl`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/telemetry/samples.csv`
- `outputs/controlled-augmentation2-mac-m4-aug05/batch/**/summary.json`
- `outputs/controlled-augmentation2-mac-m4-aug05/audit/*/audit/audit.json`

**Conclusão:** o treino não está parado agora; ele está ativo. A execução anterior parou entre 04/09 12:35:07 e 08/09 13:34:05 BRT sem deixar causa explícita nos logs. Os dados, métricas e telemetry dos runs concluídos estão sendo salvos; o que ainda não está completo são os resultados das células restantes, os resumos finais da matriz, a quantização e as ativações.
