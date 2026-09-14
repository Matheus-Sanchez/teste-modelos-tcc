# Progresso da quantização, previsão e uso de hardware — 14/09/2026 (atualizado)

## Escopo

Este relatório considera **somente** a execução solicitada de quantização em `outputs/controlled-quantization-fast-mac-m4`. As execuções batch em `outputs/controlled-augmentation2-mac-m4-aug05` e as ativações em `outputs/controlled-augmentation05-activations-mac2` são históricas e não entram nos números abaixo.

## Resumo executivo

- A matriz de quantização tem **18 jobs planejados**: FP32 e FP16 para cada um dos 9 datasets.
- **7 jobs foram concluídos**: MNIST FP32/FP16, Fashion-MNIST FP32/FP16, KMNIST FP32/FP16 e EMNIST Balanced FP32.
- O erro anterior da MNIST FP16 (`Broken pipe`) foi recuperado com sucesso na retomada.
- **1 job está marcado como `running`, mas está stale**: EMNIST Balanced FP16 parou na época 66/100; não há artefatos novos desde 12/09.
- **10 jobs ainda não foram iniciados**.
- Não há evidência atual de processo ativo; a rodada precisa ser retomada.
- O `pipeline-status.json` global ainda conserva o estado histórico `failed` da tentativa anterior e não deve ser usado sozinho para classificar a retomada.

## O que já foi feito

### Gates da rodada

Os gates da execução foram concluídos:

- preflight do Apple M4 e TensorFlow/Metal;
- auditoria dos 9 datasets;
- smoke tests.

Os smoke tests servem para validar capacidade e configuração. Eles não são contabilizados como resultados da matriz de quantização.

### Quantização MNIST FP32

O job MNIST FP32 foi concluído com:

- 100 épocas registradas;
- avaliação final presente;
- status `completed`;
- aproximadamente 2,40 horas de telemetria;
- pressão térmica `nominal`.

### Recuperação da quantização MNIST FP16

O primeiro intento do job MNIST FP16 executou 59 épocas e falhou em 10/09 às 17:02, com:

```text
[Errno 32] Broken pipe
```

Após a retomada com a GPU detectada, o mesmo job foi concluído em **11/09 às 03:22**, com 100 épocas e avaliação final. A falha anterior não permanece como falha ativa da matriz.

### Jobs adicionais concluídos

Também foram concluídos:

- Fashion-MNIST FP32: aproximadamente 2,39 horas;
- Fashion-MNIST FP16: aproximadamente 4,19 horas;
- KMNIST FP32: aproximadamente 2,39 horas;
- KMNIST FP16: concluída em 11/09, com avaliação final;
- EMNIST Balanced FP32: concluída em 11/09, com avaliação final.

Os sete jobs concluídos têm status `completed`, métricas finais e telemetria.

### Último job parcial e estado atual

EMNIST Balanced FP16 está marcado como `running`, mas a última época persistida é a **66/100**, com o arquivo de métricas modificado em **12/09**. O status não foi finalizado nem atualizado desde então; por isso o job é tratado como parcial/stale, não como execução ativa confirmada. O ritmo observado foi de aproximadamente **297 segundos por época**, indicando cerca de **2,8 horas de treino restantes** quando for retomado, além da avaliação final.

## Estado atual da matriz

| Estado | Quantidade |
|---|---:|
| FP32 concluída | 4 |
| FP16 concluída | 3 |
| Parcial/stale a retomar | 1 |
| Falha ativa | 0 |
| Jobs ainda não iniciados | 10 |
| **Jobs planejados** | **18** |

Os 10 jobs ainda não iniciados correspondem às variantes FP32/FP16 dos cinco datasets restantes depois de MNIST, Fashion-MNIST, KMNIST e EMNIST Balanced.

## Previsão de duração

### Base observada

Os sete jobs concluídos levaram entre aproximadamente 1,7 e 4,2 horas. O EMNIST Balanced FP16 parcial tem cerca de 2,8 horas de treino restantes pela velocidade observada. Como ainda não há tempos observados para os cinco datasets restantes, a previsão continua incerta.

| Trabalho restante | Estimativa |
|---|---:|
| Retomar e finalizar EMNIST Balanced FP16 | ~2,8 h |
| 10 jobs ainda não iniciados | 25–50 h |
| **Total restante estimado** | **31–56 h** |

Em execução serial contínua, isso representa aproximadamente **1,2–2,2 dias**, sem contar o tempo parado até a retomada. A janela é indicativa: pode aumentar se algum dataset exigir mais tempo de preparação ou se ocorrer nova falha de execução.

### Cenário central

Usando aproximadamente 3 horas para finalizar EMNIST Balanced FP16 e 3,5 horas por job restante, o cenário central é de **38–40 horas**, ou cerca de **1,6 dia de execução contínua após a retomada**.

## Uso do hardware

Os números abaixo vêm da telemetria dos sete jobs concluídos. O job EMNIST Balanced FP16 ainda não tem resumo final de telemetria. Os valores são históricos da rodada, não uma leitura instantânea do computador.

| Job | GPU média | Pico de memória GPU compartilhada | CPU do processo média | RSS máximo | Pressão térmica |
|---|---:|---:|---:|---:|---|
| MNIST FP32 | ~91,8% | ~1,65 GiB | ~92,2% | ~4,94 GiB | nominal |
| MNIST FP16 concluído | ~82,3% | ~1,57 GiB | — | ~3,89 GiB | nominal |
| Fashion-MNIST FP32/FP16 | ~87% | ~1,15 GiB | — | ~4,21 GiB | nominal |
| KMNIST FP32/FP16 | ~91,9% | ~1,13 GiB | — | ~4,27 GiB | nominal |
| EMNIST Balanced FP32 | — | — | — | — | nominal |

Outros pontos observados:

- O hardware é um Apple M4 com memória unificada; não há VRAM dedicada separada.
- A alocação do driver Metal permaneceu compatível com a rodada anterior, sem sinal de saturação de memória unificada.
- A pressão térmica permaneceu `nominal` nos cinco jobs concluídos.
- O espaço livre no volume permaneceu suficiente; não há evidência de falta de espaço como causa da falha FP16 anterior.
- O backend de telemetria não fornece temperatura e potência detalhadas de forma consistente.

## Próximas ações necessárias

1. Retomar EMNIST Balanced FP16 com `--resume` e confirmar novas épocas no arquivo de métricas.
2. Continuar os 10 jobs restantes com `--resume`.
3. Validar os 18 `status.json`, métricas finais, artefatos e telemetria.

## Evidências locais

- [Status global da quantização](../outputs/controlled-quantization-fast-mac-m4/pipeline-status.json)
- [Log da rodada](../outputs/controlled-quantization-fast-mac-m4/logs/pipeline.log)
- [Configuração da quantização](../configs/controlled-quantization-fast-mac-m4.yaml)
- [Saída dos jobs de quantização](../outputs/controlled-quantization-fast-mac-m4/quantization/)
- [Comando documentado para quantização isolada](../README.md)

*Relatório atualizado em 14/09/2026 por leitura somente dos artefatos da rodada de quantização. Nenhum processo ou arquivo de resultado foi alterado.*
