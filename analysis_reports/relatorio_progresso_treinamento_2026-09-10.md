# Progresso do treinamento, previsão e uso de hardware — 10/09/2026

## Resumo executivo

O experimento controlado tem **34 de 81 execuções concluídas**, uma execução de quantização falhada e **46 células ainda sem conclusão**. A fase final de ativações já terminou: **27/27** execuções foram concluídas no segundo Mac. O trabalho pendente está concentrado no primeiro Mac: **30 treinos batch** e a quantização restante.

O último erro observado foi na quantização **MNIST FP16**, após 59 épocas, com `Broken pipe`. O registro indica que a falha ocorreu sem alterar batch, resolução ou hiperparâmetros; portanto, ela deve ser tratada como uma repetição isolada de pós-processamento/execução, não como evidência de que o modelo ou o treino batch falhou.

As previsões abaixo são estimativas operacionais. Com execução serial e os tempos observados, o encerramento do primeiro Mac tende a exigir **aproximadamente 12–21 dias de execução**, ou uma janela aproximada de **22/09 a 01/10**, dependendo do custo dos seis datasets ainda não medidos na fase batch e da repetição da quantização FP16.

## O que já foi feito — fatos observados

| Fase | Planejado | Concluído | Estado observado |
|---|---:|---:|---|
| Batch | 36 | 6 | Parcial; MNIST, Fashion-MNIST e KMNIST já têm resultados finais em algumas combinações |
| Quantização | 18 | 1 | MNIST FP32 concluída; MNIST FP16 falhou após 59 épocas |
| Ativações | 27 | 27 | Rodada final concluída, com ReLU, sigmoid e softmax nos 9 datasets |
| **Total** | **81** | **34** | **42% das execuções concluídas** |

As nove execuções `smoke` presentes nos diretórios de saída foram usadas como gates de validação e não entram na contagem das 81 execuções científicas.

### Execuções batch concluídas

As seis células batch finalizadas são:

- MNIST: batch 32, 64 e 256;
- Fashion-MNIST: batch 128 e 256;
- KMNIST: batch 32.

O tempo acumulado observado nesses seis treinos foi de aproximadamente **57,5 horas**, com 618 épocas registradas. Os registros possuem métricas, checkpoints e telemetria.

### Ativações concluídas

A rodada final `controlled-augmentation05-activations-mac2` foi concluída em **08/09**, com 27 execuções e 2.700 épocas. O tempo acumulado de telemetria foi de aproximadamente **104,4 horas**. A exportação LiteRT foi explicitamente ignorada nessa rodada (`litert_status: skipped`), o que não invalida os resultados de treinamento e avaliação das ativações.

### Quantização

MNIST FP32 foi concluída em aproximadamente **2,4 horas**. MNIST FP16 falhou em 10/09 com:

```text
[Errno 32] Broken pipe
```

Essa falha não possui avaliação final e precisa ser repetida. Os oito diretórios `smoke` da saída de quantização também não são resultados científicos da matriz.

## O que está acontecendo agora

Não há uma execução atual confiável indicada pelos artefatos. Existe um `status.json` com `status: running` na antiga tentativa de ativações Fashion-MNIST/softmax, mas ele não é atualizado desde **28/08**, registra apenas 45 épocas e não corresponde à rodada final de ativações, que já foi concluída. Portanto, ele deve ser tratado como registro antigo/stale até que um processo vivo seja confirmado.

O `pipeline-status.json` do batch não contém um fechamento atualizado da matriz, e o pipeline de quantização terminou como `failed` por causa da MNIST FP16. A leitura atual deve ser feita pelos `status.json` individuais e pelos artefatos por execução, não pelo status global isoladamente.

## O que falta fazer

1. Executar ou retomar as **30 células batch restantes** dos 36 planejados.
2. Repetir a quantização **MNIST FP16**.
3. Executar as **16 quantizações restantes** depois que suas células batch de origem estiverem disponíveis.
4. Reexecutar a auditoria de completude, validar checkpoints, métricas de teste, previsões e telemetria.
5. Consolidar o relatório final da matriz de 81 execuções, excluindo registros antigos `running`, smoke tests e artefatos incompletos.

## Previsão de duração

### Base usada

Os seis treinos batch concluídos consumiram 57,5 horas, ou cerca de 9,6 horas por execução em média. Como os datasets restantes incluem bases RGB e datasets potencialmente mais pesados, usei uma faixa mais larga para evitar transformar a média atual em uma promessa.

| Trabalho restante | Faixa estimada | Confiança |
|---|---:|---|
| 30 células batch | 240–450 h | Baixa a média |
| 16 quantizações novas | 38–60 h | Baixa |
| Repetição MNIST FP16 | 2–4 h | Média-baixa |
| **Total no primeiro Mac** | **280–514 h** | **Baixa** |

Em execução serial contínua, 280–514 horas equivalem a aproximadamente **12–21 dias**. Tomando 10/09 como data de referência, a faixa de calendário é aproximadamente **22/09 a 01/10**. Essa janela não inclui interrupções humanas, falhas adicionais, manutenção ou tempo extra para análise e publicação.

### Cenário central

Um cenário central razoável é concluir o restante do batch em cerca de **330 horas**, seguido de aproximadamente **45 horas** de quantização e repetição da falha, totalizando **375 horas**, ou cerca de **15,6 dias** de execução. Isso aponta para o fim do primeiro Mac por volta de **25–26/09**, se o processo permanecer serial e contínuo.

## Uso do hardware

Os valores abaixo são históricos, obtidos da telemetria salva em `telemetry/summary.json`; não representam uma leitura instantânea do computador em 10/09.

| Grupo observado | GPU média por execução | Pico de memória GPU compartilhada | CPU do processo | Pressão térmica |
|---|---:|---:|---:|---|
| 6 batch concluídos | ~41% | até ~1,56 GiB | ~79% | nominal |
| 1 quantização FP32 concluída | ~31% | até ~3,10 GiB | ~76% | nominal |
| 27 ativações finais | ~88% | até ~3,73 GiB | ~87% | nominal |

Observações importantes:

- O Mac usa memória unificada; os números são alocação/uso reportado pelo driver Metal, não uma VRAM dedicada independente.
- A telemetria das ativações mostra uso de GPU mais alto e sustentado que a dos treinos batch já concluídos.
- O RSS máximo observado ficou na ordem de **5,5 GiB** nos grupos mais pesados.
- O espaço livre observado nos runs concluídos ficou aproximadamente entre **260 e 302 GiB**, sem evidência de esgotamento de disco.
- O backend `apple-metal-ioreg` não fornece de forma consistente temperatura e potência detalhadas; a evidência disponível é a categoria de pressão térmica, que permaneceu `nominal` nos resumos consultados.

## Riscos e interpretação

- A matriz ainda não está cientificamente completa: resultados de 6 batch, 1 quantização e 27 ativações não substituem as 81 combinações planejadas.
- O registro antigo `running` de Fashion-MNIST/softmax não deve ser contado como execução ativa nem como resultado parcial atual.
- A falha FP16 de MNIST é de execução/conversão e precisa de repetição; ela não deve ser apresentada como falha de treinamento batch.
- A previsão tem baixa confiança para os datasets ainda não medidos no batch, pois a duração pode variar bastante com resolução, volume de dados e custo de pré-processamento.

## Evidências locais

- [Configuração do batch Mac M4](../configs/controlled-augmentation2-mac-m4-aug05.yaml)
- [Configuração da quantização Mac M4](../configs/controlled-quantization-fast-mac-m4.yaml)
- [Status global da quantização](../outputs/controlled-quantization-fast-mac-m4/pipeline-status.json)
- [Saída batch e seus `status.json`](../outputs/controlled-augmentation2-mac-m4-aug05/)
- [Rodada final de ativações](../outputs/controlled-augmentation05-activations-mac2/pipeline-status.json)
- [Telemetria por execução](../outputs/controlled-augmentation05-activations-mac2/activations/)

*Relatório construído a partir de leitura somente dos artefatos locais. Nenhum processo, checkpoint ou resultado foi alterado.*
