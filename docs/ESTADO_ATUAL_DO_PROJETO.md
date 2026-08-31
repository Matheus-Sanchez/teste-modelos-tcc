# Estado atual do projeto — TCC Dataset Benchmark

_Atualizado em 31 de agosto de 2026._

Este documento substitui, para fins de acompanhamento técnico, trechos
divergentes de documentos anteriores sobre quantidade de runs, batch padrão e
escopo de experimentos. Ele foi produzido a partir das configurações ativas,
do código em `src/`, dos relatórios de análise e dos artefatos locais em
`outputs/` e `output/`.

## Visão geral

O projeto é uma suíte local, retomável e observável para comparar uma mesma
CNN treinada do zero em tarefas de classificação de imagens. Não há uso de
pesos pré-treinados. Cada execução registra a configuração, o split, o
ambiente, telemetria, checkpoints, métricas por época e avaliação final.

Os nove datasets configurados no registro padrão são:

- MNIST, Fashion-MNIST, KMNIST e EMNIST Balanced;
- CIFAR-10 e CIFAR-100 coarse (20 superclasses);
- SVHN, GTSRB e FER2013.

`CINIC-10` aparece em textos históricos como adaptador suportado, mas **não
faz parte de `configs/datasets.yaml` nem da matriz ativa**. Portanto, não deve
ser contado como dataset do experimento atual.

## Perfil final configurado

O arquivo [configs/full-100-epochs.yaml](../configs/full-100-epochs.yaml) é a
referência da matriz final. Ele planeja **36 runs sequenciais**:

```text
9 datasets × 1 seed (42) × 1 normalização (z-score) × 4 modos de balanceamento
= 36 runs
```

| Aspecto | Configuração atual |
|---|---|
| Divisão dos dados | Estratificada e determinística: 70% treino, 15% validação, 15% teste |
| Normalização | `zscore`, ajustada somente no treino bruto redimensionado |
| Balanceamento | `all_raw`, `undersample`, `oversample` e `class_weight`; apenas no treino |
| Treino | Até 100 épocas, Adam com LR 0,0003 e `mixed_float16` |
| Batch | 512 |
| Imagem | 64×64; GTSRB em 128×128; canais originais preservados |
| Augmentação | `extra_fraction: 2.0` e transformações aplicadas somente no treino |
| Parada antecipada | Paciência 10 no Macro-F1 de validação |
| Seleção de checkpoint | Maior Macro-F1 de validação (`best.keras`) |

O limite de 100 épocas não garante que toda run chegue à época 100: a parada
antecipada pode encerrar uma run antes. A matriz não reduz batch, resolução ou
épocas automaticamente após um erro de memória; a run é marcada como falha
para preservar a comparabilidade.

## Arquitetura e protocolo de avaliação

A CNN legada, implementada em [src/tcc_benchmark/model.py](../src/tcc_benchmark/model.py),
tem cinco blocos de convoluções separáveis com GroupNormalization e pooling,
duas agregações globais (média e máximo), três dropouts e duas camadas densas
de 256 unidades antes da camada final de logits. A topologia é sempre criada
do zero.

O fluxo de cada run é:

```text
dados locais -> split estratificado -> estatísticas do treino -> balanceamento
do treino -> augmentação do treino -> fit e seleção por Macro-F1 de validação
-> avaliação única no teste -> artefatos, telemetria e relatórios
```

Validação e teste não recebem reamostragem, pesos de classe, augmentação ou
test-time augmentation. Essa separação evita que transformações do treino
contaminem as métricas de seleção ou de teste.

## Resultados já consolidados

### Grid controlado com augmentação 2.0

O arquivo [analysis/benchmark-continuation-metrics-2026-08-28.csv](../analysis/benchmark-continuation-metrics-2026-08-28.csv)
consolida a etapa de batch concluída para MNIST, Fashion-MNIST e KMNIST. Todas
as linhas abaixo usam seed 42, `unit_interval`, `all_raw`, 100 épocas e
`extra_fraction: 2.0`.

| Dataset | Batch vencedor | Macro-F1 | Tempo de treino | Observação |
|---|---:|---:|---:|---|
| MNIST | 32 | 0,9915 | 7,12 h | Batches maiores diminuíram o tempo, mas tiveram Macro-F1 levemente menor. |
| Fashion-MNIST | 32 | 0,9301 | 9,36 h | Batch 256 reduziu o tempo em 46,1%, com queda de 0,36 p.p. de Macro-F1. |
| KMNIST | 32 | 0,9910 | 8,71 h | Batch 256 reduziu o tempo em 43,1%, com queda de 0,30 p.p. de Macro-F1. |

Pelo critério do supervisor — maximizar Macro-F1 e usar a duração média da
época somente em empate exato — o batch 32 é a escolha preliminar para esses
três datasets. Isso **não é** uma decisão para os nove datasets: a etapa ainda
não foi concluída em todo o escopo.

No controle KMNIST de batch 32, retirar a augmentação reduziu o tempo de
treino de 8,71 h para 2,57 h, mas também reduziu o Macro-F1 de 0,9910 para
0,9866 (−0,44 p.p.). A comparação é válida apenas para esse par controlado:
mesmo dataset, seed, split, batch e 100 épocas.

### Quantização isolada em KMNIST

O experimento histórico em
`outputs/kmnist-quantization-2026-08-09/analysis/run_summary.csv` comparou
cinco variantes no KMNIST, batch 256, sem augmentação e por 50 épocas.

| Variante | Macro-F1 | Estado LiteRT | Interpretação |
|---|---:|---|---|
| FP32 | 0,9786 | Concluído | Referência de qualidade. |
| FP16 | 0,9730 | Falhou na conversão | Resultado de treino válido, mas não é artefato LiteRT utilizável neste registro. |
| INT8-QAT | 0,9772 | Concluído | Alternativa implantável com perda pequena frente ao FP32. |
| INT4-QAT | 0,9801 | Não exportado | Resultado emulado; não usar como tamanho ou desempenho físico. |
| INT8-PTQ | 0,9664 | Concluído | Implantável, porém com queda maior de Macro-F1. |

Esses valores pertencem a um protocolo específico; não devem ser comparados
diretamente com a matriz final de z-score com augmentação 2.0.

### Evidência histórica e artefatos brutos

- O relatório técnico histórico reúne 36 runs, 2.486 épocas e 88,86 horas de
  treino. A tabela individual histórica original não está montada neste
  ambiente; os 36 resultados disponíveis vêm de um snapshot de notebook.
- O batch sweep KMNIST sem augmentação reúne 32 runs e 1.600 épocas. Os CSVs
  consolidados estão em
  `outputs/kmnist-alldata-noaug-batch-sweep-2026-08-06/analysis-2026-08-09-delivery-final/`.
- Há 12 manifestos brutos de perfil em `outputs/remaining-ram-capped/`. Eles
  são evidência operacional, não substituem a matriz final de 36 runs.

## Execuções em andamento ou incompletas

`outputs/controlled-augmentation05-batch-activation/` contém uma tentativa
posterior com `extra_fraction: 0.5`. A configuração está em
`configs/controlled-augmentation05-batch-activation.yaml`. O
`pipeline-status.json` dessa pasta registra falha na etapa de batch para
EMNIST Balanced com batch 128; há arquivos de status individuais mais novos,
o que indica retomadas fora daquele ciclo do supervisor. Portanto, os seus
resultados não devem ser usados para eleger batch, quantização ou ativação até
que o supervisor seja retomado e os relatórios de fase sejam regenerados.

O diretório `outputs/controlled-augmentation2/` também é parcial: as métricas
consolidadas acima cobrem somente os três datasets com grid finalizado. Não há
evidência consolidada para afirmar vencedores nos demais datasets.

## Como reproduzir e retomar

O projeto requer Python `>=3.10,<3.11`. Para execução completa em GPU, use o
ambiente WSL e o invólucro que expõe as bibliotecas CUDA:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/wsl-gpu.txt

bash scripts/wsl-gpu-env.sh tcc-benchmark preflight --require-tensorflow
bash scripts/wsl-gpu-env.sh tcc-benchmark audit --dataset mnist
bash scripts/wsl-gpu-env.sh tcc-benchmark run --all \
  --suite configs/full-100-epochs.yaml
```

Para interromper e continuar uma matriz compatível:

```bash
bash scripts/wsl-gpu-env.sh tcc-benchmark resume --all \
  --suite configs/full-100-epochs.yaml
```

Antes de executar a matriz completa, faça um smoke test especialmente no
GTSRB, que é RGB e usa 128×128:

```bash
bash scripts/wsl-gpu-env.sh tcc-benchmark smoke --dataset gtsrb \
  --epochs 1 --examples-per-class 32 \
  --suite configs/full-100-epochs.yaml
```

Os datasets não são baixados durante auditoria, smoke test ou treino. Para
preparar o layout local padrão, execute previamente:

```bash
python scripts/fetch_datasets.py --all
```

## Estrutura dos resultados

Cada run é armazenada sob:

```text
<output_root>/<dataset>/runs/
  <dataset>__<normalization>__<balance_mode>__seed-<seed>/
```

Os principais arquivos são `manifest.json`, `status.json`,
`checkpoints/epoch_metrics.csv`, `checkpoints/best.keras`,
`checkpoints/last.keras`, `logs/training_summary.json`,
`telemetry/summary.json` e `artifacts/test_metrics.json`. O manifesto e o
fingerprint do split devem ser preservados junto aos resultados: eles são a
base para verificar se uma retomada ou comparação é compatível.

## Limites e próximos passos

1. Retomar o experimento controlado somente após identificar a causa da falha
   registrada e regenerar os relatórios de fase.
2. Não misturar métricas de protocolos diferentes (por exemplo, KMNIST sem
   augmentação, batch controlado com augmentação 2.0 e matriz final em
   z-score).
3. Quando uma nova matriz for iniciada, usar um `output_root` cujo nome reflita
   o batch configurado. A pasta atual contém `batch64` no nome histórico, mas
   o YAML ativo define batch 512.
4. Manter a limitação da matriz histórica explícita: sem o `run_metrics.csv`
   original, ela serve como snapshot consolidado, não como base para
   reconstruir cada run individual.

## Fontes verificadas

- [pyproject.toml](../pyproject.toml) e
  [src/tcc_benchmark/cli.py](../src/tcc_benchmark/cli.py): dependências e
  comandos disponíveis;
- [configs/datasets.yaml](../configs/datasets.yaml) e
  [configs/full-100-epochs.yaml](../configs/full-100-epochs.yaml): escopo e
  perfil final;
- [src/tcc_benchmark/model.py](../src/tcc_benchmark/model.py),
  [src/tcc_benchmark/data.py](../src/tcc_benchmark/data.py) e
  [src/tcc_benchmark/runner.py](../src/tcc_benchmark/runner.py): arquitetura,
  preparação e ciclo de execução;
- [analysis/benchmark-continuation-methodology-2026-08-28.md](../analysis/benchmark-continuation-methodology-2026-08-28.md)
  e seu CSV de métricas: grid controlado concluído;
- `outputs/` e `output/`: resultados, relatórios e entregáveis locais.
