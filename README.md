# TCC Dataset Benchmark

Suíte local e retomável para comparar o comportamento da mesma CNN treinada do
zero em nove datasets de classificação de imagens. Os dados ficam somente na
subpasta `datasets/` do projeto e não são versionados pelo Git.

Para o mapa de chamadas, explicação arquivo a arquivo e guia de alterações
manuais, leia [docs/GUIA_TECNICO.md](docs/GUIA_TECNICO.md).

## Escopo fixo

- Datasets: MNIST, Fashion-MNIST, KMNIST, EMNIST Balanced, CIFAR-10,
  CIFAR-100 (superclasses), SVHN, GTSRB e FER2013.
- O perfil final atual produz 36 runs: uma seed (42), normalização z-score e
  quatro modos de balanceamento para cada dataset.
- Divisão determinística nova de 70/15/15; validação e teste nunca recebem
  oversampling, undersampling ou pesos.
- CNN herdada apenas em topologia, sem pesos pré-treinados. Entradas têm 64 px,
  exceto GTSRB com 128 px; canais permanecem nativos.
- No perfil WSL, o cache de pré-processamento é limitado a 2 GiB e os buffers
  de shuffle dividem um orçamento total de 1 GiB, para preservar a RAM do host.

## Instalação

Use Python 3.10. No WSL com GPU:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/wsl-gpu.txt
```

No WSL, execute os comandos de treino pelo invólucro abaixo. Ele torna visíveis
as bibliotecas CUDA instaladas pelo extra `tensorflow[and-cuda]` e ativa o
crescimento progressivo da VRAM:

```bash
bash scripts/wsl-gpu-env.sh tcc-benchmark preflight
bash scripts/wsl-gpu-env.sh tcc-benchmark run --dataset mnist
```

Para smoke tests no Windows, use PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\windows-smoke.txt
```

O preflight confirma TensorFlow, GPU, driver, espaço em disco e dependências
de telemetria antes do treino. O perfil WSL usa TensorFlow 2.21 com o extra
`and-cuda`; o perfil Windows é deliberadamente CPU-first.

## Dados locais

Baixe os dez conjuntos para o layout padrão do projeto:

```powershell
python scripts\fetch_datasets.py --all
```

O comando cria `datasets/<nome-do-dataset>/` para cada base e grava um
`SOURCE.json` com a URL, checksum quando publicado e data de obtenção. O
registro padrão `configs/datasets.yaml` já aponta automaticamente para esses
caminhos, portanto não é preciso passar `--registry` nos comandos usuais.

Para instalar somente uma base, use por exemplo
`python scripts\fetch_datasets.py --dataset mnist`. O baixador é deliberado:
treino, auditoria e smoke test nunca fazem downloads por conta própria.

O FER2013 é obtido de um espelho público documentado no respectivo
`SOURCE.json`; confira os termos de uso e a procedência antes de publicar ou
redistribuir os dados. Para usar cópias externas, copie
`configs/datasets.example.yaml`, ajuste os caminhos e informe esse YAML com
`--registry`.

Os adaptadores aceitam também as estruturas locais comuns:

| Dataset | Conteúdo esperado |
|---|---|
| MNIST/Fashion-MNIST | arquivos IDX (`*-images-idx3-ubyte[.gz]`, rótulos correspondentes) ou `.npz` local |
| KMNIST | arquivos `.npz` oficiais (`kmnist-*-imgs/labels.npz`) |
| EMNIST Balanced | arquivos IDX gzip/sem gzip da variante `balanced` |
| CIFAR-10/100 | diretórios oficiais `cifar-*-batches-py` e `cifar-100-python` |
| CINIC-10 | `train`, `valid` e `test`, cada qual com subpastas por classe |
| SVHN | `train_32x32.mat` e `test_32x32.mat` |
| GTSRB | estrutura oficial por classes ou layout CSV comum (`Train.csv`/`Test.csv`) |
| FER2013 | arquivo `fer2013.csv` com colunas `emotion` e `pixels` |

O comando `audit` informa exatamente quais arquivos estão ausentes, sem tentar
baixá-los.

## Comandos

```bash
# Inspeciona a integridade de uma base local (arquivos, rótulos, duplicatas).
tcc-benchmark audit --dataset mnist

# Gera a lista de jobs sem treinar.
tcc-benchmark run --dataset mnist --dry-run

# Executa as 40 runs de uma base; use --resume após uma interrupção.
tcc-benchmark run --dataset mnist
tcc-benchmark resume --dataset mnist

# Executa datasets configurados sequencialmente, sem compartilhar GPU.
tcc-benchmark run --all

# Teste curto de capacidade: 5 épocas, seed 42, normalização [0,1] e all_raw.
# No WSL, use o invólucro de GPU descrito acima.
bash scripts/wsl-gpu-env.sh tcc-benchmark run --all --suite configs/gpu-memory-check.yaml

# Matriz final: 100 épocas por run, todas as normalizações, balanceamentos e seeds.
bash scripts/wsl-gpu-env.sh tcc-benchmark run --all --suite configs/full-100-epochs.yaml

# Perfil otimizado com datasets no filesystem nativo do WSL.
bash scripts/wsl-gpu-env.sh tcc-benchmark run --all --suite configs/full-100-epochs-wsl-optimized.yaml --registry configs/datasets.wsl.yaml

# Encadeia automaticamente o fim do teste curto, o relatório consolidado e a
# matriz de 100 épocas. Não promove a matriz se qualquer run curta falhar.
bash scripts/wsl-gpu-env.sh python scripts/supervise_benchmarks.py

# Benchmark KMNIST de quantização: FP32, FP16, INT8-QAT, INT4-QAT e INT8-PTQ.
# Cada variante fica em uma saída isolada e --resume continua somente o que falta.
bash scripts/wsl-gpu-env.sh python scripts/run_kmnist_quantization_benchmark.py --resume

# Recria somente a consolidação HTML/CSV/JSON/PNG de uma base.
tcc-benchmark report --output-root artifacts --dataset mnist

# Validação curta em CPU/GPU com amostra local, sem contaminar os resultados.
tcc-benchmark smoke --dataset mnist
```

Use `--suite configs/suite.yaml` para trocar somente a configuração declarada.
Não há redução automática de batch, resolução ou épocas quando ocorre OOM: a
execução é marcada como falha e os parâmetros permanecem comparáveis.

O treino usa a política Keras `mixed_float16`: convoluções e ativações calculam
em FP16 para reduzir o uso de VRAM, enquanto pesos, softmax final e loss ficam
em FP32 para estabilidade numérica. As imagens não são carregadas inteiramente
na GPU; a pipeline envia batches conforme `suite.training.batch_size`.

Cada linha de `checkpoints/epoch_metrics.csv` inclui
`train_examples_per_second`: exemplos de treino, incluindo os extras de
augmentação, divididos pelo tempo total da época. Para comparar batches,
compare essa métrica a partir da segunda época, com o mesmo dataset e protocolo.

O perfil `kmnist-quantization.yaml` fixa KMNIST em `all_raw`, sem augmentação,
split 70/15/15, seed 42, batch 256 e 50 épocas. FP32, FP16 e as duas variantes
QAT são treinadas separadamente; INT8-PTQ é convertido do checkpoint FP32. O
INT4-QAT é fake quantization W4A8 para pesquisa: seus números de tamanho físico
e desempenho são explicitamente rotulados como estimados/emulados.

## Artefatos

## Backup consolidado e organização das saídas

O backup consolidado do projeto usa uma única raiz no HDD:

```text
G:\repos\teste-modelos-tcc\
```

O código e as configurações vêm de `C:\source\repos\teste-modelos-tcc`; os
resultados externos vêm de `E:\tcc-benchmark\outputs`. Ambos devem ser
preservados dentro dessa mesma raiz, reunindo os resultados em `outputs/` sem
alterar o nome das pastas de experimento. `node_modules/` não faz parte do
backup, pois pode ser recriado a partir dos arquivos de dependência.

| Pasta | Conteúdo |
|---|---|
| `outputs/` | Runs, checkpoints, telemetria, relatórios e análises de experimentos. |
| `output/` | Entregáveis portáteis, incluindo relatórios PDF. |
| `artifacts/` | Artefatos auxiliares gerados por ferramentas do projeto. |
| `docs/` | Guias técnicos e documentação do backup. |

Veja [docs/BACKUP_ESTRUTURA.md](docs/BACKUP_ESTRUTURA.md) para o procedimento
de sincronização e [outputs/README.md](outputs/README.md) para a convenção das
saídas.

Cada run fica em:

```text
artifacts/<dataset>/runs/<dataset>__<normalization>__<balance_mode>__seed-<seed>/
```

Inclui `manifest.json`, `status.json`, `telemetry/environment.json`, histórico
por época, amostras e sumário de hardware, checkpoints `last.keras` e
`best.keras`, predições do teste, relatório por classe, matriz de confusão e
métricas finais. Em cada diretório há ainda `checkpoints/`, `logs/`,
`telemetry/` e `artifacts/`, para separar estado, recuperação e relatórios.
O diretório do dataset reúne `summary.csv`, `summary.json`, gráficos PNG e
`report.html`. O índice global é apenas operacional: não ordena datasets com
números de classes diferentes.

## Reprodutibilidade e retomada

O manifesto registra configuração, fingerprint do split, dados auditados,
ambiente e seed. O estado é escrito atomicamente. `last.keras` e
`BackupAndRestore` permitem retomar da última época concluída; `best.keras` é
selecionado por Macro-F1 de validação e é o modelo usado na avaliação final.

A normalização z-score aprende média e desvio por canal exclusivamente no
treino bruto já redimensionado. O balanceamento vem depois; isso impede que a
duplicação do oversampling altere a estatística de normalização.
