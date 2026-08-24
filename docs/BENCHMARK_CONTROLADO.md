# Benchmark controlado em três fases

Este protocolo executa **MNIST, Fashion-MNIST, KMNIST, EMNIST Balanced,
CIFAR-10, CIFAR-100 coarse, SVHN, GTSRB e FER2013**, sem
paralelismo, usando seed 42, split estratificado 70/15/15, `all_raw`,
augmentation padrão e `extra_fraction: 2.0`.

## Preparação no macOS 15 / Apple M4

O perfil nativo do Mac usa TensorFlow 2.18.1 com `tensorflow-metal` 1.2.0 e
mantém todos os artefatos separados do WSL em
`outputs/controlled-augmentation2-mac-m4/`:

```bash
brew install python@3.10
/opt/homebrew/bin/python3.10 -m venv .venv-mac
.venv-mac/bin/python -m pip install --upgrade pip setuptools wheel
.venv-mac/bin/python -m pip install -r requirements/macos-metal.txt
.venv-mac/bin/python scripts/fetch_datasets.py --all
```

Valide o backend antes da matriz:

```bash
.venv-mac/bin/python -m tcc_benchmark preflight \
  --require-tensorflow --require-gpu \
  --output-root outputs/controlled-augmentation2-mac-m4/preflight \
  --data-path datasets
```

O comando deve mostrar uma GPU TensorFlow `GPU:0`, suporte Metal e o backend
de telemetria `apple-metal-ioreg`. O monitor não chama CUDA, NVML ou
`nvidia-smi` no macOS.

Na auditoria do protocolo, duplicatas exatas com rótulos conflitantes são
mantidas no JSON como avisos documentados, pois aparecem naturalmente em
algumas bases públicas. Corrupção, arquivo ausente, rótulo inválido ou forma
de imagem inválida continuam bloqueando as fases seguintes.

## Único comando de execução

```bash
set -o pipefail
PYTHONUNBUFFERED=1 caffeinate -dimsu \
  .venv-mac/bin/python scripts/run_controlled_pipeline.py \
  --suite configs/controlled-augmentation2-mac-m4.yaml \
  --registry configs/datasets.yaml \
  --output-root outputs/controlled-augmentation2-mac-m4 \
  2>&1 | tee -a outputs/controlled-augmentation2-mac-m4/logs/pipeline.log
```

O comando executa cada run uma por vez, sem exigir que Codex permaneça ativo.
Ele usa 100 épocas completas e grava progresso em
`outputs/controlled-augmentation2-mac-m4/pipeline-status.json`. Em caso de erro,
corrija o problema e retome apenas o que falta:

```bash
set -o pipefail
PYTHONUNBUFFERED=1 caffeinate -dimsu \
  .venv-mac/bin/python scripts/run_controlled_pipeline.py \
  --suite configs/controlled-augmentation2-mac-m4.yaml \
  --registry configs/datasets.yaml \
  --output-root outputs/controlled-augmentation2-mac-m4 \
  --resume 2>&1 | tee -a outputs/controlled-augmentation2-mac-m4/logs/pipeline.log
```

## Seleção por fase

1. **Batch:** 32, 64, 128 e 256 para cada dataset. Vence o maior Macro-F1;
   empate usa a menor duração média por época.
2. **Quantização:** FP32, FP16 e INT8-PTQ usando o batch vencedor. Variantes
   até 1 ponto percentual abaixo do Macro-F1 FP32 concorrem por menor latência
   LiteRT (CPU); os desempates são menor arquivo, maior throughput e maior
   Macro-F1.
3. **Ativação:** ReLU, sigmoide e softmax em todas as ativações internas, com
   batch e quantização vencedores. A saída permanece em logits para manter a
   mesma loss e métrica entre variantes.

Cada fase produz CSV, JSON, HTML e PNG em
`outputs/controlled-augmentation2-mac-m4/reports/`, além de relatórios por
dataset. O CINIC-10 não participa do downloader, auditoria, matriz ou relatório.
Use somente para o protocolo novo: os resultados históricos de KMNIST não têm
augmentation 2.0 e não são comparáveis.
