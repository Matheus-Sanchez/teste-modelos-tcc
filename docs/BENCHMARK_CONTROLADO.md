# Benchmark controlado em três fases

Este protocolo executa **MNIST, Fashion-MNIST, KMNIST, EMNIST Balanced,
CIFAR-10, CIFAR-100 coarse, SVHN, GTSRB e FER2013**, sem
paralelismo, usando seed 42, split estratificado 70/15/15, `all_raw`,
augmentation padrão e `extra_fraction: 2.0`.

## Preparação necessária

O computador atual ainda não possui uma distribuição Ubuntu WSL utilizável.
Abra PowerShell como administrador e instale-a uma única vez:

```powershell
wsl --install -d Ubuntu-22.04
```

Reinicie se o Windows solicitar. No Ubuntu, abra o repositório pela montagem
`/mnt/c/.../teste-modelos-tcc`, então execute:

```bash
bash scripts/bootstrap_controlled_benchmark.sh
```

O bootstrap instala Python 3.10 e TensorFlow GPU, baixa todas as bases
necessárias e executa o preflight. O treino só deve começar se o preflight
confirmar TensorFlow e uma GPU visível.

## Único comando de execução

```bash
source .venv/bin/activate
PYTHON_BIN="$PWD/.venv/bin/python" scripts/wsl-gpu-env.sh \
  "$PWD/.venv/bin/python" scripts/run_controlled_pipeline.py
```

O comando executa cada run uma por vez, sem exigir que Codex permaneça ativo.
Ele usa 100 épocas completas e grava progresso em
`outputs/controlled-augmentation2/pipeline-status.json`. Em caso de erro,
corrija o problema e retome apenas o que falta:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" scripts/wsl-gpu-env.sh \
  "$PWD/.venv/bin/python" scripts/run_controlled_pipeline.py --resume
```

## Seleção por fase

1. **Batch:** 32, 64, 128 e 256 para cada dataset. Vence o maior Macro-F1;
   empate usa a menor duração média por época.
2. **Quantização:** FP32, FP16 e INT8-PTQ usando o batch vencedor. Variantes
   até 1 ponto percentual abaixo do Macro-F1 FP32 concorrem por menor latência
   LiteRT (CPU); depois, menor arquivo.
3. **Ativação:** ReLU, sigmoide e softmax em todas as ativações internas, com
   batch e quantização vencedores. A saída permanece em logits para manter a
   mesma loss e métrica entre variantes.

Cada fase produz CSV, JSON, HTML e PNG em `outputs/controlled-augmentation2/reports/`.
Use somente para o protocolo novo: os resultados históricos de KMNIST não têm
augmentation 2.0 e não são comparáveis.
