# Snapshot de campanhas de ativações e quantização

Este diretório reúne evidências portáteis para as campanhas executadas no Mac M4. O relatório visual executado está em [notebooks/activation_quantization_full_report.ipynb](../../notebooks/activation_quantization_full_report.ipynb).

## Cobertura

- 27 treinos de ativação: nove datasets × ReLU, Sigmoid e Softmax; seed 42; 100 épocas por run; augmentation `extra_fraction=0,5`.
- 18 treinos para comparação de precisão: nove datasets × FP32 e FP16; seed 42; 100 épocas; batch 256 e augmentation `extra_fraction=0,5`.
- Nove avaliações adicionais de INT8 PTQ/LiteRT. São resultados de conversão e inferência dos modelos, não treinamentos INT8/QAT.
- 45 históricos de treino × 100 épocas; métricas de teste por classe; métricas e resumos de uso de hardware por run.

## Arquivos

- `data/runs.csv`: um registro para cada um dos 54 resultados (45 treinos + 9 PTQ), com protocolo, métricas finais, tempos e estatísticas resumidas de CPU/GPU/memória.
- `data/epoch_history.csv`: métricas de treino/validação por época para os 45 treinos completos.
- `data/per_class_metrics.csv`: precisão, recall, F1 e suporte por classe para todos os resultados avaliados.
- `data/snapshot_manifest.json`: contagens e escopo do snapshot.
- `../../scripts/export_activation_quantization_evidence.py`: exporta estes arquivos a partir dos diretórios locais de execução.

As telemetrias foram resumidas a cada cinco segundos durante as execuções. Os valores de GPU usam o backend registrado pelo run; no Apple M4 a memória é unificada e não corresponde a VRAM dedicada. Energia/potência não foi medida. Uma seed por condição não permite estimar variabilidade estatística entre execuções.

## Regeneração dos dados no Mac que contém os outputs

Os diretórios fonte sob `outputs/` são ignorados pelo Git e não fazem parte deste snapshot. Para reconstruir os CSVs, use o checkout Mac que ainda contém esses diretórios:

```bash
.venv-mac/bin/python scripts/export_activation_quantization_evidence.py
```

Para reexecutar o notebook a partir da raiz do repositório, com as dependências de relatório instaladas:

```bash
.venv-mac/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/activation_quantization_full_report.ipynb
```

O snapshot contém evidência numérica sanitizada e os gráficos/tabelas do notebook. Ele não copia checkpoints, previsões/logits ou amostras individuais de telemetria.
