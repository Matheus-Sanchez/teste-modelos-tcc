# Testes de treinamento e benchmarks

Volte à [Central de resultados](INDEX.md).

## Perfis de treinamento completos

| Perfil | Local | Destaque |
| --- | --- | --- |
| Batch 32 | [Resultados](../../outputs/full-100-epochs-batch32/) | Perfil completo com batch pequeno. |
| Batch 64 | [Resultados](../../outputs/full-100-epochs-batch64/) | Perfil completo de referência. |
| Batch 512 otimizado para WSL | [Resultados](../../outputs/full-100-epochs-batch512-wsl-optimized/) | Perfil de alto batch com ajustes para WSL. |

## Experimentos direcionados

| Experimento | Local | Evidências principais |
| --- | --- | --- |
| Batch sweep KMNIST | [Resultados](../../outputs/kmnist-alldata-noaug-batch-sweep-2026-08-06/) | [Análise final](../../outputs/kmnist-alldata-noaug-batch-sweep-2026-08-06/analysis-2026-08-09-delivery-final/), métricas por época e relatório HTML/PDF. |
| Quantização KMNIST | [Resultados](../../outputs/kmnist-quantization-2026-08-09/) | [Análise](../../outputs/kmnist-quantization-2026-08-09/analysis/), comparação FP32/FP16/INT8/INT4 e relatório PDF. |
| Limite de RAM | [Resultados](../../outputs/remaining-ram-capped/) | Comparações por dataset e configurações de amostragem. |

## Como navegar dentro de uma execução

- `report.html` e `comparison.png`: leitura rápida por dataset.
- `runs/<identificador>/artifacts/`: métricas, matriz de confusão, curvas e
  relatórios de classificação.
- `runs/<identificador>/logs/`: histórico por época e eventos de treinamento.
- `checkpoints/`: modelos e pontos de retomada; modelos `.keras` e `.h5` são
  tratados como artefatos grandes.
- `analysis/`: comparações, tabelas agregadas e relatórios de conclusão.
