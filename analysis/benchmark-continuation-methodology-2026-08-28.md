# Metodologia da decisão de continuação

Esta tabela foi compilada em 28 de agosto de 2026 a partir de artefatos locais de runs concluídas.

Fontes diretas:

- `outputs/controlled-augmentation2/batch/**/manifest.json`
- `outputs/controlled-augmentation2/batch/**/artifacts/test_metrics.json`
- `outputs/controlled-augmentation2/batch/**/logs/epoch_metrics.csv`
- `outputs/kmnist-batch32-noaugmentation-control-2026-08-28/**/manifest.json`
- `outputs/kmnist-batch32-noaugmentation-control-2026-08-28/**/artifacts/test_metrics.json`
- `outputs/kmnist-batch32-noaugmentation-control-2026-08-28/**/logs/epoch_metrics.csv`

O tempo de treino é a soma de `epoch_seconds` em `logs/epoch_metrics.csv`. Esta regra evita usar `training_summary.json` em runs retomadas, pois esse resumo pode conter apenas a tentativa mais recente.

O grid de batch inclui somente MNIST, Fashion-MNIST e KMNIST, todos com `unit_interval`, `all_raw`, seed 42, 100 épocas e augmentation 2.0. O controle de augmentation compara duas runs de KMNIST com batch 32, mesmo split e seed, também por 100 épocas. Apenas `extra_fraction` e a política de transforms variam.
