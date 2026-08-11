# Testes automatizados

Volte à [Central de resultados](INDEX.md).

## Suíte atual

Os testes de código estão em [`tests/`](../../tests/) e cobrem:

| Módulo | Cobertura principal |
| --- | --- |
| `test_adapters_audit.py` | Adaptadores e auditoria. |
| `test_cli.py` | Interface de linha de comando. |
| `test_config.py` | Configurações, registros e perfis. |
| `test_core_training.py` | Fluxo central de treinamento. |
| `test_fetch_datasets.py` | Obtenção e preparação de datasets. |
| `test_observability.py` | Telemetria e observabilidade. |
| `test_preflight.py` | Verificações antes da execução. |
| `test_quantization.py` | Fluxos de quantização. |
| `test_runner.py` | Orquestração de execuções. |

## Execução

No PowerShell, a partir da raiz do projeto:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest
```

Última validação registrada: **34 testes aprovados** em 10 de agosto de 2026.

> O ambiente precisa ter `pytest` e `PyYAML` instalados. As dependências do
> projeto estão declaradas em `pyproject.toml` e nos perfis em `requirements/`.
