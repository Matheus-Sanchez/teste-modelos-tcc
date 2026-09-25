# Imports e configuração global

O ponto único de inicialização do pacote é `tcc_benchmark.bootstrap`. Ele é
reexportado por `tcc_benchmark`, portanto consumidores externos podem usar:

```python
from tcc_benchmark import DEFAULT_PROJECT_PATHS, load_global_configuration

configuration = load_global_configuration()
```

`load_global_configuration()` carrega e valida o YAML da suíte, o registro de
datasets e resolve os caminhos. As opções de linha de comando continuam sendo
as fontes de sobrescrita para `--suite`, `--registry` e `--output-root`.
`configure_training_runtime()` concentra a configuração de ambiente exigida
antes do primeiro import do TensorFlow.

## Inventário de imports

O catálogo executável `IMPORT_CATALOG`, em
`src/tcc_benchmark/bootstrap.py`, é a fonte de referência. Ele cobre todos os
imports utilizados pelos módulos de `src/tcc_benchmark`, agrupados em:

| Grupo | Imports |
| --- | --- |
| Biblioteca padrão | `argparse`, `collections`, `contextlib`, `csv`, `dataclasses`, `datetime`, `gzip`, `hashlib`, `html`, `importlib.metadata`, `io`, `json`, `math`, `os`, `pathlib`, `pickle`, `platform`, `re`, `shutil`, `statistics`, `struct`, `subprocess`, `sys`, `tempfile`, `threading`, `time`, `traceback`, `typing`, `unicodedata`, `zlib` |
| Dependências externas | `matplotlib`, `matplotlib.pyplot`, `numpy`, `PIL`, `PIL.Image`, `psutil`, `pynvml`, `scipy.io`, `sklearn.metrics`, `tensorflow`, `yaml` |
| Módulos locais | `adapters`, `audit`, `cli`, `config`, `controlled`, `data`, `metrics`, `model`, `preflight`, `quantization`, `reporting`, `runner`, `state`, `telemetry` |

Os nomes de instalação correspondentes estão em `pyproject.toml`: Pillow fornece
`PIL`, PyYAML fornece `yaml`, `nvidia-ml-py` fornece `pynvml`, e
scikit-learn fornece `sklearn`. TensorFlow é intencionalmente opcional, definido
nos extras `windows-smoke` e `wsl-gpu`.

## Por que o catálogo não importa tudo de uma vez

Centralizar `import tensorflow`, `matplotlib` ou `PIL` no startup impediria os
comandos leves (`audit`, configuração e relatórios sem gráficos) de rodarem em
ambientes sem o perfil de treino. O bootstrap concentra a configuração e a
lista de dependências, enquanto os imports pesados permanecem locais e tardios.
