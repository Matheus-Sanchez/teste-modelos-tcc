# Relatório de memória e adaptação do `tcc_benchmark`

## 1. Objetivo

Este documento registra o estado atual do pacote `src/tcc_benchmark` e as
alterações recomendadas para transferi-lo para outro projeto junto com o
código-fonte.

O objetivo da adaptação é preservar o protocolo experimental e reduzir
duplicações internas sem remover funcionalidades necessárias por acidente.
Este relatório descreve mudanças; ele não significa que essas mudanças já
foram aplicadas no código.

## 2. Visão geral atual

O pacote implementa uma suíte local, retomável e observável para treinar a
mesma CNN do zero em diferentes datasets de imagens.

Fluxo principal:

```text
tcc-benchmark / python -m tcc_benchmark
  -> cli.py
  -> runner.py
  -> config.py + bootstrap.py
  -> adapters.py
  -> data.py
  -> model.py
  -> metrics.py + telemetry.py
  -> state.py
  -> reporting.py
```

Características que devem ser preservadas durante a migração:

- dados são locais; os adaptadores não fazem download;
- divisão estratificada determinística, normalmente 70/15/15;
- normalização calculada apenas com o treino bruto;
- balanceamento e augmentação aplicados somente ao treino;
- avaliação final no teste uma única vez;
- manifestos, fingerprints e estados persistentes permitem `resume`;
- cada combinação de dataset, normalização, balanceamento e seed possui uma
  saída própria;
- a arquitetura é uma CNN scratch baseada em convoluções separáveis;
- o protocolo atual usa TensorFlow/Keras 2.21 e Python 3.10.

## 3. Principais duplicações encontradas

### 3.1 Serialização, fingerprints e IDs

`config.py` possui `canonical_json()`, `fingerprint()` e `run_key()`.
`state.py` possui `canonical_json()`, `config_fingerprint()` e
`stable_run_id()`.

Recomendação:

- manter uma única implementação de JSON canônico;
- manter uma única implementação de fingerprint SHA-256;
- escolher `stable_run_id()` como identificador oficial das pastas;
- remover ou tornar `run_key()` um alias compatível;
- preservar a diferença conceitual entre identidade da run e fingerprint da
  configuração: o ID localiza a run e o fingerprint decide se ela pode ser
  retomada.

Sugestão de destino: criar `serialization.py` ou `identity.py` com funções
compartilhadas. Não fazer `state.py` depender de `config.py`, pois o estado é
intencionalmente utilizável sem carregar o restante da aplicação.

### 3.2 Métricas de classificação

`metrics.py:classification_metrics()` e
`reporting.py:compute_classification_metrics()` calculam essencialmente as
mesmas métricas, mas retornam estruturas diferentes.

Recomendação:

- criar uma função matemática única para matriz de confusão, precisão, recall,
  F1, acurácia e balanced accuracy;
- converter o resultado para `ClassificationMetrics` quando usado durante o
  treino;
- converter o mesmo resultado para dicionário quando usado em relatórios;
- manter a implementação sem dependência obrigatória de scikit-learn, pois
  relatórios podem ser gerados em um ambiente sem TensorFlow.

Não remover `evaluate_model()` nem os callbacks de `metrics.py`; eles são
integrações específicas com Keras.

### 3.3 Geração de relatórios

Há lógica de escrita de CSV, JSON, HTML e PNG em três lugares:

- `reporting.py`;
- `controlled.py:write_phase_report()`;
- `quantization.py:build_comparison_report()` e funções relacionadas.

Recomendação:

- centralizar funções genéricas de escrita e escape em `reporting.py` ou em um
  novo `report_io.py`;
- deixar `controlled.py` responsável apenas por filtrar linhas e escolher
  vencedores;
- deixar `quantization.py` responsável apenas por conversão, benchmarking e
  comparação específica de modelos quantizados;
- preservar os nomes e formatos dos artefatos existentes durante a migração.

### 3.4 Preflight e telemetria

`preflight.py` coleta informações de CPU, RAM, GPU, versões e dependências
antes do treino. `telemetry.py` coleta informações semelhantes durante a
execução, além de uso do processo, disco e eventos de época.

Eles não são substitutos diretos e não devem ser simplesmente fundidos.
Recomendação: extrair apenas os probes comuns para um módulo interno, por
exemplo `hardware.py`, mantendo:

- `preflight.py` como diagnóstico inicial;
- `telemetry.py` como amostrador contínuo;
- valores indisponíveis representados por `null`/`None`.

## 4. Adaptações recomendadas por arquivo

| Arquivo | Ação recomendada na transferência |
|---|---|
| `__init__.py` | Manter, mas reduzir os exports ao que o outro projeto realmente usa. Os exports de `bootstrap.py` são úteis como API pública, mas podem ser removidos se o pacote for somente CLI. |
| `__main__.py` | Manter se o projeto aceitar `python -m tcc_benchmark`. Pode ser removido se o único entry point for o comando definido no `pyproject.toml`. |
| `adapters.py` | Manter como camada de entrada de dados. Se o novo projeto usar poucos datasets, remover adaptadores não registrados e seus testes; atualizar também `DATASET_ORDER`, YAML e documentação. Não misturar regras de treino neste arquivo. |
| `audit.py` | Manter separado dos adaptadores. Adaptar apenas o contrato de `LoadedDataset` se a representação dos dados mudar. Preservar a possibilidade de auditar sem TensorFlow. |
| `bootstrap.py` | Manter se houver configuração global e imports opcionais. Simplificar `IMPORT_CATALOG`, `ProjectPaths` e `DEFAULT_PROJECT_PATHS` se não forem usados como API ou em testes. Não importar TensorFlow antecipadamente. |
| `cli.py` | Manter como camada fina de argumentos. Ajustar nomes de comandos, defaults e caminhos do novo projeto; não colocar lógica de treinamento aqui. |
| `config.py` | Manter as dataclasses de configuração e validação. Consolidar `canonical_json`, `fingerprint` e `run_key` com `state.py`; adaptar o schema YAML e os datasets disponíveis. |
| `controlled.py` | Tratar como módulo opcional. Ele é usado principalmente por `scripts/run_controlled_pipeline.py`, não pelo fluxo principal de `runner.py`. Se o experimento controlado não for transferido, remover este arquivo, o script correspondente e `test_controlled.py`. |
| `data.py` | Manter como núcleo de preparação. Preservar a ordem split → estatística de normalização → balanceamento → augmentação. Adaptar somente formatos, tamanhos e opções realmente suportados pelo novo projeto. |
| `metrics.py` | Manter callbacks, avaliação Keras e objetos de métricas. Extrair a matemática comum para uma função compartilhada com `reporting.py`. |
| `model.py` | Manter separado para preservar a arquitetura. Só alterar filtros, blocos, dropout, loss ou otimizador se o novo projeto aceitar um protocolo experimental diferente; nesse caso, criar novo perfil de saída e documentar a quebra de comparabilidade. |
| `preflight.py` | Manter se o projeto depender de GPU, TensorFlow ou execução longa. Adaptar versões, comandos de GPU e pacotes esperados. Pode ser reduzido em projetos CPU-only. |
| `quantization.py` | Tornar opcional se quantização/LiteRT não fizer parte do escopo. Se mantido, verificar compatibilidade entre TensorFlow, Keras e LiteRT; manter explícito que INT4 é emulação/fake quantization quando aplicável. |
| `reporting.py` | Manter como ponto central dos relatórios. Absorver utilitários comuns de `controlled.py` e `quantization.py`, sem alterar os schemas dos relatórios já consumidos por notebooks e scripts. |
| `runner.py` | Manter como orquestrador, mas considerar dividir em `commands.py`, `execution.py` e `smoke.py`. A função `_run_cell()` é o ponto crítico; qualquer adaptação deve continuar criando manifesto antes do treino, validando fingerprints e atualizando status. |
| `state.py` | Manter como camada independente. É responsável por escrita atômica, manifestos, transições de estado, IDs e compatibilidade de configuração. Não acoplar ao TensorFlow ou aos adaptadores. |
| `telemetry.py` | Manter se houver necessidade de comparar tempo, CPU, RAM, GPU, VRAM ou disco. Compartilhar apenas probes básicos com `preflight.py`; não transformar telemetria em requisito obrigatório para executar o treino. |

## 5. Arquivos que podem ser removidos com segurança condicional

As remoções abaixo só são seguras se as dependências indicadas também forem
removidas ou adaptadas.

### Remover `controlled.py`

Somente se o novo projeto não usar:

- `scripts/run_controlled_pipeline.py`;
- seleção de batch, quantização ou ativação;
- relatórios de fases controladas;
- `tests/test_controlled.py`.

### Remover `quantization.py`

Somente se o projeto não executar os experimentos FP32, FP16, QAT, PTQ ou
LiteRT. Também será necessário remover ou adaptar os scripts de quantização e
os testes que importam `FakeQuantize`, `QATDense`, `QATSeparableConv2D` e os
benchmarks LiteRT.

### Remover `preflight.py` ou `telemetry.py`

Somente em um projeto deliberadamente CPU-only e de execução curta. Para
experimentos longos ou uso de GPU, recomenda-se mantê-los.

### Remover `__main__.py`

Somente se `tcc-benchmark` for o único modo de execução suportado.

## 6. Estrutura simplificada sugerida

Uma organização mais enxuta, mantendo o comportamento, seria:

```text
src/tcc_benchmark/
  __init__.py
  __main__.py                  # opcional
  cli.py
  config.py
  identity.py                  # JSON canônico, fingerprints e IDs
  adapters.py
  audit.py
  data.py
  model.py
  metrics.py
  runner.py
  state.py
  reporting.py
  preflight.py                 # opcional em CPU-only
  telemetry.py                 # opcional em execuções curtas
  quantization.py              # opcional fora do escopo de quantização
```

Se o `runner.py` continuar crescendo, dividir apenas a orquestração:

```text
runner.py       -> seleção de datasets, comandos e ciclo geral
execution.py    -> execução de uma run e tratamento de falhas
smoke.py        -> execução reduzida para validação
```

Essa divisão é preferível a mover lógica de negócio para `cli.py`.

## 7. Ordem recomendada de adaptação

1. Copiar `src/tcc_benchmark`, `configs`, `tests` e o `pyproject.toml` como referência.
2. Decidir o escopo do novo projeto: datasets, quantização, pipeline controlado, GPU e relatórios.
3. Adaptar primeiro `config.py`, `bootstrap.py` e os YAMLs.
4. Adaptar `adapters.py` e executar `audit` sem TensorFlow.
5. Adaptar e testar a camada de dados (`data.py`).
6. Validar `model.py`, `metrics.py` e `runner.py` com `smoke`.
7. Consolidar fingerprints, IDs e métricas duplicadas.
8. Consolidar os escritores de relatórios sem alterar os campos de saída.
9. Adaptar `preflight.py`, `telemetry.py` e, se necessário, `quantization.py`.
10. Executar a suíte de testes e um `dry-run` antes de qualquer matriz longa.

## 8. Critérios de aceitação da migração

A adaptação deve ser considerada correta quando:

- `python -m pytest` passa no novo projeto;
- `tcc-benchmark --help` lista apenas os comandos suportados;
- `audit` funciona sem TensorFlow e sem download automático;
- `run --dry-run` cria a matriz esperada sem iniciar treinamento;
- `smoke` conclui em uma amostra pequena;
- uma execução interrompida pode ser retomada somente com configuração compatível;
- o fingerprint impede retomar uma run com split ou hiperparâmetros diferentes;
- relatórios continuam sendo gerados no formato esperado;
- dependências opcionais ausentes geram mensagens claras, sem quebrar comandos leves;
- o modelo usado para comparação continua com a mesma arquitetura e política de precisão, caso a comparabilidade científica seja necessária.

## 9. Cuidados para não quebrar a validade experimental

Não alterar silenciosamente:

- seed;
- frações de treino, validação e teste;
- ordem do split;
- estatística de normalização;
- regras de balanceamento;
- augmentação;
- batch size e resolução;
- arquitetura da CNN;
- critério de seleção do melhor checkpoint;
- nomes e localização dos artefatos.

Se qualquer um desses itens mudar, usar uma nova raiz de saída e registrar a alteração como um novo experimento.

## 10. Observação sobre o estado atual do repositório

No momento da análise, o working tree já continha alterações e arquivos novos fora deste relatório, incluindo mudanças em `__init__.py`, `cli.py` e `runner.py`, além de `bootstrap.py`, testes e scripts auxiliares. Essas alterações devem ser transferidas junto com o código somente depois de serem revisadas no projeto de destino; não devem ser descartadas automaticamente.

