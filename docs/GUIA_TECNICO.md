# Guia técnico do benchmark

Este guia explica a estrutura atual do projeto e indica o lugar correto para cada alteração manual. A matriz ativa é configurada em [configs/full-100-epochs.yaml](../configs/full-100-epochs.yaml): batch 64, mixed_float16, no máximo 100 épocas, resolução 64x64 na regra geral e 128x128 no GTSRB.

## 1. O que o projeto faz

A mesma CNN é treinada do zero em diferentes datasets, normalizações, balanceamentos e seeds. Não existem pesos pré-treinados.

As partições oficiais disponíveis de cada dataset são unidas. O projeto cria então um split estratificado e determinístico:

~~~text
dados locais unidos
  -> 70% treino / 15% validação / 15% teste
  -> estatística de normalização calculada só no treino bruto
  -> balanceamento aplicado só no treino
  -> augmentação aplicada só no treino
  -> treino e seleção pelo Macro-F1 da validação
  -> avaliação final única no teste
~~~

A configuração final atual cria 112 runs:

~~~text
MNIST:              2 normalizações x 4 balanceamentos x 5 seeds = 40
Outros 9 datasets:  2 normalizações x 4 balanceamentos x 1 seed  = 72
Total: 112 runs sequenciais
~~~

## 2. Quem chama quem

~~~text
tcc-benchmark ou python -m tcc_benchmark
  -> cli.py: main()
     -> preflight.py: run_preflight()               [preflight]
     -> runner.py: command_audit()                  [audit]
     -> runner.py: command_run()                    [run e resume]
        -> config.py: load_suite_settings()
        -> config.py: load_dataset_registry()
        -> preflight.py: run_preflight()
        -> runner.py: _run_dataset_matrix()
           -> adapters.py: load_local_dataset()
           -> runner.py: _join_source_samples()
           -> runner.py: _run_cell()                [uma combinação]
              -> state.py: initialize_run()
              -> data.py: prepare_run_datasets()
              -> model.py: build_and_compile_legacy_cnn()
              -> metrics.py: make_training_callbacks()
              -> telemetry.py: TelemetrySampler()
              -> Keras model.fit()
              -> metrics.py: evaluate_model()
              -> reporting.py: write_run_report()
        -> reporting.py: build_dataset_report() e build_global_index()
     -> runner.py: command_smoke()                  [smoke]
     -> runner.py: command_report()                 [report]
~~~

O principal ponto de depuração é runner.py, função _run_cell. Ela recebe uma combinação de dataset, normalização, balanceamento e seed, cria ou valida o manifesto, prepara os dados, treina, avalia e salva a run.

## 3. Fonte única dos parâmetros de treino

O bloco suite.training do YAML escolhido é a fonte de verdade. Para a matriz principal, edite apenas configs/full-100-epochs.yaml.

~~~yaml
suite:
  training:
    max_epochs: 100
    batch_size: 64
    learning_rate: 0.0003
    extra_fraction: 2.0
    dtype_policy: mixed_float16
    keras_verbose: 1
    default_image_size: 64
    image_size_overrides: {gtsrb: 128}
    augmentation: { ... }
    early_stopping_patience: 10
    reduce_lr_factor: 0.3
    reduce_lr_patience: 4
    reduce_lr_min_lr: 0.0000001
~~~

| Campo | Efeito | Cuidado |
|---|---|---|
| max_epochs | Limite máximo de épocas de uma run. | Early stopping pode encerrar antes. |
| batch_size | Imagens por passo; altera VRAM, throughput e a dinâmica do Adam. | Testar no GTSRB a 128x128 antes da matriz. O último batch pode ser menor. |
| learning_rate | LR inicial do Adam. | Muda o experimento; não misturar outputs. |
| extra_fraction | Exemplos aumentados extras. 2.0 gera base + duas cópias aumentadas, aproximadamente 3x passos. | É o principal controle de duração por época. |
| dtype_policy | Política de precisão do Keras. | O protocolo atual aceita apenas mixed_float16. |
| keras_verbose | 1 mostra barra por batch; 2 mostra linha por época; 0 silencia. | Métricas por classe continuam em arquivo. |
| default_image_size | Resolução padrão. | A CNN exige no mínimo 64 por conter cinco pools. |
| image_size_overrides | Exceções por dataset, atualmente GTSRB=128. | Aumenta VRAM e tempo. |
| augmentation | Flip, brilho, contraste, translação, zoom, ruído e cutout. | Só vale para treino; flip pode ser inadequado em alguns dados. |
| preprocess_cache_max_mib | Só mantém em RAM um conjunto já decodificado quando sua estimativa cabe nesse teto. | Aumentar pode acelerar épocas, mas não limita o shuffle. |
| shuffle_buffer_max_mib | Orçamento total para os buffers de shuffle; com augmentação ele é dividido entre os fluxos raw e aumentado. `0` preserva o shuffle completo. | Use 512–1024 no WSL quando a RAM for limitada; reduz picos sem mudar batch, modelo ou augmentação. |
| early_stopping_patience | Épocas sem melhora de Macro-F1 antes de parar. | Muda o total efetivo de épocas. |
| reduce_lr_* | Redução de LR em platô. | Muda a trajetória de otimização. |

Os campos externos a training controlam a matriz: output_root, seeds, dataset_seeds, normalizations, balance_modes, as frações de split e a telemetria.

Regra: alterar batch, resolução, precisão, augmentação, seed, split, normalização, balanceamento ou arquitetura cria um experimento novo. Use outro output_root. Uma run interrompida com configuração diferente não deve ser retomada.

## 4. Arquivo por arquivo

### Entrada e CLI

| Arquivo | Responsabilidade | Chamado por |
|---|---|---|
| [pyproject.toml](../pyproject.toml) | Pacote, dependências e entry point tcc-benchmark. | pip e ambiente Python. |
| [src/tcc_benchmark/__main__.py](../src/tcc_benchmark/__main__.py) | Permite python -m tcc_benchmark. | Python. |
| [src/tcc_benchmark/__init__.py](../src/tcc_benchmark/__init__.py) | Exporta constantes públicas simples. | Imports externos. |
| [src/tcc_benchmark/cli.py](../src/tcc_benchmark/cli.py) | Declara argumentos e encaminha os comandos; não treina diretamente. | Entry point. |

A função main de cli.py trata Ctrl+C. O comando sai com código 130 e a próxima execução deve usar resume.

### Configuração, dados locais e auditoria

| Arquivo | Responsabilidade | Onde modificar |
|---|---|---|
| [configs/full-100-epochs.yaml](../configs/full-100-epochs.yaml) | Perfil ativo: matriz, saída e parâmetros globais. | Primeiro lugar para editar o experimento final. |
| [configs/gpu-memory-check.yaml](../configs/gpu-memory-check.yaml) | Perfil curto histórico: batch 16 e cinco épocas. | Somente para repetir a checagem curta antiga. |
| [configs/suite.yaml](../configs/suite.yaml) | Perfil usado se --suite não for informado. | Mantenha coerente com o padrão desejado. |
| [configs/datasets.yaml](../configs/datasets.yaml) | Nome de dataset, adaptador e diretório local. | Caminhos e opções da fonte. |
| [config.py](../src/tcc_benchmark/config.py) | Lê YAML, resolve caminhos, valida e cria TrainingSettings, SuiteSettings e DatasetEntry. Copia a configuração para cada manifesto. | Ao criar campo novo, acrescente dataclass, validação, YAML e teste. |
| [scripts/fetch_datasets.py](../scripts/fetch_datasets.py) | Baixa e instala datasets, registrando SOURCE.json e checksums quando disponíveis. | Aquisição somente; treino nunca o chama. |
| [adapters.py](../src/tcc_benchmark/adapters.py) | Lê os formatos locais e entrega LoadedDataset e DatasetSplit. | Adicione adaptadores novos aqui. |
| [audit.py](../src/tcc_benchmark/audit.py) | Verifica arquivos, rótulos, formatos, imagens corrompidas, distribuição e duplicatas. | Ajuste regras de qualidade. |

A função adapters.load_local_dataset escolhe o carregador pelo campo adapter no YAML:

- MNIST, Fashion-MNIST e KMNIST: IDX ou NPZ;
- EMNIST Balanced: IDX, NPZ ou MAT;
- CIFAR-10 e CIFAR-100 coarse: pickles oficiais; CIFAR-100 converte para 20 superclasses;
- CINIC-10: pastas train, valid e test por classe;
- SVHN: arquivos MAT;
- GTSRB: pastas por classe ou CSV, retornando caminhos para leitura preguiçosa;
- FER2013: CSV com emotion e pixels.

### Dados, normalização e balanceamento

| Arquivo | Responsabilidade | Chamado por |
|---|---|---|
| [data.py](../src/tcc_benchmark/data.py) | Split, normalização, balanceamento, augmentação e tf.data. | runner._run_cell. |
| [model.py](../src/tcc_benchmark/model.py) | CNN scratch, seed, precisão e compilação. | runner._run_cell. |
| [metrics.py](../src/tcc_benchmark/metrics.py) | Macro-F1, balanced accuracy, métricas por classe, callbacks e avaliação. | runner._run_cell. |

A função prepare_run_datasets de data.py executa a sequência abaixo:

~~~text
rótulos
  -> stratified_split_indices()
  -> compute_normalization_stats() no treino bruto e redimensionado
  -> balance_training_data() apenas no treino
  -> build_tf_dataset() para treino, validação e teste
~~~

build_tf_dataset decodifica imagem ou caminho, mantém canais nativos (1 ou 3), redimensiona com padding proporcional, normaliza e, no treino, aplica augmentação com seeds stateless. Em seguida cria base + extra_fraction exemplos, embaralha, faz batch e prefetch AUTOTUNE.

| Modo de balanceamento | Ação no treino |
|---|---|
| all_raw | Mantém todas as amostras, sem pesos. |
| undersample | Reduz todas as classes ao tamanho da menor. |
| oversample | Repete classes menores até o tamanho da maior. |
| class_weight | Mantém dados e aplica N / (K x n_classe) na loss. |

Validação e teste nunca recebem reamostragem, pesos, augmentação ou TTA.

### CNN

model.build_legacy_cnn mantém a topologia herdada:

~~~text
Entrada H x W x 1 ou 3
 -> 5 blocos:
    SeparableConv 5x5 -> GroupNorm -> swish
    SeparableConv 3x3 -> GroupNorm -> swish -> MaxPool
 -> GlobalAveragePooling + GlobalMaxPooling
 -> Dropout 0.4 -> Dense 256 SiLU
 -> Dropout 0.4 -> Dense 256 SiLU
 -> Dropout 0.4 -> Dense num_classes softmax FP32
~~~

compile_legacy_cnn usa Adam, SparseCategoricalCrossentropy, accuracy e jit_compile=False. set_dtype_policy configura mixed_float16; variáveis, softmax final e loss permanecem FP32.

Alterar filtros, blocos, dropout, loss ou otimizador significa trocar a arquitetura/protocolo. Crie uma pasta de output nova e documente a decisão.

### Orquestração, estado, telemetria e relatórios

| Arquivo | Responsabilidade | Detalhe |
|---|---|---|
| [runner.py](../src/tcc_benchmark/runner.py) | Orquestra a matriz. | command_run -> _run_dataset_matrix -> _run_cell. |
| [state.py](../src/tcc_benchmark/state.py) | Escrita atômica, IDs estáveis, manifestos, status e compatibilidade. | Impede retomada de configuração incompatível. |
| [telemetry.py](../src/tcc_benchmark/telemetry.py) | CPU, RAM, I/O, disco, GPU, VRAM, temperatura e potência via NVML; fallback nvidia-smi. | Amostra a cada 5 s e em marcos de época. |
| [reporting.py](../src/tcc_benchmark/reporting.py) | HTML, CSV, JSON, PNG, matriz de confusão e índices. | Chamado após run e dataset. |
| [preflight.py](../src/tcc_benchmark/preflight.py) | TensorFlow, GPU, driver, disco e dependências. | Antes do treino. |

metrics.make_training_callbacks cria, nesta ordem:

1. EpochMetricsCallback: Macro-F1, balanced accuracy e métricas por classe da validação; grava checkpoints/epoch_metrics.csv;
2. ModelCheckpoint best.keras: maior val_macro_f1;
3. ModelCheckpoint last.keras: fim de cada época;
4. EarlyStopping com valores de training;
5. ReduceLROnPlateau com valores de training;
6. BackupAndRestore: modelo, otimizador e época;
7. callback de telemetria.

A barra Keras mostra métricas globais, LR e tempo. Métricas por classe não entram no terminal porque GTSRB teria centenas de campos; elas continuam no CSV e nos relatórios.

### Scripts auxiliares

| Arquivo | Papel | Uso |
|---|---|---|
| [scripts/wsl-gpu-env.sh](../scripts/wsl-gpu-env.sh) | Configura bibliotecas CUDA do ambiente WSL e executa o comando recebido. | Use em comandos de treino GPU. |
| [scripts/live_benchmark_monitor.py](../scripts/live_benchmark_monitor.py) | Gera snapshot operacional de status e GPU. | Manual; não iniciar com outro supervisor. |
| [scripts/summarize_short_test.py](../scripts/summarize_short_test.py) | Consolida o teste curto em HTML, CSV e JSON. | Manual. |
| [scripts/supervise_benchmarks.py](../scripts/supervise_benchmarks.py) | Encadeia teste curto, resumo e retomada. | Não usar simultaneamente a run --all direta. |

## 5. Estrutura de saída

~~~text
outputs/full-100-epochs-batch64/
├── preflight.json
├── index.html e index.json
├── README.md
└── <dataset>/
    ├── summary.csv, summary.json, report.html e gráficos PNG
    └── runs/<dataset>__<normalization>__<balance>__seed-<seed>/
        ├── manifest.json             configuração completa e fingerprints
        ├── status.json               estado da run
        ├── checkpoints/
        │   ├── best.keras            maior Macro-F1 da validação
        │   ├── last.keras            fim da última época
        │   ├── backup/               BackupAndRestore
        │   └── epoch_metrics.csv
        ├── logs/
        │   ├── model_summary.txt
        │   ├── keras_history.json
        │   └── training_summary.json
        ├── telemetry/
        │   ├── samples.csv
        │   ├── summary.json
        │   └── environment.json
        └── artifacts/
            ├── test_metrics.json
            ├── predictions.csv
            ├── confusion_matrix.*
            └── report.html
~~~

best.keras é o modelo pronto daquela run. Não compare pesos diretamente entre datasets: o número de classes e a camada final podem ser diferentes.

## 6. Procedimentos de alteração manual

### Batch, resolução ou precisão

1. Pare o processo atual.
2. Edite suite.training.
3. Mude output_root se já existem runs da matriz anterior.
4. Rode smoke no GTSRB, que é RGB 128x128.
5. Somente depois rode run --all.

Exemplo de experimento batch 128 separado:

~~~yaml
suite:
  output_root: ../outputs/full-100-epochs-batch128
  training:
    batch_size: 128
~~~

### Matriz sem mudar modelo

Edite seeds, dataset_seeds, normalizations, balance_modes ou as frações de split. A CNN permanece igual, mas o experimento muda e precisa de saída nova.

### CNN

Edite model.py, principalmente LEGACY_FILTERS ou build_legacy_cnn. Depois teste e trate o resultado como nova arquitetura.

### Novo dataset

1. Criar um carregador em adapters.py que devolva LoadedDataset;
2. registrá-lo no mapa de adaptadores;
3. adicionar nome em DATASET_ORDER de config.py;
4. configurar caminho e adaptador em datasets.yaml;
5. definir image_size_overrides se não usar 64;
6. criar teste e executar audit.

## 7. Testes

| Arquivo | Cobertura |
|---|---|
| tests/test_config.py | YAML, registry, fingerprints e TrainingSettings. |
| tests/test_adapters_audit.py | Adaptadores e auditoria. |
| tests/test_core_training.py | Split, normalização, balanceamento, CNN, callbacks e retomada. |
| tests/test_runner.py | Planejamento, estado e smoke. |
| tests/test_observability.py | Telemetria e relatórios. |
| tests/test_preflight.py | Diagnóstico de ambiente. |
| tests/test_cli.py | CLI. |
| tests/test_fetch_datasets.py | Baixador local. |

~~~powershell
wsl.exe -d Ubuntu-22.04 --cd /mnt/c/source/repos/teste-modelos-tcc -- /home/msduda/.venvs/tcc-benchmark/bin/python -m pytest -q
~~~

## 8. Comandos operacionais

~~~powershell
# Smoke com a configuração escolhida
wsl.exe -d Ubuntu-22.04 -- bash -lc "cd /mnt/c/source/repos/teste-modelos-tcc && PYTHON_BIN=/home/msduda/.venvs/tcc-benchmark/bin/python scripts/wsl-gpu-env.sh /home/msduda/.venvs/tcc-benchmark/bin/tcc-benchmark smoke --dataset gtsrb --epochs 1 --examples-per-class 32 --suite configs/full-100-epochs.yaml"

# Iniciar a matriz nova
wsl.exe -d Ubuntu-22.04 -- bash -lc "cd /mnt/c/source/repos/teste-modelos-tcc && PYTHON_BIN=/home/msduda/.venvs/tcc-benchmark/bin/python scripts/wsl-gpu-env.sh /home/msduda/.venvs/tcc-benchmark/bin/tcc-benchmark run --all --suite configs/full-100-epochs.yaml"

# Retomar apenas runs compatíveis e interrompidas
wsl.exe -d Ubuntu-22.04 -- bash -lc "cd /mnt/c/source/repos/teste-modelos-tcc && PYTHON_BIN=/home/msduda/.venvs/tcc-benchmark/bin/python scripts/wsl-gpu-env.sh /home/msduda/.venvs/tcc-benchmark/bin/tcc-benchmark resume --all --suite configs/full-100-epochs.yaml"
~~~

Se houver OOM, o runner registra a run como failed e não reduz batch, resolução ou épocas automaticamente. Corrija o YAML, escolha uma saída nova e rode smoke antes de reiniciar.

## 9. Backup consolidado

Para restaurar o projeto sem separar código e resultados, mantenha uma cópia
consolidada em `G:\repos\teste-modelos-tcc`. A raiz combina:

- `C:\source\repos\teste-modelos-tcc`: código, configurações, documentos,
  notebooks e resultados já mantidos dentro do repositório;
- `E:\tcc-benchmark\outputs`: resultados produzidos no segundo SSD, copiados
  para `outputs/` na raiz consolidada.

Não mova nem renomeie pastas de experimento ao consolidar: seus caminhos são
parte das referências usadas por scripts de análise. A separação é funcional:
`outputs/` guarda resultados de execução; `output/` guarda entregáveis finais;
`artifacts/` guarda produtos auxiliares. `node_modules/` fica fora do backup e
deve ser recriado com o gerenciador de pacotes do respectivo aplicativo.

O backup é incremental e não apaga arquivos já presentes no HDD. A estrutura
detalhada e o procedimento de conferência estão em
[BACKUP_ESTRUTURA.md](BACKUP_ESTRUTURA.md).
