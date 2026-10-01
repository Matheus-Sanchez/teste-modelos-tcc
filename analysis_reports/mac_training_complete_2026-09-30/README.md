# Treinamentos Mac com augmentation 0,5

O entregável é [o notebook Python executado](../../notebooks/mac_all_training_complete_report.ipynb).
Ele contém descrição do ambiente e hardware histórico, protocolo, comparações,
curvas, uso dos recursos, auditoria dos tempos e fichas interpretadas de cada run.

## Cobertura

- 36 treinos de batch (32, 64, 128 e 256 × nove datasets).
- 27 treinos de ativações ocultas (ReLU, Sigmoid e Softmax, batch 256).
- 18 treinos FP32/FP16 misto (batch 256).
- Nove avaliações INT8 PTQ, separadas dos treinamentos.
- Nove smoke tests de duas épocas em apêndice, fora das comparações principais.

Todos os 81 treinos principais usam 100 épocas e `extra_fraction=0.5`.
O total de duração das épocas preservadas é 505,10 horas; ele não inclui
intervalos ociosos ou trabalho descartado em épocas interrompidas. WSL/RTX 3050
e augmentation 2,0/0,0 não entram no relatório.

## Evidência reproduzível

`evidence/` contém:

- `runs.csv`: 99 identidades de execução/avaliação, parâmetros, resultados, custos e estatísticas de hardware.
- `epoch_history.csv`: 8.118 épocas (8.100 principais + 18 smoke), sem duplicar as cópias de checkpoint/logs.
- `per_class_metrics.csv`: 1.837 resultados por classe, com suporte, precisão, recall e F1.
- `telemetry_minute.csv`: 21.760 janelas de telemetria bruta por sessão e minuto UTC.
- `environments.json`: 45 ambientes históricos das execuções locais de batch/smoke.
- `campaign_metadata.json`: metadados do relatório histórico da campanha de ativações.
- `quality_checks.csv`: reconciliação por classe, sequência/contagem de épocas, espelhos locais e divergências de tempo.
- `sources.json`: caminho relativo, tamanho e SHA-256 de cada arquivo-fonte utilizado.
- `manifest.json`: escopo, contagens e fuso do relatório.

Os snapshots de ativações/precisão têm curvas de treino e hardware agregado,
mas não têm amostras brutas locais. Séries de hardware aparecem somente para
batch/smoke. Valores ausentes não são inventados. Estatísticas brutas são
recalculadas sobre amostras periódicas; estatísticas exportadas preservam a
semântica do coletor original. Nenhuma delas é uma medição de energia.

Há 19 diferenças maiores que um segundo entre o resumo de duração e a soma
das épocas nos treinos principais. O notebook documenta ambas e usa a soma
das 100 épocas para as comparações, inclusive nas retomadas.

## Reexecução

Para ler, basta abrir o notebook: tabelas e gráficos já estão nas saídas.
Para reexecutar, preserve a estrutura relativa entre `notebooks/` e esta pasta.
Não é preciso TensorFlow, pesos ou datasets.

```bash
python3.10 -m venv .venv-report
.venv-report/bin/python -m pip install -r requirements/mac-report.txt
.venv-report/bin/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/mac_all_training_complete_report.ipynb
```

Para reconstruir a evidência a partir dos outputs locais e dos snapshots e
executar o relatório em um kernel temporário com o mesmo interpretador:

```bash
.venv-report/bin/python scripts/build_mac_training_complete_report.py --execute
```

As figuras PNG e a versão de leitura HTML são geradas nesta pasta e ignoradas
pelo Git; o notebook conserva todas as figuras inline. `source_index.csv`
oferece a proveniência também em formato tabular. A data do relatório é
30/09/2026 (America/Sao_Paulo), com datas reais das fontes preservadas nos dados.
