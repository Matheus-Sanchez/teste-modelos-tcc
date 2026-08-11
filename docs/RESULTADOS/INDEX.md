# Central de resultados

Este diretório é o ponto de entrada para localizar relatórios, diagnósticos e
evidências de testes do projeto. Os artefatos permanecem nos seus diretórios de
origem para preservar a reprodutibilidade dos scripts; esta central não duplica
arquivos nem altera os caminhos de execução.

## Acesso rápido

| Necessidade | Local |
| --- | --- |
| Relatório técnico consolidado dos 36 treinamentos | [PDF](../../output/pdf/analise-tecnica-dos-36-treinamentos-tcc.pdf) |
| Verificação visual final do relatório | [contato das páginas](../../output/pdf/verification-final/contact-sheet.png) e [registro da verificação](../../output/pdf/verification-final/report.txt) |
| Diagnósticos de datasets e auditorias | [Diagnósticos](DIAGNOSTICOS.md) |
| Benchmarks, testes de treinamento e quantização | [Testes e benchmarks](TESTES_E_BENCHMARKS.md) |
| Suíte de testes do código | [Testes automatizados](TESTES_AUTOMATIZADOS.md) |

## Onde os dados ficam

- `output/`: relatório técnico final e suas verificações em PDF/PNG.
- `outputs/`: resultados reproduzíveis dos experimentos, incluindo métricas,
  gráficos, relatórios HTML, CSVs, logs e checkpoints.
- `G:\tcc-benchmark\outputs`: cópia de segurança dos resultados. Em 10 de
  agosto de 2026, ela foi verificada como idêntica a
  `E:\tcc-benchmark\outputs` por tamanho e SHA-256.

## Resultados versionados

Os conjuntos abaixo fazem parte da branch
`agent/publish-benchmark-results`. Os arquivos de modelo `.keras` e `.h5` são
armazenados via Git LFS.

- [Batch sweep KMNIST](../../outputs/kmnist-alldata-noaug-batch-sweep-2026-08-06/)
- [Quantização KMNIST](../../outputs/kmnist-quantization-2026-08-09/)
- [Execuções com limite de RAM](../../outputs/remaining-ram-capped/)

Os demais conjuntos históricos em `outputs/` continuam disponíveis localmente
e na cópia de segurança em `G:`; eles são listados nas páginas desta central.

## Convenção para novos resultados

Para facilitar consultas futuras, cada nova execução deve ter uma pasta de
primeiro nível em `outputs/`, com nome descritivo e data (`<assunto>-AAAA-MM-DD`)
e, quando aplicável, subpastas `analysis/`, `runs/`, `artifacts/`,
`checkpoints/` e `logs/`.
