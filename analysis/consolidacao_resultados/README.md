# Consolidação de resultados — Windows × Mac

Esta pasta contém a consolidação reproduzível dos treinamentos capturados até o snapshot de 15/09/2026.

## Entregáveis

- `RELATORIO_CONSOLIDADO.md`: relatório completo em Markdown, com tabelas, interpretações e anexos.
- `relatorio-consolidado-windows-mac.html`: relatório HTML autônomo.
- `../../notebooks/consolidacao_resultados_windows_mac.ipynb`: notebook Python pré-executado.
- `data/`: CSVs e JSONs normalizados, manifesto de captura, validações e checksums.
- `figures/`: 18 gráficos e matrizes de confusão.

## Cobertura

- 119 runs Windows com métricas finais.
- 42 runs Mac com métricas finais exatas.
- 21 pares descritivos de ativações.
- 6 pares de batch com protocolo alinhado.
- 2.307 linhas de métricas por classe.
- 47.954 células de matrizes de confusão.
- 10.243 registros por epoch.
- 22.872 registros de telemetria resumida e 2.552 bins temporais.

Resultados Mac presentes apenas em relatórios históricos permanecem identificados como `report_only`; nenhum valor ausente foi inferido.

## Reprodução

Use o Python configurado para o projeto e execute, a partir da raiz do repositório:

```powershell
python analysis/consolidacao_resultados/consolidate_results.py
python analysis/consolidacao_resultados/render_figures.py
python analysis/consolidacao_resultados/build_notebook.py
python analysis/consolidacao_resultados/build_markdown_report.py
```

O consolidator lê os artefatos Windows locais e a referência `origin/codex/mac-training-split`. Os hashes dos dados produzidos ficam em `data/checksums.sha256`.
