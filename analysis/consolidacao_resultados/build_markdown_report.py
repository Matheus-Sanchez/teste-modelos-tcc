from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUTPUT = BASE / "RELATORIO_CONSOLIDADO.md"


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def br(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{br(number * 100, digits)}%" if np.isfinite(number) else "—"


def pp(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{'+' if number > 0 else ''}{br(number, digits)} p.p." if np.isfinite(number) else "—"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    headers = [str(header) for header in headers]
    body = [list(row) for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in body:
        values = []
        for value in row:
            text = "—" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def boolean(value: object) -> str:
    if pd.isna(value):
        return "n/d"
    return "sim" if bool(value) else "não"


def class_deltas(classes: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, pair in pairs.iterrows():
        windows = classes[classes.run_uid == pair.windows_run_uid][["class_index", "class_name", "support", "f1"]].rename(
            columns={"class_name": "class_name_windows", "support": "support_windows", "f1": "f1_windows"}
        )
        mac = classes[classes.run_uid == pair.mac_run_uid][["class_index", "class_name", "support", "f1"]].rename(
            columns={"class_name": "class_name_mac", "support": "support_mac", "f1": "f1_mac"}
        )
        merged = windows.merge(mac, on="class_index", how="inner")
        if merged.empty:
            continue
        merged.insert(0, "dataset", pair.dataset)
        merged.insert(1, "batch", int(pair.batch_size_windows))
        merged["class_name"] = merged.class_name_windows.fillna(merged.class_name_mac)
        merged["delta_f1_pp"] = (merged.f1_windows - merged.f1_mac) * 100
        merged["abs_delta_f1_pp"] = merged.delta_f1_pp.abs()
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    runs = csv("run_results")
    campaigns = csv("campaign_summary")
    exact = csv("comparisons_batch_windows_mac")
    activations = csv("comparisons_activation_windows_mac")
    classes = csv("class_metrics")
    validations = csv("validation_checks")
    reported_batch = csv("reported_mac_batch_cells")
    quant = csv("reported_mac_quantization_status")
    deltas = class_deltas(classes, exact)

    windows = summary["platforms"]["Windows"]
    mac = summary["platforms"]["Mac"]
    exact_summary = summary["comparisons"]["batch_exact"]
    activation_summary = summary["comparisons"]["activation_descriptive"]
    gaps = summary["reported_mac_gaps"]

    campaign_table = markdown_table(
        ["Plataforma", "Campanha", "Fase", "Runs", "Datasets", "Epochs", "Horas-run", "Macro F1 médio", "Mediana/epoch", "GPU média", "RAM média"],
        (
            (
                row.platform,
                row.campaign,
                row.phase,
                int(row.completed_runs),
                int(row.datasets),
                int(row.epochs),
                f"{br(row.training_hours, 2)} h",
                pct(row.mean_macro_f1),
                f"{br(row.median_epoch_seconds, 1)} s",
                f"{br(row.weighted_gpu_util_mean_pct, 1)}%",
                f"{br(row.weighted_ram_mean_pct, 1)}%",
            )
            for row in campaigns.itertuples()
        ),
    )

    exact_table = markdown_table(
        ["Dataset", "Batch", "Acc. W", "Acc. Mac", "Macro F1 W", "Macro F1 Mac", "Δ F1 W−Mac", "Loss W", "Loss Mac", "Epoch W", "Epoch Mac", "Mac/Windows"],
        (
            (
                row.dataset,
                int(row.batch_size_windows),
                pct(row.accuracy_windows),
                pct(row.accuracy_mac),
                pct(row.macro_f1_windows),
                pct(row.macro_f1_mac),
                pp(row.delta_macro_f1_pp_windows_minus_mac),
                br(row.loss_windows, 4),
                br(row.loss_mac, 4),
                f"{br(row.mean_epoch_seconds_windows, 1)} s",
                f"{br(row.mean_epoch_seconds_mac, 1)} s",
                f"{br(row.mean_epoch_seconds_mac / row.mean_epoch_seconds_windows, 2)}×",
            )
            for row in exact.sort_values(["dataset", "batch_size_windows"]).itertuples()
        ),
    )

    activation_table = markdown_table(
        ["Dataset", "Ativação", "Batch W", "Batch Mac", "Acc. W", "Acc. Mac", "Macro F1 W", "Macro F1 Mac", "Δ F1 W−Mac", "Loss W", "Loss Mac", "Epoch W", "Epoch Mac"],
        (
            (
                row.dataset,
                row.activation,
                int(row.batch_size_windows),
                int(row.batch_size_mac),
                pct(row.accuracy_windows),
                pct(row.accuracy_mac),
                pct(row.macro_f1_windows),
                pct(row.macro_f1_mac),
                pp(row.delta_macro_f1_pp_windows_minus_mac),
                br(row.loss_windows, 4),
                br(row.loss_mac, 4),
                f"{br(row.mean_epoch_seconds_windows, 1)} s",
                f"{br(row.mean_epoch_seconds_mac, 1)} s",
            )
            for row in activations.sort_values(["dataset", "activation"]).itertuples()
        ),
    )

    telemetry_table = markdown_table(
        ["Dataset", "Batch", "GPU W", "GPU Mac", "Mem. GPU W", "Mem. GPU Mac", "CPU proc. W", "CPU proc. Mac", "RAM W", "RAM Mac"],
        (
            (
                row.dataset,
                int(row.batch_size_windows),
                f"{br(row.gpu_util_pct_mean_windows, 1)}%",
                f"{br(row.gpu_util_pct_mean_mac, 1)}%",
                f"{br(row.gpu_memory_used_bytes_mean_windows / 1024**3, 2)} GiB",
                f"{br(row.gpu_memory_used_bytes_mean_mac / 1024**3, 2)} GiB",
                f"{br(row.process_cpu_pct_mean_windows, 1)}%",
                f"{br(row.process_cpu_pct_mean_mac, 1)}%",
                f"{br(row.ram_pct_mean_windows, 1)}%",
                f"{br(row.ram_pct_mean_mac, 1)}%",
            )
            for row in exact.sort_values(["dataset", "batch_size_windows"]).itertuples()
        ),
    )

    windows_batch = runs[(runs.platform == "Windows") & (runs.campaign == "controlled-augmentation05-batch-activation") & (runs.phase == "batch")]
    best_windows = windows_batch.loc[windows_batch.groupby("dataset_key").macro_f1.idxmax()].sort_values("dataset")
    best_batch_table = markdown_table(
        ["Dataset", "Batch vencedor Windows", "Macro F1", "Acurácia", "Loss", "Média/epoch"],
        (
            (row.dataset, int(row.batch_size), pct(row.macro_f1), pct(row.accuracy), br(row.loss, 4), f"{br(row.mean_epoch_seconds, 1)} s")
            for row in best_windows.itertuples()
        ),
    )

    activation_runs = runs[runs.phase == "activation"]
    best_activation = activation_runs.loc[activation_runs.groupby(["platform", "dataset_key"]).macro_f1.idxmax()].sort_values(["platform", "dataset"])
    best_activation_table = markdown_table(
        ["Plataforma", "Dataset", "Ativação vencedora", "Macro F1", "Acurácia", "Loss", "Batch"],
        (
            (row.platform, row.dataset, row.activation, pct(row.macro_f1), pct(row.accuracy), br(row.loss, 4), int(row.batch_size))
            for row in best_activation.itertuples()
        ),
    )

    hardware = (
        runs.groupby(["platform", "gpu_name", "gpu_backend", "tensorflow_version", "tensorflow_metal_version"], dropna=False)
        .agg(runs=("run_uid", "nunique"), horas=("training_hours", "sum"))
        .reset_index()
        .sort_values(["platform", "runs"], ascending=[True, False])
    )
    hardware_table = markdown_table(
        ["Plataforma", "GPU", "Backend", "TensorFlow", "tensorflow-metal", "Runs", "Horas-run"],
        (
            (
                row.platform,
                row.gpu_name if pd.notna(row.gpu_name) else "não registrado",
                row.gpu_backend if pd.notna(row.gpu_backend) else "não registrado",
                row.tensorflow_version if pd.notna(row.tensorflow_version) else "não registrado",
                row.tensorflow_metal_version if pd.notna(row.tensorflow_metal_version) else "—",
                int(row.runs),
                f"{br(row.horas, 2)} h",
            )
            for row in hardware.itertuples()
        ),
    )

    class_delta_table = markdown_table(
        ["Dataset", "Batch", "Classe", "Suporte W", "Suporte Mac", "F1 W", "F1 Mac", "Δ F1 W−Mac"],
        (
            (
                row.dataset,
                int(row.batch),
                row.class_name,
                int(row.support_windows),
                int(row.support_mac),
                pct(row.f1_windows),
                pct(row.f1_mac),
                pp(row.delta_f1_pp),
            )
            for row in deltas.sort_values("abs_delta_f1_pp", ascending=False).head(40).itertuples()
        ),
    )

    all_runs_table = markdown_table(
        ["Plataforma", "Campanha", "Fase", "Dataset", "Variante", "Acc.", "Bal. Acc.", "Macro P", "Macro R", "Macro F1", "Loss", "Epochs", "Horas", "s/epoch", "GPU"],
        (
            (
                row.platform,
                row.campaign,
                row.phase,
                row.dataset,
                row.variant,
                pct(row.accuracy),
                pct(row.balanced_accuracy),
                pct(row.macro_precision),
                pct(row.macro_recall),
                pct(row.macro_f1),
                br(row.loss, 4),
                int(row.epochs_completed) if pd.notna(row.epochs_completed) else "—",
                br(row.training_hours, 2),
                br(row.mean_epoch_seconds, 1),
                row.gpu_name if pd.notna(row.gpu_name) else "n/d",
            )
            for row in runs.sort_values(["platform", "campaign", "phase", "dataset", "variant"]).itertuples()
        ),
    )

    validation_table = markdown_table(
        ["Check", "Status", "Observado", "Esperado", "Detalhe"],
        ((row.check, row.status, row.observed, row.expected, row.detail) for row in validations.itertuples()),
    )

    mac_batch_table = markdown_table(
        ["Dataset", "Batch", "Status", "Bruto versionado", "Melhor reportado", "Macro F1 reportado", "Evidência"],
        (
            (
                row.dataset,
                int(row.batch_size),
                row.status,
                boolean(row.raw_artifact_versioned),
                boolean(row.is_reported_best),
                pct(row.reported_macro_f1) if pd.notna(row.reported_macro_f1) else "—",
                row.evidence_level,
            )
            for row in reported_batch.sort_values(["dataset", "batch_size"]).itertuples()
        ),
    )

    quant_table = markdown_table(
        ["Dataset", "Variante", "Status", "Epochs observadas", "Métricas finais reportadas", "Valores exatos disponíveis", "Evidência"],
        (
            (
                row.dataset,
                row.variant,
                row.status,
                int(row.epochs_observed),
                boolean(row.has_final_metrics_reported),
                boolean(row.exact_metric_values_available),
                row.evidence_level,
            )
            for row in quant.sort_values(["dataset", "variant"]).itertuples()
        ),
    )

    report = f"""# Relatório consolidado dos treinamentos — Windows × Mac

Snapshot UTC: `{summary['snapshot_at']}`  
Branch: `{summary['branch']}`  
Commit Windows: `{summary['head_commit']}`  
Referência Mac: `{summary['mac_ref']}` (`{summary['mac_commit']}`)

## Resumo executivo

Foram consolidados **{windows['completed_runs_with_metrics']} runs Windows** e **{mac['completed_runs_with_metrics']} runs Mac** com métricas finais exatas, totalizando **{windows['epochs'] + mac['epochs']:,} epochs** e **{br(windows['training_hours_observed'] + mac['training_hours_observed'], 2)} horas-run** observadas.

Nos **seis pares de batch com protocolo alinhado**, a diferença média de Macro F1 Windows − Mac foi de **{pp(exact_summary['mean_delta_macro_f1_pp_windows_minus_mac'])}**, com mediana de **{pp(exact_summary['median_delta_macro_f1_pp_windows_minus_mac'])}**. O Windows venceu 2/6 pares e o Mac 4/6. Em tempo, o Windows foi mais rápido em todos os seis; a razão mediana `Windows/Mac` foi **{br(exact_summary['median_training_time_ratio_windows_over_mac'], 3)}**, equivalente a cerca de **{br(exact_summary['median_training_time_ratio_windows_over_mac'] * 100, 1)}%** do tempo do Mac.

Nos **21 pares de ativações**, a diferença média descritiva foi **{pp(activation_summary['mean_delta_macro_f1_pp_windows_minus_mac'])}**, mas o batch e a pilha de software diferem entre ambientes. Esse resultado não isola um efeito causal do sistema operacional. Softmax usada como ativação oculta colapsou próximo ao acaso nos dois ambientes; ReLU e Sigmoid concentram os resultados úteis.

## 1. Escopo e cobertura

| Plataforma | Runs com métricas | Campanhas | Datasets | Epochs | Horas-run | Class metrics | Confusion matrices | Séries por epoch | Telemetria |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Windows | {windows['completed_runs_with_metrics']} | {windows['campaigns']} | {windows['datasets']} | {windows['epochs']} | {br(windows['training_hours_observed'], 2)} | {windows['runs_with_class_metrics']} | {windows['runs_with_confusion_matrix']} | {windows['runs_with_epoch_metrics']} | {windows['runs_with_telemetry_summary']} |
| Mac | {mac['completed_runs_with_metrics']} | {mac['campaigns']} | {mac['datasets']} | {mac['epochs']} | {br(mac['training_hours_observed'], 2)} | {mac['runs_with_class_metrics']} | {mac['runs_with_confusion_matrix']} | {mac['runs_with_epoch_metrics']} | {mac['runs_with_telemetry_summary']} |

![Cobertura dos resultados](figures/01_cobertura_resultados.png)

### Campanhas

{campaign_table}

![Tempo acumulado por campanha](figures/02_tempo_acumulado_campanhas.png)

### Hardware e runtime

{hardware_table}

O Windows combina execução host Windows com treinamento em WSL2/Linux. Os resultados históricos usam NVIDIA GeForce RTX 3050 ou RTX A2000 12 GB e TensorFlow 2.21.0. O Mac usa Apple M4, macOS 15.5, TensorFlow 2.18.1 e tensorflow-metal 1.2.0. Uma execução Windows não preservou todos os campos de ambiente.

## 2. Resultados de batch no Windows

### Melhor batch observado por dataset

{best_batch_table}

![Melhores batches conhecidos](figures/13_melhores_batches_por_dataset.png)

## 3. Resultados de ativações

### Melhor ativação por plataforma e dataset

{best_activation_table}

![Macro F1 das ativações](figures/03_macro_f1_ativacoes_windows_mac.png)

![Diferença de Macro F1 das ativações](figures/04_delta_macro_f1_ativacoes_heatmap.png)

### Todos os 21 pares de ativações

{activation_table}

![Softmax oculta](figures/12_softmax_macro_f1.png)

## 4. Comparação Windows × Mac: pares batch equivalentes

Estes pares usam o mesmo dataset, batch, augmentation, seed e fingerprint de split. Ainda diferem em hardware, sistema, backend e versão do TensorFlow.

{exact_table}

![Macro F1 dos pares de batch](figures/05_macro_f1_pares_batch.png)

![Tempo médio por epoch](figures/06_tempo_epoca_pares_batch.png)

## 5. Loss, curvas e duração das epochs

As séries por epoch incluem loss, acurácia, `val_loss`, `val_accuracy`, `val_balanced_accuracy`, `val_macro_f1`, learning rate, duração e throughput. O anexo `epoch_metrics.csv` contém {summary['evidence']['epoch_rows']:,} linhas. Nos seis pares equivalentes, o Mac levou aproximadamente **5,1× a 6,6×** mais tempo por epoch.

![Curvas de validação](figures/08_curvas_validacao_pares_batch.png)

![Tempo de cada epoch](figures/09_tempo_por_epoca_pares_batch.png)

## 6. Telemetria de hardware

{telemetry_table}

![Telemetria nos pares de batch](figures/07_telemetria_pares_batch.png)

Os percentuais de GPU têm semântica distinta entre CUDA/NVML e Metal. Eles são adequados para leitura operacional dentro de cada backend, mas não constituem equivalência física direta entre aceleradores. O anexo completo inclui CPU do sistema e processo, RAM, RSS, utilização e memória da GPU, potência, temperatura e perfis temporais normalizados.

## 7. Métricas por classe

Foram preservadas **{summary['evidence']['class_metric_rows']:,} linhas** de precisão, recall, F1 e suporte por classe. Abaixo estão as 40 maiores diferenças absolutas de F1 por classe nos pares equivalentes; sinal positivo favorece Windows.

{class_delta_table}

![Diferença de F1 por classe](figures/10_delta_f1_por_classe_batch.png)

O arquivo [class_metrics.csv](data/class_metrics.csv) contém todas as classes de todos os runs disponíveis.

## 8. Matrizes de confusão

As matrizes são normalizadas por classe verdadeira nos gráficos. O arquivo [confusion_matrices_long.csv](data/confusion_matrices_long.csv) preserva as **{summary['evidence']['confusion_matrix_cells']:,} células** e contagens absolutas.

![Fashion-MNIST batch 128](figures/11_01_matriz_confusao_fashion_mnist_batch_128.png)

![Fashion-MNIST batch 256](figures/11_02_matriz_confusao_fashion_mnist_batch_256.png)

![KMNIST batch 32](figures/11_03_matriz_confusao_kmnist_batch_032.png)

![MNIST batch 32](figures/11_04_matriz_confusao_mnist_batch_032.png)

![MNIST batch 64](figures/11_05_matriz_confusao_mnist_batch_064.png)

![MNIST batch 256](figures/11_06_matriz_confusao_mnist_batch_256.png)

## 9. Lacunas documentadas no Mac

O relatório de batch do Mac registra **{gaps['batch_completed_reported']} células concluídas**, das quais **{gaps['batch_completed_with_raw_versioned']}** têm artefatos brutos versionados e **{gaps['batch_completed_report_only']}** permanecem report-only.

{mac_batch_table}

Na quantização do Mac, o relatório registra **{gaps['quantization_completed_reported']} concluídos**, **{gaps['quantization_partial_stale']} parcial/stale** e **{gaps['quantization_not_started']} não iniciados**. Os valores finais exatos não foram versionados, portanto nenhum deles entra nas comparações numéricas.

{quant_table}

## 10. Validações de integridade

{validation_table}

Os checks confirmam unicidade do `run_uid`, métricas dentro de `[0,1]`, recomposição do Macro F1 pelas classes, soma das matrizes de confusão e contagem de epochs.

## 11. Limitações

- Os seis pares batch são a comparação mais forte disponível, mas hardware, sistema operacional, backend e TensorFlow ainda diferem.
- Os 21 pares de ativações usam batches distintos e não isolam o efeito do sistema operacional.
- Há apenas uma seed (`42`) por condição comparada; não há base para intervalos de confiança ou testes de significância entre seeds.
- Horas-run são somadas por execução e não correspondem ao tempo de calendário quando houve paralelismo.
- Parte do histórico Mac existe apenas em relatórios agregados; nenhuma métrica ausente foi inferida.
- Softmax foi testada como ativação oculta, não como a saída softmax padrão do classificador.

## 12. Conclusão

1. **Qualidade nos pares batch:** empate prático em média, com resultados alternando por dataset e batch.
2. **Tempo:** o Windows foi sistematicamente mais rápido nos seis pares protocolarmente alinhados.
3. **Ativações:** ReLU e Sigmoid funcionam; Softmax oculta apresenta desempenho degenerado.
4. **Telemetria:** maior utilização percentual no Mac não compensou a duração maior das epochs; as APIs de medição não são diretamente equivalentes.
5. **Evidência:** Windows possui cobertura granular quase completa; parte dos resultados Mac permanece apenas em relatórios.

## Apêndice A — inventário completo dos 161 runs

{all_runs_table}

## Apêndice B — arquivos reproduzíveis

- [Notebook pré-executado](../../notebooks/consolidacao_resultados_windows_mac.ipynb)
- [Resultados por run](data/run_results.csv)
- [Métricas por classe](data/class_metrics.csv)
- [Matrizes de confusão](data/confusion_matrices_long.csv)
- [Métricas por epoch](data/epoch_metrics.csv)
- [Telemetria resumida](data/telemetry_metrics_long.csv)
- [Perfis temporais de telemetria](data/telemetry_profiles.csv)
- [Comparação batch](data/comparisons_batch_windows_mac.csv)
- [Comparação de ativações](data/comparisons_activation_windows_mac.csv)
- [Manifesto de captura](data/capture_manifest.json)
- [Checksums SHA-256](data/checksums.sha256)
"""

    OUTPUT.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "runs": len(runs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
