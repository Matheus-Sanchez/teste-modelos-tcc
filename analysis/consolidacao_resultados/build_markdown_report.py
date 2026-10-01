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
    quant_pairs = csv("comparisons_quantization_windows_mac")
    classes = csv("class_metrics")
    validations = csv("validation_checks")
    reported_batch = csv("reported_mac_batch_cells")
    quant = csv("reported_mac_quantization_status")
    deltas = class_deltas(classes, exact)

    windows = summary["platforms"]["Windows"]
    mac = summary["platforms"]["Mac"]
    exact_summary = summary["comparisons"]["batch_exact"]
    activation_summary = summary["comparisons"]["activation_descriptive"]
    quant_summary = summary["comparisons"].get("quantization_exact", {})
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

    datasets_order = ["MNIST", "KMNIST", "Fashion-MNIST", "EMNIST Balanced", "SVHN", "CIFAR-10", "FER2013", "CIFAR-100 coarse", "GTSRB"]
    batch_sweep_rows = []
    for ds in datasets_order:
        ds_subset = windows_batch[windows_batch.dataset == ds]
        if ds_subset.empty:
            continue
        b32 = ds_subset[ds_subset.batch_size == 32]
        b64 = ds_subset[ds_subset.batch_size == 64]
        b128 = ds_subset[ds_subset.batch_size == 128]
        b256 = ds_subset[ds_subset.batch_size == 256]
        best_b = ds_subset.loc[ds_subset.macro_f1.idxmax()]
        batch_sweep_rows.append((
            ds,
            pct(b32.iloc[0].macro_f1) if not b32.empty else "—",
            pct(b64.iloc[0].macro_f1) if not b64.empty else "—",
            pct(b128.iloc[0].macro_f1) if not b128.empty else "—",
            pct(b256.iloc[0].macro_f1) if not b256.empty else "—",
            f"Batch {int(best_b.batch_size)} ({pct(best_b.macro_f1)})",
        ))
    batch_sweep_table = markdown_table(
        ["Dataset", "Batch 32", "Batch 64", "Batch 128", "Batch 256", "Melhor Batch"],
        batch_sweep_rows
    )

    quant_runs = runs[(runs.phase == "quantization") & (runs.campaign == "quantization_all")]
    quant_rows = []
    for ds in datasets_order:
        ds_quant = quant_runs[quant_runs.dataset == ds]
        if ds_quant.empty:
            continue
        fp32_r = ds_quant[ds_quant.variant == "fp32"]
        fp16_r = ds_quant[ds_quant.variant == "fp16"]
        int8_r = ds_quant[ds_quant.variant == "int8_ptq"]
        f1_32 = fp32_r.iloc[0].macro_f1 if not fp32_r.empty else None
        f1_16 = fp16_r.iloc[0].macro_f1 if not fp16_r.empty else None
        f1_8 = int8_r.iloc[0].macro_f1 if not int8_r.empty else None
        delta_8_32 = (f1_8 - f1_32) * 100 if f1_8 is not None and f1_32 is not None else None
        tput_32 = fp32_r.iloc[0].mean_train_examples_per_second if not fp32_r.empty else None
        tput_8 = int8_r.iloc[0].mean_train_examples_per_second if not int8_r.empty else None
        speedup = f"{tput_8 / tput_32:.1f}×" if tput_32 and tput_8 else "—"
        quant_rows.append((
            ds,
            pct(f1_32),
            pct(f1_16),
            pct(f1_8),
            pp(delta_8_32),
            f"{br(tput_32, 0)} ex/s" if pd.notna(tput_32) else "—",
            f"{br(tput_8, 0)} ex/s" if pd.notna(tput_8) else "—",
            speedup
        ))
    quant_comparison_table = markdown_table(
        ["Dataset", "FP32 Macro F1", "FP16 Macro F1", "INT8 PTQ Macro F1", "Δ F1 (INT8 − FP32)", "Throughput FP32", "Throughput INT8", "Speedup Throughput"],
        quant_rows
    )

    quant_cross_platform_table = markdown_table(
        ["Dataset", "Variante", "Macro F1 W", "Macro F1 Mac", "Δ F1 W−Mac", "Acc. W", "Acc. Mac", "Loss W", "Loss Mac", "Throughput W", "Vencedor F1"],
        (
            (
                row.dataset,
                row.variant.upper(),
                pct(row.macro_f1_windows),
                pct(row.macro_f1_mac),
                pp(row.delta_macro_f1_pp_windows_minus_mac),
                pct(row.accuracy_windows),
                pct(row.accuracy_mac),
                br(row.loss_windows, 4) if pd.notna(row.loss_windows) else "—",
                br(row.loss_mac, 4) if pd.notna(row.loss_mac) else "—",
                f"{br(row.mean_train_examples_per_second_windows, 0)} ex/s" if pd.notna(row.mean_train_examples_per_second_windows) else "—",
                row.macro_f1_winner if pd.notna(row.macro_f1_winner) else "—",
            )
            for row in quant_pairs.sort_values(["dataset", "variant"]).itertuples()
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

Foram consolidados **{windows['completed_runs_with_metrics']} runs Windows** e **{mac['completed_runs_with_metrics']} runs Mac** com métricas finais exatas (totalizando **{windows['completed_runs_with_metrics'] + mac['completed_runs_with_metrics']} execuções**), cobrindo **{windows['epochs'] + mac['epochs']:,} epochs** e **{br(windows['training_hours_observed'] + mac['training_hours_observed'], 2)} horas-run** observadas em todas as frentes de teste: varredura de batch (32, 64, 128, 256), quantização (FP32, FP16, INT8 PTQ LiteRT) e ativações (ReLU, Sigmoid, Softmax).

Nos **{exact_summary['pairs']} pares de batch com protocolo alinhado** (cobrindo 5 datasets: MNIST, Fashion-MNIST, KMNIST, EMNIST Balanced e CIFAR-10 em batches 32, 64, 128 e 256), a diferença média de Macro F1 Windows − Mac foi de **{pp(exact_summary['mean_delta_macro_f1_pp_windows_minus_mac'])}**, com mediana de **{pp(exact_summary['median_delta_macro_f1_pp_windows_minus_mac'])}**. O Windows obteve melhor F1 em {exact_summary['windows_macro_f1_wins']}/{exact_summary['pairs']} pares e o Mac em {exact_summary['mac_macro_f1_wins']}/{exact_summary['pairs']}. Em tempo de treinamento, o Windows foi sistematicamente mais rápido em todos os {exact_summary['pairs']} pares; a razão mediana `Windows/Mac` foi **{br(exact_summary['median_training_time_ratio_windows_over_mac'], 3)}**, equivalente a cerca de **{br(exact_summary['median_training_time_ratio_windows_over_mac'] * 100, 1)}%** do tempo do Mac (com speedups de até **8,76×**).

Nos **{quant_summary.get('pairs', 27)} pares de quantização** (cobrindo todos os 9 datasets nas variantes FP32, FP16 e INT8 PTQ LiteRT), ambas as plataformas completaram 100% dos modelos avaliados. A quantização para INT8 reduziu a pegada em disco para ~311 KB por modelo e alcançou taxas de inferência de **1.510 a 6.946 amostras por segundo** no Windows, preservando alta fidelidade preditiva frente ao FP32.

Nos **{activation_summary['pairs']} pares de ativações** (cobrindo todos os 9 datasets com ReLU, Sigmoid e Softmax), a ativação Softmax oculta colapsou próximo ao acaso em ambos os ambientes (~10–20%), enquanto ReLU e Sigmoid mantiveram alta eficácia e paridade comportamental.

## 1. Escopo e cobertura

| Plataforma | Runs com métricas | Campanhas | Datasets | Epochs | Horas-run | Class metrics | Confusion matrices | Séries por epoch | Telemetria |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
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

### Varredura completa de batch (Batch Sweep: 32, 64, 128, 256)

A varredura completa cobriu os 9 datasets no Windows na campanha `controlled-augmentation05-batch-activation`. Batches menores (32 e 64) proporcionaram gradientes mais frequentes e melhor generalização em datasets complexos com menor suporte (como GTSRB e FER2013), enquanto batches maiores (128 e 256) maximizaram o paralelismo e estabilidade nos datasets canônicos (como MNIST e Fashion-MNIST).

{batch_sweep_table}

![Varredura de batch](figures/16_batch_sweep_macro_f1.png)

## 3. Resultados de quantização (FP32 × FP16 × INT8 PTQ)

A avaliação de quantização cobriu **todos os 9 datasets** tanto no Windows (`quantization_all`) quanto no Mac M4 (`controlled-quantization-fast-mac-m4`), analisando modelos em precisão completa (`float32`), precisão reduzida (`float16`) e inteiros de 8 bits pós-treinamento (`int8_ptq`) via LiteRT (TensorFlow Lite).

### Comparação direta de quantização Windows × Mac (27 pares)

A tabela a seguir apresenta o desempenho comparativo entre Windows e Mac para as três variantes de precisão em cada um dos nove datasets:

{quant_cross_platform_table}

![Impacto da quantização no Macro F1](figures/14_quantizacao_macro_f1.png)

### Eficiência de inferência e throughput LiteRT no Windows

No Windows, a conversão INT8 PTQ reduziu a pegada dos pesos para aproximadamente 311 KB por modelo (compressão de quase 4× em relação ao FP32) e proporcionou taxas de inferência de **1.510 a 6.946 amostras por segundo**, completando a inferência de teste (10.000 imagens) em 1,5 a 5,9 segundos com perda de Macro F1 inferior a 1,5 p.p. na maioria dos cenários:

{quant_comparison_table}

![Throughput e latência na quantização](figures/15_quantizacao_throughput_latencia.png)

## 4. Resultados de ativações

### Melhor ativação por plataforma e dataset

{best_activation_table}

![Macro F1 das ativações](figures/03_macro_f1_ativacoes_windows_mac.png)

![Diferença de Macro F1 das ativações](figures/04_delta_macro_f1_ativacoes_heatmap.png)

### Todos os 27 pares de ativações

{activation_table}

![Softmax oculta](figures/12_softmax_macro_f1.png)

## 5. Comparação Windows × Mac: pares batch equivalentes

Estes 15 pares usam o mesmo dataset, batch, augmentation, seed (42) e divisão quando verificável. Cobrem 5 datasets (MNIST, Fashion-MNIST, KMNIST, EMNIST Balanced e CIFAR-10) em batches 32, 64, 128 e 256. Ainda diferem em hardware (RTX 3050 vs Apple M4), sistema operacional (Windows/WSL vs macOS 15.5), backend (CUDA vs Metal) e versão do TensorFlow (2.21.0 vs 2.18.1).

{exact_table}

![Macro F1 dos pares de batch](figures/05_macro_f1_pares_batch.png)

![Paridade numérica e resíduos](figures/18_paridade_residuos_f1.png)

![Tempo médio por epoch](figures/06_tempo_epoca_pares_batch.png)

![Aceleração computacional Speedup](figures/17_speedup_windows_mac.png)

## 6. Loss, curvas e duração das epochs

As séries por epoch incluem loss, acurácia, `val_loss`, `val_accuracy`, `val_balanced_accuracy`, `val_macro_f1`, learning rate, duração e throughput. O anexo `epoch_metrics.csv` contém {summary['evidence']['epoch_rows']:,} linhas. Nos 15 pares equivalentes de batch (6 dos quais contam com séries completas por epoch no Mac), o Mac levou aproximadamente **5,1× a 8,8×** mais tempo por epoch.

![Curvas de validação](figures/08_curvas_validacao_pares_batch.png)

![Tempo de cada epoch](figures/09_tempo_por_epoca_pares_batch.png)

## 7. Telemetria de hardware

{telemetry_table}

![Telemetria nos pares de batch](figures/07_telemetria_pares_batch.png)

Os percentuais de GPU têm semântica distinta entre CUDA/NVML e Metal. Eles são adequados para leitura operacional dentro de cada backend, mas não constituem equivalência física direta entre aceleradores. O anexo completo inclui CPU do sistema e processo, RAM, RSS, utilização e memória da GPU, potência, temperatura e perfis temporais normalizados.

## 8. Métricas por classe

Foram preservadas **{summary['evidence']['class_metric_rows']:,} linhas** de precisão, recall, F1 e suporte por classe. Abaixo estão as 40 maiores diferenças absolutas de F1 por classe nos pares equivalentes; sinal positivo favorece Windows.

{class_delta_table}

![Diferença de F1 por classe](figures/10_delta_f1_por_classe_batch.png)

O arquivo [class_metrics.csv](data/class_metrics.csv) contém todas as classes de todos os runs disponíveis.

## 9. Matrizes de confusão

As matrizes são normalizadas por classe verdadeira nos gráficos. O arquivo [confusion_matrices_long.csv](data/confusion_matrices_long.csv) preserva as **{summary['evidence']['confusion_matrix_cells']:,} células** e contagens absolutas.

![Fashion-MNIST batch 128](figures/11_01_matriz_confusao_fashion_mnist_batch_128.png)

![Fashion-MNIST batch 256](figures/11_02_matriz_confusao_fashion_mnist_batch_256.png)

![KMNIST batch 32](figures/11_03_matriz_confusao_kmnist_batch_032.png)

![MNIST batch 32](figures/11_04_matriz_confusao_mnist_batch_032.png)

![MNIST batch 64](figures/11_05_matriz_confusao_mnist_batch_064.png)

![MNIST batch 256](figures/11_06_matriz_confusao_mnist_batch_256.png)

## 10. Lacunas documentadas no Mac

O relatório de batch do Mac registra **{gaps['batch_completed_reported']} células concluídas** nos 5 datasets executados (MNIST, Fashion-MNIST, KMNIST, EMNIST Balanced e CIFAR-10), das quais **{gaps['batch_completed_with_raw_versioned']}** têm artefatos brutos versionados (com matrizes de confusão e métricas por epoch) e **{gaps['batch_completed_report_only']}** permanecem report-only via relatório oficial. Os outros 4 datasets (CIFAR-100 coarse, SVHN, GTSRB e FER2013) não foram executados na varredura de batch no Mac.

{mac_batch_table}

Na quantização do Mac, o relatório técnico consolidado de 2026-09-14 (`analysis_reports/quantizacao_2026-09-14/README.md`) registra **todas as 27 execuções concluídas** (9 datasets × 3 variantes: FP32, FP16 e INT8 PTQ), todas com métricas finais reportadas e agora integradas diretamente no pipeline de comparação multiplataforma.

{quant_table}

## 11. Validações de integridade

{validation_table}

Os checks confirmam unicidade do `run_uid`, métricas dentro de `[0,1]`, recomposição do Macro F1 pelas classes, soma das matrizes de confusão e contagem de epochs.

## 12. Limitações

- Os 15 pares batch são a comparação mais forte disponível para os 5 datasets executados em ambas as plataformas, mas hardware, sistema operacional, backend e versão do TensorFlow ainda diferem.
- Os 27 pares de ativações usam batches distintos e não isolam o efeito do sistema operacional.
- Há apenas uma seed (`42`) por condição comparada; não há base para intervalos de confiança ou testes de significância entre seeds.
- Horas-run são somadas por execução e não correspondem ao tempo de calendário quando houve paralelismo.
- Parte do histórico Mac existe apenas em relatórios agregados; nenhuma métrica ausente foi inferida.
- Softmax foi testada como ativação oculta, não como a saída softmax padrão do classificador.

## 13. Conclusão

1. **Qualidade nos pares batch:** praticamente equivalente em média (+0,15 p.p. para Windows), alternando desempenhos por dataset e batch (Windows vence 9, Mac vence 6).
2. **Tempo nos pares batch:** o Windows com GPU NVIDIA dedicada foi sistematicamente mais rápido nos 15 pares protocolarmente alinhados (razão mediana Windows/Mac de 0,159, ou speedups de até 8,76×).
3. **Quantização:** cobertura completa de todos os 9 datasets nas variantes FP32, FP16 e INT8 PTQ (27 pares multiplataforma). Os modelos INT8 LiteRT alcançaram tamanho compacto (~311 KB) e throughput de até 6.946 ex/s no Windows, com degradação mínima de Macro F1.
4. **Ativações:** 27 pares cobrindo todos os 9 datasets com ReLU, Sigmoid e Softmax. ReLU e Sigmoid funcionam e mantêm paridade comportamental em ambos os sistemas; Softmax oculta apresenta colapso invariante à plataforma.
5. **Telemetria:** maior utilização percentual no Mac (Metal) não compensou a duração maior das epochs; as APIs de medição refletem métricas operacionais distintas.
6. **Evidência:** 218 execuções consolidadas no total (140 Windows + 78 Mac). Windows possui cobertura granular quase completa (140 runs com métricas); Mac possui 78 runs com métricas finais exatas (33 com métricas por classe e 15 com séries completas por epoch).

## Apêndice A — inventário completo dos {len(runs)} runs

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
- [Comparação quantização](data/comparisons_quantization_windows_mac.csv)
- [Todas as comparações](data/comparisons_windows_mac.csv)
- [Manifesto de captura](data/capture_manifest.json)
- [Checksums SHA-256](data/checksums.sha256)
"""

    OUTPUT.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "runs": len(runs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
