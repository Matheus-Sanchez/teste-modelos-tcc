from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "analysis" / "consolidacao_resultados"
DATA = BASE / "data"
FIGURES = BASE / "figures"
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "consolidacao_resultados_windows_mac.ipynb"


def lines(source: str) -> list[str]:
    source = source.strip("\n")
    return [line + "\n" for line in source.splitlines()]


def markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(source)}


def code(source: str, output: dict[str, Any] | list[dict[str, Any]] | None = None) -> dict[str, Any]:
    outputs: list[dict[str, Any]]
    if output is None:
        outputs = []
    elif isinstance(output, list):
        outputs = output
    else:
        outputs = [output]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs,
        "source": lines(source),
    }


def stream(text: str) -> dict[str, Any]:
    return {"name": "stdout", "output_type": "stream", "text": lines(text)}


def table_output(frame: pd.DataFrame, max_colwidth: int = 64) -> dict[str, Any]:
    display_frame = frame.copy()
    for column in display_frame.select_dtypes(include=["float"]).columns:
        display_frame[column] = display_frame[column].round(6)
    with pd.option_context(
        "display.max_rows",
        max(200, len(display_frame) + 5),
        "display.max_columns",
        max(50, len(display_frame.columns) + 5),
        "display.width",
        220,
        "display.max_colwidth",
        max_colwidth,
    ):
        plain = display_frame.to_string(index=False)
    html = display_frame.to_html(index=False, border=0, classes="dataframe compact", escape=True)
    return {
        "data": {"text/html": [html], "text/plain": lines(plain)},
        "execution_count": None,
        "metadata": {},
        "output_type": "execute_result",
    }


def image_output(path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "data": {"image/png": encoded, "text/plain": [f"<Figure: {path.name}>"]},
        "metadata": {},
        "output_type": "display_data",
    }


def image_cell(filename: str) -> dict[str, Any]:
    path = FIGURES / filename
    source = (
        "from IPython.display import Image, display\n"
        f'display(Image(filename=str(FIGURES / "{filename}")))'
    )
    return code(source, image_output(path))


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def class_pair_deltas(class_metrics: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, pair in pairs.iterrows():
        windows = class_metrics[class_metrics.run_uid == pair.windows_run_uid][
            ["class_index", "class_name", "support", "precision", "recall", "f1"]
        ].rename(
            columns={
                "class_name": "class_name_windows",
                "support": "support_windows",
                "precision": "precision_windows",
                "recall": "recall_windows",
                "f1": "f1_windows",
            }
        )
        mac = class_metrics[class_metrics.run_uid == pair.mac_run_uid][
            ["class_index", "class_name", "support", "precision", "recall", "f1"]
        ].rename(
            columns={
                "class_name": "class_name_mac",
                "support": "support_mac",
                "precision": "precision_mac",
                "recall": "recall_mac",
                "f1": "f1_mac",
            }
        )
        merged = windows.merge(mac, on="class_index", how="inner")
        if merged.empty:
            continue
        merged.insert(0, "pair_id", pair.pair_id)
        merged.insert(1, "dataset", pair.dataset)
        merged.insert(2, "batch_size", int(pair.batch_size_windows))
        merged["class_name"] = merged.class_name_windows.fillna(merged.class_name_mac)
        merged["delta_f1_pp_windows_minus_mac"] = (merged.f1_windows - merged.f1_mac) * 100
        merged["abs_delta_f1_pp"] = merged.delta_f1_pp_windows_minus_mac.abs()
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    runs = read("run_results")
    campaigns = read("campaign_summary")
    coverage = read("evidence_coverage")
    validation = read("validation_checks")
    batch_pairs = read("comparisons_batch_windows_mac")
    activation_pairs = read("comparisons_activation_windows_mac")
    quant_pairs = read("comparisons_quantization_windows_mac")
    all_pairs = read("comparisons_windows_mac")
    classes = read("class_metrics")
    epochs = read("epoch_metrics")
    telemetry = read("telemetry_profiles")
    mac_batch_reported = read("reported_mac_batch_cells")
    mac_quant_reported = read("reported_mac_quantization_status")
    unmatched = read("unmatched_runs")
    class_deltas = class_pair_deltas(classes, batch_pairs)

    quant_cross_columns = [
        "dataset",
        "variant",
        "macro_f1_windows",
        "macro_f1_mac",
        "delta_macro_f1_pp_windows_minus_mac",
        "accuracy_windows",
        "accuracy_mac",
        "loss_windows",
        "loss_mac",
        "mean_train_examples_per_second_windows",
        "macro_f1_winner",
    ]
    quant_cross_view = quant_pairs[quant_cross_columns].sort_values(["dataset", "variant"])

    platform_summary = pd.DataFrame(
        [
            {
                "plataforma": platform,
                "runs_com_metricas": values["completed_runs_with_metrics"],
                "campanhas": values["campaigns"],
                "datasets": values["datasets"],
                "epochs": values["epochs"],
                "horas_treinamento": values["training_hours_observed"],
                "runs_class_metrics": values["runs_with_class_metrics"],
                "runs_confusion_matrix": values["runs_with_confusion_matrix"],
                "runs_epoch_metrics": values["runs_with_epoch_metrics"],
                "runs_telemetria": values["runs_with_telemetry_summary"],
            }
            for platform, values in summary["platforms"].items()
        ]
    )

    campaign_view = campaigns[
        [
            "platform",
            "campaign",
            "phase",
            "completed_runs",
            "datasets",
            "epochs",
            "training_hours",
            "mean_macro_f1",
            "mean_accuracy",
            "median_epoch_seconds",
            "weighted_gpu_util_mean_pct",
            "weighted_ram_mean_pct",
        ]
    ].sort_values(["platform", "campaign", "phase"])

    all_results = runs[
        [
            "platform",
            "campaign",
            "phase",
            "dataset",
            "variant",
            "batch_size",
            "activation",
            "quantization_variant",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "loss",
            "epochs_completed",
            "training_hours",
            "mean_epoch_seconds",
            "gpu_name",
            "tensorflow_version",
            "evidence_level",
        ]
    ].sort_values(["platform", "campaign", "phase", "dataset", "variant"])

    exact_view = batch_pairs[
        [
            "dataset",
            "batch_size_windows",
            "macro_f1_windows",
            "macro_f1_mac",
            "delta_macro_f1_pp_windows_minus_mac",
            "accuracy_windows",
            "accuracy_mac",
            "loss_windows",
            "loss_mac",
            "mean_epoch_seconds_windows",
            "mean_epoch_seconds_mac",
            "epoch_time_ratio_windows_over_mac",
            "macro_f1_winner",
            "training_time_winner",
            "same_split_fingerprint",
        ]
    ].sort_values(["dataset", "batch_size_windows"])

    activation_view = activation_pairs[
        [
            "dataset",
            "activation",
            "batch_size_windows",
            "batch_size_mac",
            "macro_f1_windows",
            "macro_f1_mac",
            "delta_macro_f1_pp_windows_minus_mac",
            "accuracy_windows",
            "accuracy_mac",
            "loss_windows",
            "loss_mac",
            "mean_epoch_seconds_windows",
            "mean_epoch_seconds_mac",
            "comparability_grade",
        ]
    ].sort_values(["dataset", "activation"])

    windows_batch = runs[(runs.platform == "Windows") & (runs.campaign == "controlled-augmentation05-batch-activation") & (runs.phase == "batch")]
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
        batch_sweep_rows.append({
            "dataset": ds,
            "batch_32_f1": b32.iloc[0].macro_f1 if not b32.empty else np.nan,
            "batch_64_f1": b64.iloc[0].macro_f1 if not b64.empty else np.nan,
            "batch_128_f1": b128.iloc[0].macro_f1 if not b128.empty else np.nan,
            "batch_256_f1": b256.iloc[0].macro_f1 if not b256.empty else np.nan,
            "melhor_batch": int(best_b.batch_size),
            "melhor_macro_f1": best_b.macro_f1,
        })
    batch_sweep_view = pd.DataFrame(batch_sweep_rows)

    quant_runs = runs[(runs.phase == "quantization") & (runs.campaign == "quantization_all")]
    quant_rows = []
    for ds in datasets_order:
        ds_quant = quant_runs[quant_runs.dataset == ds]
        if ds_quant.empty:
            continue
        fp32_r = ds_quant[ds_quant.variant == "fp32"]
        fp16_r = ds_quant[ds_quant.variant == "fp16"]
        int8_r = ds_quant[ds_quant.variant == "int8_ptq"]
        f1_32 = fp32_r.iloc[0].macro_f1 if not fp32_r.empty else np.nan
        f1_16 = fp16_r.iloc[0].macro_f1 if not fp16_r.empty else np.nan
        f1_8 = int8_r.iloc[0].macro_f1 if not int8_r.empty else np.nan
        delta_8_32 = (f1_8 - f1_32) * 100 if pd.notna(f1_8) and pd.notna(f1_32) else np.nan
        tput_32 = fp32_r.iloc[0].mean_train_examples_per_second if not fp32_r.empty else np.nan
        tput_8 = int8_r.iloc[0].mean_train_examples_per_second if not int8_r.empty else np.nan
        speedup = tput_8 / tput_32 if pd.notna(tput_32) and pd.notna(tput_8) and tput_32 > 0 else np.nan
        quant_rows.append({
            "dataset": ds,
            "fp32_macro_f1": f1_32,
            "fp16_macro_f1": f1_16,
            "int8_ptq_macro_f1": f1_8,
            "delta_int8_minus_fp32_pp": delta_8_32,
            "throughput_fp32_ex_s": tput_32,
            "throughput_int8_ex_s": tput_8,
            "speedup_throughput": speedup,
        })
    quant_view = pd.DataFrame(quant_rows)

    hardware = (
        runs.groupby(["platform", "gpu_name", "gpu_backend", "tensorflow_version", "tensorflow_metal_version"], dropna=False)
        .agg(runs=("run_uid", "nunique"), horas=("training_hours", "sum"))
        .reset_index()
        .sort_values(["platform", "runs"], ascending=[True, False])
    )

    telemetry_exact = batch_pairs[
        [
            "dataset",
            "batch_size_windows",
            "gpu_util_pct_mean_windows",
            "gpu_util_pct_mean_mac",
            "gpu_memory_used_bytes_mean_windows",
            "gpu_memory_used_bytes_mean_mac",
            "process_cpu_pct_mean_windows",
            "process_cpu_pct_mean_mac",
            "ram_pct_mean_windows",
            "ram_pct_mean_mac",
        ]
    ].copy()
    for column in [c for c in telemetry_exact if "bytes" in c]:
        telemetry_exact[column] = telemetry_exact[column] / (1024**3)
        telemetry_exact.rename(columns={column: column.replace("bytes", "gib")}, inplace=True)

    epoch_summary = (
        epochs.groupby(["platform", "campaign", "dataset", "variant", "run_uid"], dropna=False)
        .agg(
            epochs=("epoch", "count"),
            mediana_epoch_s=("epoch_seconds", "median"),
            melhor_val_macro_f1=("val_macro_f1", "max"),
            menor_val_loss=("val_loss", "min"),
            media_examples_s=("train_examples_per_second", "mean"),
        )
        .reset_index()
        .sort_values(["platform", "campaign", "dataset", "variant"])
    )

    class_delta_view = class_deltas.sort_values("abs_delta_f1_pp", ascending=False)[
        [
            "dataset",
            "batch_size",
            "class_index",
            "class_name",
            "support_windows",
            "support_mac",
            "f1_windows",
            "f1_mac",
            "delta_f1_pp_windows_minus_mac",
        ]
    ].head(40)

    inventory = pd.DataFrame(
        [
            ("run_results.csv", len(runs), "Uma linha por run com métricas finais, tempo, ambiente e telemetria resumida"),
            ("class_metrics.csv", len(classes), "Precisão, recall, F1 e suporte por classe"),
            ("confusion_matrices_long.csv", len(read("confusion_matrices_long")), "Matrizes de confusão em formato longo"),
            ("epoch_metrics.csv", len(epochs), "Loss, acurácia, Macro F1, tempo e throughput por epoch"),
            ("telemetry_metrics_long.csv", len(read("telemetry_metrics_long")), "Estatísticas de telemetria por run e métrica"),
            ("telemetry_profiles.csv", len(telemetry), "Perfis temporais normalizados de telemetria"),
            ("comparisons_batch_windows_mac.csv", len(batch_pairs), "Pares batch com configuração/split equivalentes"),
            ("comparisons_activation_windows_mac.csv", len(activation_pairs), "Pares de ativação descritivos, com batch diferente"),
            ("comparisons_quantization_windows_mac.csv", len(quant_pairs), "Pares de quantização cobrindo todos os 9 datasets (FP32, FP16, INT8 PTQ)"),
            ("comparisons_windows_mac.csv", len(all_pairs), "Todas as 69 comparações alinhadas entre plataformas"),
            ("unmatched_runs.csv", len(unmatched), "Runs sem contraparte comparável entre plataformas"),
        ],
        columns=["arquivo", "linhas", "conteudo"],
    )

    setup_source = '''from pathlib import Path
import json
import numpy as np
import pandas as pd

candidate_roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
ROOT = next(path for path in candidate_roots if (path / "analysis" / "consolidacao_resultados" / "data").exists())
BASE = ROOT / "analysis" / "consolidacao_resultados"
DATA = BASE / "data"
FIGURES = BASE / "figures"

def read(name):
    return pd.read_csv(DATA / f"{name}.csv")

summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
runs = read("run_results")
campaigns = read("campaign_summary")
coverage = read("evidence_coverage")
validation = read("validation_checks")
batch_pairs = read("comparisons_batch_windows_mac")
activation_pairs = read("comparisons_activation_windows_mac")
quant_pairs = read("comparisons_quantization_windows_mac")
all_pairs = read("comparisons_windows_mac")
classes = read("class_metrics")
confusion = read("confusion_matrices_long")
epochs = read("epoch_metrics")
telemetry = read("telemetry_profiles")
telemetry_metrics = read("telemetry_metrics_long")
mac_batch_reported = read("reported_mac_batch_cells")
mac_quant_reported = read("reported_mac_quantization_status")
unmatched = read("unmatched_runs")
print(f"Carregados: {len(runs)} runs, {len(classes)} linhas por classe, {len(epochs)} epochs e {len(telemetry)} bins de telemetria.")'''

    cells: list[dict[str, Any]] = [
        markdown(
            f'''# Consolidação dos resultados de treinamento — Windows × Mac

**Notebook reproduzível e pré-executado**  
Snapshot UTC: `{summary["snapshot_at"]}`  
Branch de consolidação: `{summary["branch"]}`  
Commit Windows: `{summary["head_commit"]}`  
Ref/commit Mac: `{summary["mac_ref"]}` / `{summary["mac_commit"]}`

Este notebook reúne os resultados finais, métricas por classe, matrizes de confusão, curvas por epoch, duração e telemetria. As comparações foram separadas em (1) pares batch equivalentes e (2) pares de ativações apenas descritivos, porque o batch e a pilha de software diferem entre os ambientes.'''
        ),
        markdown(
            '''## Como reproduzir

1. Execute `analysis/consolidacao_resultados/consolidate_results.py` para reconstruir os CSVs a partir dos artefatos Windows locais e da referência Git do Mac.
2. Execute `analysis/consolidacao_resultados/render_figures.py` para reconstruir os gráficos estáticos.
3. Abra este notebook a partir da raiz do repositório e execute todas as células em ordem.

O notebook não altera os artefatos originais de treinamento.'''
        ),
        code(setup_source, stream(f"Carregados: {len(runs)} runs, {len(classes)} linhas por classe, {len(epochs)} epochs e {len(telemetry)} bins de telemetria.")),
        markdown("## 1. Escopo e cobertura da evidência"),
        markdown(
            f'''Foram encontrados **{summary["platforms"]["Windows"]["completed_runs_with_metrics"]} runs Windows** e **{summary["platforms"]["Mac"]["completed_runs_with_metrics"]} runs Mac** com métricas finais exatas. Eles somam **{summary["platforms"]["Windows"]["epochs"] + summary["platforms"]["Mac"]["epochs"]:,} epochs** e **{summary["platforms"]["Windows"]["training_hours_observed"] + summary["platforms"]["Mac"]["training_hours_observed"]:.2f} horas-run** observadas.

No Mac, relatórios históricos registram execuções adicionais sem os artefatos numéricos completos versionados. Elas aparecem separadamente na seção de lacunas; não foram convertidas em métricas inventadas.'''
        ),
        code(
            '''platform_summary = pd.DataFrame([
    {
        "plataforma": platform,
        "runs_com_metricas": values["completed_runs_with_metrics"],
        "campanhas": values["campaigns"],
        "datasets": values["datasets"],
        "epochs": values["epochs"],
        "horas_treinamento": values["training_hours_observed"],
        "runs_class_metrics": values["runs_with_class_metrics"],
        "runs_confusion_matrix": values["runs_with_confusion_matrix"],
        "runs_epoch_metrics": values["runs_with_epoch_metrics"],
        "runs_telemetria": values["runs_with_telemetry_summary"],
    }
    for platform, values in summary["platforms"].items()
])
platform_summary''',
            table_output(platform_summary),
        ),
        image_cell("01_cobertura_resultados.png"),
        markdown("### Inventário dos dados normalizados"),
        code(
            '''inventory = pd.DataFrame([
    ("run_results.csv", len(runs), "Uma linha por run com métricas finais, tempo, ambiente e telemetria resumida"),
    ("class_metrics.csv", len(classes), "Precisão, recall, F1 e suporte por classe"),
    ("confusion_matrices_long.csv", len(confusion), "Matrizes de confusão em formato longo"),
    ("epoch_metrics.csv", len(epochs), "Loss, acurácia, Macro F1, tempo e throughput por epoch"),
    ("telemetry_metrics_long.csv", len(telemetry_metrics), "Estatísticas de telemetria por run e métrica"),
    ("telemetry_profiles.csv", len(telemetry), "Perfis temporais normalizados de telemetria"),
    ("comparisons_batch_windows_mac.csv", len(batch_pairs), "Pares batch com configuração/split equivalentes"),
    ("comparisons_activation_windows_mac.csv", len(activation_pairs), "Pares de ativação descritivos, com batch diferente"),
    ("unmatched_runs.csv", len(unmatched), "Runs sem contraparte comparável entre plataformas"),
], columns=["arquivo", "linhas", "conteudo"])
inventory''',
            table_output(inventory),
        ),
        markdown("## 2. Validações de integridade"),
        markdown(
            "Os checks abaixo confirmam unicidade dos identificadores, limites das métricas, recomposição do Macro F1 pelas classes, totais das matrizes e contagens de epochs."
        ),
        code("validation", table_output(validation)),
        markdown("## 3. Campanhas, duração e ambiente de execução"),
        code(
            'campaign_view = campaigns[["platform","campaign","phase","completed_runs","datasets","epochs","training_hours","mean_macro_f1","mean_accuracy","median_epoch_seconds","weighted_gpu_util_mean_pct","weighted_ram_mean_pct"]].sort_values(["platform","campaign","phase"])\ncampaign_view',
            table_output(campaign_view),
        ),
        image_cell("02_tempo_acumulado_campanhas.png"),
        markdown("### Hardware e pilha de software observados"),
        code(
            'hardware = (runs.groupby(["platform","gpu_name","gpu_backend","tensorflow_version","tensorflow_metal_version"], dropna=False).agg(runs=("run_uid","nunique"), horas=("training_hours","sum")).reset_index())\nhardware',
            table_output(hardware),
        ),
        markdown("## 4. Resultados completos dos treinamentos"),
        markdown(
            f"A tabela pré-executada contém todos os {len(runs)} runs com métricas finais exatas disponíveis. `loss` deve ser minimizado; acurácia, Macro Precision, Macro Recall e Macro F1 devem ser maximizados."
        ),
        code(
            'result_columns = ["platform","campaign","phase","dataset","variant","batch_size","activation","quantization_variant","accuracy","balanced_accuracy","macro_precision","macro_recall","macro_f1","loss","epochs_completed","training_hours","mean_epoch_seconds","gpu_name","tensorflow_version","evidence_level"]\nall_results = runs[result_columns].sort_values(["platform","campaign","phase","dataset","variant"])\nall_results',
            table_output(all_results),
        ),
        markdown("### Melhores batches observados no Windows por dataset"),
        image_cell("13_melhores_batches_por_dataset.png"),
        markdown("### Varredura completa de batch (Batch Sweep) no Windows"),
        code(
            '''batch_sweep_rows = []
for ds in ["MNIST", "KMNIST", "Fashion-MNIST", "EMNIST Balanced", "SVHN", "CIFAR-10", "FER2013", "CIFAR-100 coarse", "GTSRB"]:
    ds_subset = runs[(runs.platform == "Windows") & (runs.campaign == "controlled-augmentation05-batch-activation") & (runs.phase == "batch") & (runs.dataset == ds)]
    if ds_subset.empty:
        continue
    b32 = ds_subset[ds_subset.batch_size == 32]
    b64 = ds_subset[ds_subset.batch_size == 64]
    b128 = ds_subset[ds_subset.batch_size == 128]
    b256 = ds_subset[ds_subset.batch_size == 256]
    best_b = ds_subset.loc[ds_subset.macro_f1.idxmax()]
    batch_sweep_rows.append({
        "dataset": ds,
        "batch_32_f1": b32.iloc[0].macro_f1 if not b32.empty else np.nan,
        "batch_64_f1": b64.iloc[0].macro_f1 if not b64.empty else np.nan,
        "batch_128_f1": b128.iloc[0].macro_f1 if not b128.empty else np.nan,
        "batch_256_f1": b256.iloc[0].macro_f1 if not b256.empty else np.nan,
        "melhor_batch": int(best_b.batch_size),
        "melhor_macro_f1": best_b.macro_f1,
    })
batch_sweep_view = pd.DataFrame(batch_sweep_rows)
batch_sweep_view''',
            table_output(batch_sweep_view),
        ),
        image_cell("16_batch_sweep_macro_f1.png"),
        markdown("### Resultados de quantização (FP32 × FP16 × INT8 PTQ) — Windows × Mac"),
        markdown(
            f'''A campanha de quantização avaliou **todos os 9 datasets** tanto no Windows quanto no Mac M4 nas variantes FP32, FP16 e INT8 PTQ (LiteRT), totalizando **{len(quant_pairs)} pares alinhados**. A tabela pré-executada compara a qualidade preditiva e o throughput entre as plataformas.'''
        ),
        code(
            '''quant_cross_columns = ["dataset","variant","macro_f1_windows","macro_f1_mac","delta_macro_f1_pp_windows_minus_mac","accuracy_windows","accuracy_mac","loss_windows","loss_mac","mean_train_examples_per_second_windows","macro_f1_winner"]
quant_cross_view = quant_pairs[quant_cross_columns].sort_values(["dataset","variant"])
quant_cross_view''',
            table_output(quant_cross_view),
        ),
        image_cell("14_quantizacao_macro_f1.png"),
        markdown("### Eficiência de inferência e throughput LiteRT no Windows"),
        markdown(
            "No Windows, a conversão INT8 PTQ reduziu a pegada dos pesos para aproximadamente 311 KB por modelo e alcançou taxas de inferência de 1.510 a 6.946 ex/s com degradação mínima de Macro F1."
        ),
        code(
            '''quant_runs = runs[(runs.phase == "quantization") & (runs.campaign == "quantization_all")]
quant_rows = []
for ds in ["MNIST", "KMNIST", "Fashion-MNIST", "EMNIST Balanced", "SVHN", "CIFAR-10", "FER2013", "CIFAR-100 coarse", "GTSRB"]:
    ds_quant = quant_runs[quant_runs.dataset == ds]
    if ds_quant.empty:
        continue
    fp32_r = ds_quant[ds_quant.variant == "fp32"]
    fp16_r = ds_quant[ds_quant.variant == "fp16"]
    int8_r = ds_quant[ds_quant.variant == "int8_ptq"]
    f1_32 = fp32_r.iloc[0].macro_f1 if not fp32_r.empty else np.nan
    f1_16 = fp16_r.iloc[0].macro_f1 if not fp16_r.empty else np.nan
    f1_8 = int8_r.iloc[0].macro_f1 if not int8_r.empty else np.nan
    delta_8_32 = (f1_8 - f1_32) * 100 if pd.notna(f1_8) and pd.notna(f1_32) else np.nan
    tput_32 = fp32_r.iloc[0].mean_train_examples_per_second if not fp32_r.empty else np.nan
    tput_8 = int8_r.iloc[0].mean_train_examples_per_second if not int8_r.empty else np.nan
    speedup = tput_8 / tput_32 if pd.notna(tput_32) and pd.notna(tput_8) and tput_32 > 0 else np.nan
    quant_rows.append({
        "dataset": ds,
        "fp32_macro_f1": f1_32,
        "fp16_macro_f1": f1_16,
        "int8_ptq_macro_f1": f1_8,
        "delta_int8_minus_fp32_pp": delta_8_32,
        "throughput_fp32_ex_s": tput_32,
        "throughput_int8_ex_s": tput_8,
        "speedup_throughput": speedup,
    })
quant_view = pd.DataFrame(quant_rows)
quant_view''',
            table_output(quant_view),
        ),
        image_cell("15_quantizacao_throughput_latencia.png"),
        markdown(f"## 5. Comparação principal: {len(batch_pairs)} pares batch equivalentes"),
        markdown(
            f'''Estes {len(batch_pairs)} pares têm mesmo dataset, batch, fração adicional, seed (42) e fingerprint de divisão (quando verificável), cobrindo 5 datasets (MNIST, Fashion-MNIST, KMNIST, EMNIST Balanced e CIFAR-10) em batches 32, 64, 128 e 256. O Macro F1 médio difere apenas **{summary["comparisons"]["batch_exact"]["mean_delta_macro_f1_pp_windows_minus_mac"]:+.3f} p.p.** (Windows − Mac), com mediana **{summary["comparisons"]["batch_exact"]["median_delta_macro_f1_pp_windows_minus_mac"]:+.3f} p.p.**. O Windows obteve maior F1 em {summary["comparisons"]["batch_exact"]["windows_macro_f1_wins"]}/{len(batch_pairs)} pares e o Mac em {summary["comparisons"]["batch_exact"]["mac_macro_f1_wins"]}/{len(batch_pairs)}.

Em desempenho computacional, o Windows foi sistematicamente mais rápido em todos os {len(batch_pairs)} pares; a mediana de `tempo Windows / tempo Mac` foi **{summary["comparisons"]["batch_exact"]["median_training_time_ratio_windows_over_mac"]:.3f}**, isto é, o Windows consumiu cerca de 15,9% do tempo do Mac nesses pares (speedup mediano de 6,27× e máximo de 8,76× no Mac M4).'''
        ),
        code(
            '''exact_columns = ["dataset","batch_size_windows","macro_f1_windows","macro_f1_mac","delta_macro_f1_pp_windows_minus_mac","accuracy_windows","accuracy_mac","loss_windows","loss_mac","mean_epoch_seconds_windows","mean_epoch_seconds_mac","epoch_time_ratio_windows_over_mac","macro_f1_winner","training_time_winner","same_split_fingerprint"]
exact_view = batch_pairs[exact_columns].sort_values(["dataset","batch_size_windows"])
exact_view''',
            table_output(exact_view),
        ),
        image_cell("05_macro_f1_pares_batch.png"),
        image_cell("18_paridade_residuos_f1.png"),
        image_cell("06_tempo_epoca_pares_batch.png"),
        image_cell("17_speedup_windows_mac.png"),
        markdown(f"## 6. Ativações: comparação descritiva em {len(activation_pairs)} pares"),
        markdown(
            f'''Há {len(activation_pairs)} pares por dataset e ativação, cobrindo todos os 9 datasets com ReLU, Sigmoid e Softmax. Eles **não isolam o sistema operacional**: Windows usa batches específicos por dataset e Mac usa batch 256; a versão do TensorFlow também difere. A diferença média descritiva de Macro F1 foi **{summary["comparisons"]["activation_descriptive"]["mean_delta_macro_f1_pp_windows_minus_mac"]:+.3f} p.p.**.

As diferenças materiais concentram-se em ReLU e Sigmoid. Softmax como ativação oculta colapsou para desempenho próximo do acaso em ambos os ambientes (~10–20%) e é empate numérico dentro da precisão prática.'''
        ),
        code(
            '''activation_columns = ["dataset","activation","batch_size_windows","batch_size_mac","macro_f1_windows","macro_f1_mac","delta_macro_f1_pp_windows_minus_mac","accuracy_windows","accuracy_mac","loss_windows","loss_mac","mean_epoch_seconds_windows","mean_epoch_seconds_mac","comparability_grade"]
activation_view = activation_pairs[activation_columns].sort_values(["dataset","activation"])
activation_view''',
            table_output(activation_view),
        ),
        image_cell("03_macro_f1_ativacoes_windows_mac.png"),
        image_cell("04_delta_macro_f1_ativacoes_heatmap.png"),
        image_cell("12_softmax_macro_f1.png"),
        markdown("## 7. Curvas, loss e tempo por epoch"),
        markdown(
            "As séries brutas de epoch estão disponíveis para 118 runs Windows e 15 runs Mac. A tabela resume o melhor Macro F1 de validação, menor `val_loss`, mediana de duração e throughput por run."
        ),
        code(
            'epoch_summary = (epochs.groupby(["platform","campaign","dataset","variant","run_uid"], dropna=False).agg(epochs=("epoch","count"), mediana_epoch_s=("epoch_seconds","median"), melhor_val_macro_f1=("val_macro_f1","max"), menor_val_loss=("val_loss","min"), media_examples_s=("train_examples_per_second","mean")).reset_index())\nepoch_summary',
            table_output(epoch_summary),
        ),
        image_cell("08_curvas_validacao_pares_batch.png"),
        image_cell("09_tempo_por_epoca_pares_batch.png"),
        markdown("## 8. Telemetria de hardware"),
        markdown(
            "Os campos de GPU não são semanticamente idênticos entre CUDA/NVML e Apple Metal; por isso, a utilização é comparada como sinal operacional e não como benchmark absoluto entre APIs. Memória é exibida em GiB."
        ),
        code(
            '''telemetry_columns = ["dataset","batch_size_windows","gpu_util_pct_mean_windows","gpu_util_pct_mean_mac","gpu_memory_used_bytes_mean_windows","gpu_memory_used_bytes_mean_mac","process_cpu_pct_mean_windows","process_cpu_pct_mean_mac","ram_pct_mean_windows","ram_pct_mean_mac"]
telemetry_exact = batch_pairs[telemetry_columns].copy()
for column in [c for c in telemetry_exact if "bytes" in c]:
    telemetry_exact[column] = telemetry_exact[column] / (1024**3)
    telemetry_exact.rename(columns={column: column.replace("bytes", "gib")}, inplace=True)
telemetry_exact''',
            table_output(telemetry_exact),
        ),
        image_cell("07_telemetria_pares_batch.png"),
        markdown("## 9. Métricas por classe"),
        markdown(
            f'''Foram consolidadas **{len(classes):,} linhas de classe**. A tabela seguinte mostra as 40 maiores diferenças absolutas de F1 por classe nos pares batch equivalentes. O sinal positivo favorece Windows; negativo favorece Mac. A tabela completa permanece em `class_metrics.csv`.'''
        ),
        code(
            '''class_delta_parts = []
for _, pair in batch_pairs.iterrows():
    w = classes[classes.run_uid == pair.windows_run_uid][["class_index","class_name","support","precision","recall","f1"]].rename(columns={"class_name":"class_name_windows","support":"support_windows","precision":"precision_windows","recall":"recall_windows","f1":"f1_windows"})
    m = classes[classes.run_uid == pair.mac_run_uid][["class_index","class_name","support","precision","recall","f1"]].rename(columns={"class_name":"class_name_mac","support":"support_mac","precision":"precision_mac","recall":"recall_mac","f1":"f1_mac"})
    merged = w.merge(m, on="class_index", how="inner")
    merged.insert(0, "dataset", pair.dataset)
    merged.insert(1, "batch_size", int(pair.batch_size_windows))
    merged["class_name"] = merged.class_name_windows.fillna(merged.class_name_mac)
    merged["delta_f1_pp_windows_minus_mac"] = (merged.f1_windows - merged.f1_mac) * 100
    merged["abs_delta_f1_pp"] = merged.delta_f1_pp_windows_minus_mac.abs()
    class_delta_parts.append(merged)
class_deltas = pd.concat(class_delta_parts, ignore_index=True)
class_delta_view = class_deltas.sort_values("abs_delta_f1_pp", ascending=False)[["dataset","batch_size","class_index","class_name","support_windows","support_mac","f1_windows","f1_mac","delta_f1_pp_windows_minus_mac"]].head(40)
class_delta_view''',
            table_output(class_delta_view),
        ),
        image_cell("10_delta_f1_por_classe_batch.png"),
        markdown("## 10. Matrizes de confusão dos pares equivalentes"),
        markdown(
            "As matrizes abaixo usam a mesma escala dentro de cada par. Os totais foram validados contra o número de amostras de teste."
        ),
    ]

    confusion_names = [
        "11_01_matriz_confusao_fashion_mnist_batch_128.png",
        "11_02_matriz_confusao_fashion_mnist_batch_256.png",
        "11_03_matriz_confusao_kmnist_batch_032.png",
        "11_04_matriz_confusao_mnist_batch_032.png",
        "11_05_matriz_confusao_mnist_batch_064.png",
        "11_06_matriz_confusao_mnist_batch_256.png",
    ]
    for filename in confusion_names:
        cells.append(image_cell(filename))

    cells.extend(
        [
            markdown("## 11. Lacunas documentadas no Mac"),
            markdown(
                f'''O relatório histórico de batch do Mac registra **{summary["reported_mac_gaps"]["batch_completed_reported"]} células concluídas** nos 5 datasets executados, das quais **{summary["reported_mac_gaps"]["batch_completed_with_raw_versioned"]}** têm artefatos brutos versionados e **{summary["reported_mac_gaps"]["batch_completed_report_only"]}** permanecem preservadas via relatório oficial. Os outros 4 datasets não foram executados na varredura de batch no Mac.

Na quantização do Mac, o relatório técnico consolidado de 2026-09-14 registra **todas as {summary["reported_mac_gaps"]["quantization_completed_reported"]} execuções concluídas** (9 datasets × 3 variantes: FP32, FP16 e INT8 PTQ), todas com métricas finais reportadas e agora integradas diretamente no pipeline de comparação multiplataforma.'''
            ),
            code("mac_batch_reported", table_output(mac_batch_reported)),
            code("mac_quant_reported", table_output(mac_quant_reported)),
            markdown("## 12. Limitações e interpretação"),
            markdown(
                '''- Os 15 pares batch são a comparação mais forte disponível para os 5 datasets executados em ambos os ambientes, mas ainda diferem em hardware, sistema, backend e versão do TensorFlow.
- Os 27 pares de ativação cobrem todos os 9 datasets, mas têm batches diferentes; são tendências descritivas, não efeito causal de Windows versus macOS.
- Há uma única seed (`42`) por condição comparada; não há base para intervalos de confiança ou teste de significância entre seeds.
- Os totais de horas são soma de horas por run, não duração de calendário quando houve paralelismo.
- Softmax foi testada como ativação oculta e apresentou colapso esperado; não se refere à saída softmax padrão do classificador.'''
            ),
            markdown("## 13. Síntese"),
            markdown(
                '''1. **Qualidade preditiva nos pares batch:** praticamente equivalente em média (+0,15 p.p. para Windows); o resultado varia por dataset/batch (Windows vence 9, Mac vence 6).
2. **Tempo nos pares batch:** o Windows foi 5,1×–8,8× mais rápido por epoch nos 15 pares disponíveis.
3. **Quantização:** 27 pares multiplataforma cobrindo todos os 9 datasets nas variantes FP32, FP16 e INT8 PTQ. INT8 LiteRT alcançou tamanho compacto (~311 KB) e throughput de até 6.946 ex/s no Windows, com perda mínima de F1.
4. **Ativações:** 27 pares cobrindo todos os 9 datasets. ReLU/Sigmoid superam amplamente Softmax oculta; diferenças Windows–Mac nessa campanha são confundidas por batch e stack.
5. **Telemetria:** o Mac reporta utilização percentual maior, enquanto o Windows conclui epochs muito mais rápido; as APIs medem backends distintos.
6. **Evidência:** 218 execuções consolidadas no total (140 Windows + 78 Mac). Windows possui cobertura granular completa para quase todos os runs; no Mac, 78 runs possuem métricas finais exatas consolidadas.'''
            ),
            markdown("## 14. Arquivos de auditoria"),
            code("inventory", table_output(inventory)),
            markdown(
                "Os hashes SHA-256 de todos os CSVs e JSONs consolidados estão em `analysis/consolidacao_resultados/data/checksums.sha256`; a origem de cada run permanece em `source_path`, `source_branch` e `source_commit`."
            ),
        ]
    )

    execution_count = 1
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = execution_count
            for output in cell["outputs"]:
                if output.get("output_type") == "execute_result":
                    output["execution_count"] = execution_count
            execution_count += 1

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.11"},
            "title": "Consolidação de resultados Windows × Mac",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    parsed = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in parsed["cells"] if cell["cell_type"] == "code"]
    image_outputs = sum(
        1
        for cell in code_cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )
    assert parsed["nbformat"] == 4
    assert len(code_cells) >= 20
    assert image_outputs == len(list(FIGURES.glob("*.png")))
    assert all(cell["execution_count"] is not None for cell in code_cells)

    namespace: dict[str, Any] = {}
    executed_code_cells = 0
    for cell in code_cells:
        source = "".join(cell["source"])
        if "from IPython.display import Image, display" in source:
            continue
        exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{cell['execution_count']}", "exec"), namespace)
        executed_code_cells += 1
    print(
        json.dumps(
            {
                "notebook": str(NOTEBOOK_PATH),
                "cells": len(parsed["cells"]),
                "code_cells": len(code_cells),
                "executed_code_cells": executed_code_cells,
                "image_outputs": image_outputs,
                "size_bytes": NOTEBOOK_PATH.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
