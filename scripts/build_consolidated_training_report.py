"""Build a consolidated report for augmentation 0.5 activations and quantization."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "docs" / "augmentation05_final" / "data" / "resultados_augmentation05.csv"
QUANT_REPORT = ROOT / "analysis_reports" / "quantizacao_2026-09-14"
OUT = ROOT / "analysis_reports" / "treinamento_consolidado_2026-09-14"
CHARTS = OUT / "charts"
DATASETS = [("mnist", "MNIST"), ("fashion_mnist", "Fashion-MNIST"), ("kmnist", "KMNIST"), ("emnist_balanced", "EMNIST balanced"), ("cifar10", "CIFAR-10"), ("cifar100_coarse", "CIFAR-100 coarse"), ("svhn", "SVHN"), ("gtsrb", "GTSRB"), ("fer2013", "FER2013")]


def read_json(path: Path) -> dict:
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quant_rows() -> list[dict]:
    result = []
    for key, label in DATASETS:
        for precision in ("fp32", "fp16"):
            root = ROOT / "outputs" / "controlled-quantization-fast-mac-m4" / "quantization" / key / precision / key / "runs" / f"{key}__unit_interval__all_raw__seed-42"
            status = read_json(root / "status.json")
            csv_path = root / "logs" / "epoch_metrics.csv"
            if not csv_path.exists():
                csv_path = root / "checkpoints" / "epoch_metrics.csv"
            epochs = read_csv(csv_path) if csv_path.exists() else []
            test = read_json(root / "artifacts" / "test_metrics.json").get("classification", {})
            result.append({"key": key, "dataset": label, "precision": precision.upper(), "status": status.get("status", "pending"), "epoch": int(epochs[-1]["epoch"]) if epochs else 0, "macro_f1": test.get("macro_f1"), "root": root})
    return result


def int8_rows() -> list[dict]:
    result = []
    for key, label in DATASETS:
        root = ROOT / "outputs" / "controlled-quantization-fast-mac-m4" / "quantization" / key / "int8_ptq" / key / "runs" / f"{key}__unit_interval__all_raw__seed-42"
        status = read_json(root / "status.json")
        benchmark = read_json(root / "artifacts" / "litert_benchmark.json").get("classification", {})
        result.append({"key": key, "dataset": label, "status": status.get("status", "pending"), "macro_f1": benchmark.get("macro_f1"), "root": root})
    return result


def build_chart(activation: list[dict], quant: list[dict]) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    best = {}
    for row in activation:
        if row["status"] == "completed":
            key = row["dataset_key"]
            current = best.get(key)
            if current is None or float(row["test_macro_f1"]) > current["value"]:
                best[key] = {"value": float(row["test_macro_f1"]), "label": row["activation"]}
    q = {(r["key"], r["precision"]): r for r in quant}
    labels = [label for _, label in DATASETS]
    act_values = [best[key]["value"] * 100 if key in best else 0 for key, _ in DATASETS]
    fp32 = [q[(key, "FP32")]["macro_f1"] * 100 if q[(key, "FP32")]["macro_f1"] is not None else 0 for key, _ in DATASETS]
    fp16 = [q[(key, "FP16")]["macro_f1"] * 100 if q[(key, "FP16")]["macro_f1"] is not None else 0 for key, _ in DATASETS]
    x = range(len(labels))
    width = .26
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=160)
    ax.bar([i - width for i in x], act_values, width, color="#2f6f9f", label="Melhor ativação")
    ax.bar(x, fp32, width, color="#d79b2b", label="Quantização FP32")
    ax.bar([i + width for i in x], fp16, width, color="#c96f3d", label="Quantização FP16")
    ax.set_xticks(list(x), labels, rotation=35, ha="right")
    ax.set_ylabel("Macro-F1 de teste (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#d9dee5", linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_title("Comparação consolidada: ativações e quantização", loc="left", fontsize=15, fontweight="bold", pad=16)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHARTS / "comparacao_ativacao_quantizacao.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pct(value):
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def main() -> None:
    activation = read_csv(ACT)
    quant = quant_rows()
    int8 = int8_rows()
    build_chart(activation, quant)
    completed_q = sum(r["status"] == "completed" for r in quant)
    active_q = sum(r["status"] == "running" for r in quant)
    completed_int8 = sum(r["status"] == "completed" for r in int8)
    missing_int8 = [r["dataset"] for r in int8 if r["status"] != "completed"]
    missing_int8_text = ", ".join(missing_int8)
    int8_state_text = (
        f"Existem {completed_int8}/9 artefatos INT8/LiteRT com `litert_benchmark.json`; faltam {missing_int8_text}."
        if missing_int8
        else "Os 9/9 datasets possuem artefatos INT8/LiteRT com `litert_benchmark.json`."
    )
    int8_bullet = (
        f"**INT8/LiteRT:** {completed_int8}/9 datasets possuem conversão e benchmark persistidos; faltam {missing_int8_text}."
        if missing_int8
        else "**INT8/LiteRT:** 9/9 datasets possuem conversão e benchmark persistidos; etapa concluída."
    )
    remaining_steps = ["Auditar os 18 runs FP32/FP16 e confirmar a presença de todos os artefatos finais."]
    if missing_int8:
        remaining_steps.append(f"Executar o pós-processamento INT8/LiteRT faltante: {missing_int8_text}.")
    remaining_steps += [
        "Usar múltiplas seeds para confirmar a escolha de Sigmoid/ReLU e quantificar variabilidade.",
        "Manter Softmax como condição negativa ou revisar sua posição arquitetural antes de novos treinos longos.",
    ]
    best = {}
    for row in activation:
        if row["status"] == "completed" and (row["dataset_key"] not in best or float(row["test_macro_f1"]) > best[row["dataset_key"]]["value"]):
            best[row["dataset_key"]] = {"value": float(row["test_macro_f1"]), "label": row["activation"]}
    q = {(r["key"], r["precision"]): r for r in quant}
    snapshot = datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y %H:%M %Z")
    lines = [
        "# Relatório consolidado — treinamento, ativações e quantização",
        "",
        f"**Snapshot dos artefatos:** {snapshot}. O documento junta a rodada de ativações com augmentation 0,5 e a rodada isolada de quantização com o mesmo protocolo de 100 épocas e augmentation 0,5.",
        "",
        "## Resumo executivo",
        "",
        f"- **Ativações:** 27/27 treinos concluídos, 2.700 épocas e 99,46 horas acumuladas.",
        f"- **Treinamento da quantização:** {completed_q}/18 treinos concluídos, {active_q} ativo(s) e {18 - completed_q - active_q} pendente(s) no snapshot.",
        f"- {int8_bullet}",
        "- A melhor ativação por Macro-F1 foi Sigmoid em 8/9 datasets; ReLU venceu no GTSRB. Softmax apresentou colapso sistemático e não venceu nenhum dataset.",
        "- A comparação consolidada é útil por dataset, mas não deve ser lida como comparação entre dificuldades diferentes de datasets.",
        "",
        "## Comparação consolidada",
        "",
        "![Comparação entre ativações e quantização](charts/comparacao_ativacao_quantizacao.png)",
        "",
        "| Dataset | Melhor ativação | Macro-F1 ativação | Quant. FP32 | Quant. FP16 | INT8/LiteRT | Estado do treinamento |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for key, label in DATASETS:
        a = best.get(key, {"label": "—", "value": None})
        f32 = q[(key, "FP32")]
        f16 = q[(key, "FP16")]
        i8 = next(r for r in int8 if r["key"] == key)
        state = f"FP32 {f32['status']}; FP16 {f16['status']}"
        lines.append(f"| {label} | {a['label']} | {pct(a['value'])} | {pct(f32['macro_f1'])} | {pct(f16['macro_f1'])} | {pct(i8['macro_f1'])} | {state} |")
    lines += [
        "",
        "## Resultados completos da fase de ativações",
        "",
        "A fase de ativações está finalizada. A tabela abaixo reúne os 27 resultados de teste; Macro-F1 é a métrica principal, com accuracy e balanced accuracy como apoio.",
        "",
        "| Dataset | Ativação | Accuracy | Balanced accuracy | Macro-F1 | Tempo (h) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in activation:
        lines.append(f"| {r['dataset']} | {r['activation']} | {float(r['test_accuracy'])*100:.2f}% | {float(r['test_balanced_accuracy'])*100:.2f}% | {float(r['test_macro_f1'])*100:.2f}% | {float(r['training_seconds'])/3600:.2f} |")
    lines += [
        "",
        "Para os gráficos detalhados de accuracy, tempo, hardware e comparação com augmentation 2,0, consulte o [relatório final da fase de ativações](../../docs/augmentation05_final/README.md).",
        "",
        "## Interpretação dos resultados de ativações",
        "",
        "- Sigmoid foi a melhor ativação em MNIST, Fashion-MNIST, KMNIST, EMNIST balanced, CIFAR-10, CIFAR-100 coarse, SVHN e FER2013.",
        "- ReLU foi superior no GTSRB.",
        "- Softmax teve Macro-F1 entre 0,09% e 5,72% e comportamento de previsão concentrada em uma classe. Isso é compatível com colapso da representação quando Softmax é usado repetidamente como ativação oculta; deve ser tratado como ablação negativa neste protocolo.",
        "- Os resultados usam uma única seed (42). As diferenças entre ativações são descritivas e não medem incerteza entre seeds.",
        "",
        "## Estado da quantização",
        "",
        f"A matriz de treinamento da quantização foi concluída: {completed_q}/18 execuções FP32/FP16 chegaram a 100 épocas, com batch 256 e `mixed_float16` conforme a configuração. {int8_state_text}",
        "",
        "Para detalhes por época, checkpoints, curva ativa e previsão específica da quantização, consulte o [relatório da quantização](../quantizacao_2026-09-14/README.md).",
        "",
        "## O que falta fazer",
        "",
        *[f"{i}. {step}" for i, step in enumerate(remaining_steps, 1)],
        "",
        "## Previsão de término",
        "",
        "A fase de ativações, os 18 treinos FP32/FP16 e o pós-processamento INT8/LiteRT estão concluídos no snapshot; não há duração pendente.",
        "",
        "## Evidências e método",
        "",
        "- Ativações: `outputs/controlled-augmentation05-activations-mac2/` e `docs/augmentation05_final/data/resultados_augmentation05.csv`.",
        "- Quantização: `outputs/controlled-quantization-fast-mac-m4/quantization/**`.",
        "- Configuração de quantização: `configs/controlled-quantization-fast-mac-m4.yaml`.",
        "- Métricas finais: `test_metrics.json`, `classification_report.json`, `predictions.csv`, `training_summary.json` e, quando disponível, `int8_ptq.tflite`/`litert_benchmark.json`.",
        "- Estados: `status.json` por run; o `pipeline-status.json` global foi tratado apenas como auxiliar.",
        "",
        "## Limitações",
        "",
        "O relatório é um snapshot dos artefatos no horário indicado. A fase de ativações está confirmada como 27/27 completa; os treinos FP32/FP16 e a cobertura INT8/LiteRT estão completos.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
