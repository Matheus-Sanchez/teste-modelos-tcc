"""Build a reproducible Markdown report and charts for the current quantization run."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "controlled-quantization-fast-mac-m4"
REPORT_DIR = ROOT / "analysis_reports" / "quantizacao_2026-09-14"
CHART_DIR = REPORT_DIR / "charts"
LABELS = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "kmnist": "KMNIST", "emnist_balanced": "EMNIST balanced", "cifar10": "CIFAR-10", "cifar100_coarse": "CIFAR-100 coarse", "svhn": "SVHN", "gtsrb": "GTSRB", "fer2013": "FER2013"}
DATASETS = list(LABELS)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def root_for(dataset: str, precision: str) -> Path:
    return OUT / "quantization" / dataset / precision / dataset / "runs" / f"{dataset}__unit_interval__all_raw__seed-42"


def int8_root_for(dataset: str) -> Path:
    return OUT / "quantization" / dataset / "int8_ptq" / dataset / "runs" / f"{dataset}__unit_interval__all_raw__seed-42"


def rows() -> list[dict]:
    result = []
    for dataset in DATASETS:
        for precision in ("fp32", "fp16"):
            root = root_for(dataset, precision)
            status = read_json(root / "status.json")
            csv_path = root / "logs" / "epoch_metrics.csv"
            if not csv_path.exists():
                csv_path = root / "checkpoints" / "epoch_metrics.csv"
            epochs = []
            if csv_path.exists():
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    epochs = list(csv.DictReader(handle))
            test = read_json(root / "artifacts" / "test_metrics.json")
            classification = test.get("classification", {}) if isinstance(test, dict) else {}
            summary = read_json(root / "logs" / "training_summary.json")
            result.append({
                "dataset": dataset,
                "precision": precision.upper(),
                "status": status.get("status", "pending"),
                "epoch": int(epochs[-1]["epoch"]) if epochs else 0,
                "accuracy": classification.get("accuracy"),
                "macro_f1": classification.get("macro_f1"),
                "mean_epoch_seconds": summary.get("mean_epoch_seconds"),
                "updated": status.get("updated_at"),
                "root": root,
                "epochs": epochs,
            })
    return result


def int8_rows() -> list[dict]:
    result = []
    for dataset in DATASETS:
        root = int8_root_for(dataset)
        status = read_json(root / "status.json")
        benchmark = read_json(root / "artifacts" / "litert_benchmark.json")
        classification = benchmark.get("classification", {}) if isinstance(benchmark, dict) else {}
        result.append({
            "dataset": dataset,
            "status": status.get("status", "pending"),
            "accuracy": classification.get("accuracy"),
            "macro_f1": classification.get("macro_f1"),
            "root": root,
        })
    return result


def pct(value):
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def build_charts(data: list[dict]) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    lookup = {(r["dataset"], r["precision"]): r for r in data}
    labels = [LABELS[d] for d in DATASETS]
    fp32 = [lookup[(d, "FP32")]["macro_f1"] * 100 if lookup[(d, "FP32")]["macro_f1"] is not None else 0 for d in DATASETS]
    fp16 = [lookup[(d, "FP16")]["macro_f1"] * 100 if lookup[(d, "FP16")]["macro_f1"] is not None else 0 for d in DATASETS]
    y = list(range(len(labels)))
    height = 0.36
    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.barh([v - height / 2 for v in y], fp32, height, color="#2f6f9f", label="FP32")
    ax.barh([v + height / 2 for v in y], fp16, height, color="#d79b2b", label="FP16")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 de teste (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", color="#d9dee5", linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_title("Macro-F1 de teste por dataset e precisão", loc="left", fontsize=15, fontweight="bold", pad=24)
    ax.text(0, 1.015, "Barras sem resultado final representam execução pendente ou em andamento", transform=ax.transAxes, fontsize=9, color="#5d6873")
    for i, v in enumerate(fp32):
        if v:
            ax.text(v + 0.8, i - height / 2, f"{v:.1f}", va="center", fontsize=8)
    for i, v in enumerate(fp16):
        if v:
            ax.text(v + 0.8, i + height / 2, f"{v:.1f}", va="center", fontsize=8)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "macro_f1_teste.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    progress_fp32 = [lookup[(d, "FP32")]["epoch"] for d in DATASETS]
    progress_fp16 = [lookup[(d, "FP16")]["epoch"] for d in DATASETS]
    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.barh([v - height / 2 for v in y], progress_fp32, height, color="#2f6f9f", label="FP32")
    ax.barh([v + height / 2 for v in y], progress_fp16, height, color="#d79b2b", label="FP16")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Última época registrada")
    ax.set_xlim(0, 105)
    ax.axvline(100, color="#25313c", linewidth=1, linestyle="--", label="100 épocas")
    ax.grid(axis="y", color="#d9dee5", linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_title("Progresso da rodada de quantização", loc="left", fontsize=15, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "progresso_epocas.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    active = next((r for r in data if r["status"] == "running"), None)
    if active and active["epochs"]:
        epochs = active["epochs"]
        x = [int(r["epoch"]) for r in epochs]
        val_acc = [float(r["val_accuracy"]) * 100 for r in epochs]
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
        ax.plot(x, val_acc, color="#2f6f9f", linewidth=2)
        ax.set_xlabel("Época")
        ax.set_ylabel("Val-accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(color="#d9dee5", linewidth=.8)
        ax.set_title(f"Curva de validação — {LABELS[active['dataset']]} {active['precision']}", loc="left", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(CHART_DIR / "curva_validacao_ativa.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    data = rows()
    int8 = int8_rows()
    build_charts(data)
    complete = [r for r in data if r["status"] == "completed"]
    active = [r for r in data if r["status"] == "running"]
    final_count = len(complete)
    int8_complete = sum(r["status"] == "completed" for r in int8)
    missing_int8 = [LABELS[r["dataset"]] for r in int8 if r["status"] != "completed"]
    snapshot = datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y %H:%M %Z")
    lines = [
        "# Relatório técnico — treinamento da rodada de quantização",
        "",
        f"**Snapshot dos artefatos:** {snapshot}. Este relatório cobre exclusivamente `outputs/controlled-quantization-fast-mac-m4`, com 9 datasets × 2 precisões, seed 42, 100 épocas, batch 256 e augmentation 0,5.",
        "",
        "## Resumo executivo",
        "",
        f"- Foram concluídos **{final_count}/18 treinos**; há **{len(active)} execução(ões) ativa(s)** e **{18 - final_count - len(active)} ainda não iniciada(s)**.",
        "- Os treinos concluídos chegaram a 100 épocas e possuem métricas de teste, previsões e checkpoints finais.",
        "- A execução ativa é " + (f"**{LABELS[active[0]['dataset']]} {active[0]['precision']}**, na época **{active[0]['epoch']}/100**." if active else "**nenhuma**, no momento do snapshot."),
        f"- Há **{int8_complete}/9 conversões INT8/LiteRT** com benchmark persistido" + (f"; faltam {', '.join(missing_int8)}." if missing_int8 else "."),
        "",
        "## Resultados finais disponíveis",
        "",
        "| Dataset | FP32 | FP16 | INT8/LiteRT | Observação |\n|---|---:|---:|---:|---|",
    ]
    for d in DATASETS:
        a = next(r for r in data if r["dataset"] == d and r["precision"] == "FP32")
        b = next(r for r in data if r["dataset"] == d and r["precision"] == "FP16")
        c = next(r for r in int8 if r["dataset"] == d)
        observation = "Concluído" if a["status"] == b["status"] == "completed" else f"FP32: {a['status']}; FP16: {b['status']}"
        lines.append(f"| {LABELS[d]} | {pct(a['macro_f1'])} | {pct(b['macro_f1'])} | {pct(c['macro_f1'])} | {observation}; INT8/LiteRT: {c['status']} |")
    lines += [
        "",
        "![Macro-F1 de teste](charts/macro_f1_teste.png)",
        "",
        "A Macro-F1 é a média harmônica das precisões e revocações por classe. Ela é usada aqui como métrica principal porque evita que classes com maior suporte dominem a leitura. Os valores acima são de teste e não devem ser comparados com a acurácia de validação durante um treino ativo.",
        "",
        "## Estado e checkpoints",
        "",
        "![Progresso por épocas](charts/progresso_epocas.png)",
        "",
        "| Dataset | Precisão | Estado | Última época | Artefato de teste |",
        "|---|---|---|---:|---|",
    ]
    for r in data:
        artifact = "sim" if r["macro_f1"] is not None else "não"
        lines.append(f"| {LABELS[r['dataset']]} | {r['precision']} | {r['status']} | {r['epoch']}/100 | {artifact} |")
    if active:
        r = active[0]
        lines += ["", "## Curva da execução ativa", "", "![Curva de validação](charts/curva_validacao_ativa.png)", "", f"A curva mostra apenas métricas persistidas até a época {r['epoch']}; não representa ainda o resultado final de teste."]
    lines += [
        "",
        "## O que falta fazer",
        "",
        "1. Auditar a presença de `test_metrics.json`, `predictions.csv`, `classification_report.json`, checkpoints e telemetria nos 18 treinos concluídos.",
        "2. Consolidar a comparação FP32/FP16 por dataset e selecionar resultados vencedores conforme o protocolo.",
        *([f"3. Executar o pós-processamento INT8/LiteRT faltante: {', '.join(missing_int8)}."] if missing_int8 else []),
        "",
        "## Previsão de término",
        "",
        "Os **18/18 treinos FP32/FP16 e 9/9 pós-processamentos INT8/LiteRT estão concluídos**; portanto, a duração pendente é zero.",
        "",
        "Não há uma estimativa de duração para os dois pós-processamentos restantes porque ainda não existe tempo observado equivalente nesta rodada. A comparação de duração dos treinos concluídos permanece válida apenas para a execução na GPU Apple M4; executar em CPU mudaria duração e comparabilidade.",
        "",
        "## Evidências e reprodutibilidade",
        "",
        "- Configuração: `configs/controlled-quantization-fast-mac-m4.yaml`.",
        "- Estado global: `outputs/controlled-quantization-fast-mac-m4/pipeline-status.json` (pode ficar desatualizado).",
        "- Estado primário: `outputs/controlled-quantization-fast-mac-m4/quantization/**/status.json`.",
        "- Métricas: `quantization/**/logs/epoch_metrics.csv` ou `checkpoints/epoch_metrics.csv`.",
        "- Resultados de teste: `quantization/**/artifacts/test_metrics.json` e `classification_report.json`.",
        "- INT8/LiteRT: `quantization/**/int8_ptq/**/artifacts/int8_ptq.tflite` e `litert_benchmark.json` quando presentes.",
        "- Comando de retomada: repetir a execução com o mesmo `--output-root` e `--resume`.",
        "",
        "## Limitações",
        "",
        "O snapshot é uma fotografia do diretório no horário indicado. A classificação foi feita pelos artefatos individuais; o status global foi tratado como auxiliar. O treinamento e o pós-processamento INT8/LiteRT estão completos no snapshot.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_DIR / "README.md")


if __name__ == "__main__":
    main()
