"""Render the completed KMNIST quantization analysis into a PDF with ReportLab."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str, digits: int = 3, percent: bool = False) -> str:
    raw = row.get(key, "")
    if not raw:
        return "—"
    try:
        number = float(raw)
    except ValueError:
        return raw
    return f"{number * 100:.{digits}f}%" if percent else f"{number:.{digits}f}"


def build_table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    table = Table([headers, *rows], colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.3),
        ("LEADING", (0, 0), (-1, -1), 8.7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8C4D0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF3F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def page_number(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#506070"))
    canvas.drawRightString(27.5 * cm, 1.0 * cm, f"Página {document.page}")
    canvas.drawString(1.5 * cm, 1.0 * cm, "Benchmark de quantização KMNIST")
    canvas.restoreState()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    analysis = args.analysis_dir.resolve()
    output = (args.output or analysis / "kmnist_quantization_report.pdf").resolve()
    summary = read_csv(analysis / "run_summary.csv")
    deltas = read_csv(analysis / "metric_deltas_vs_fp32.csv")
    quality = read_json(analysis / "data_quality.json")
    overview = read_json(analysis / "analysis_summary.json")

    document = SimpleDocTemplate(
        str(output), pagesize=landscape(A4), leftMargin=1.4 * cm, rightMargin=1.4 * cm,
        topMargin=1.25 * cm, bottomMargin=1.6 * cm,
        title="Benchmark de quantização KMNIST — análise técnica",
        author="TCC Benchmark",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=colors.HexColor("#17365D"), spaceAfter=7)
    subtitle = ParagraphStyle("SubtitleCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=11, leading=14, textColor=colors.HexColor("#506070"), spaceAfter=8)
    heading = ParagraphStyle("HeadingCustom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=colors.HexColor("#17365D"), spaceBefore=8, spaceAfter=5)
    normal = ParagraphStyle("NormalCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.3, leading=12, spaceAfter=5)
    bullet = ParagraphStyle("BulletCustom", parent=normal, leftIndent=12, firstLineIndent=-8, bulletIndent=0)
    centered = ParagraphStyle("Centered", parent=normal, alignment=TA_CENTER, textColor=colors.HexColor("#506070"))
    story: list[Any] = []
    story.append(Paragraph("Benchmark de quantização KMNIST", title))
    story.append(Paragraph("Análise técnica consolidada: qualidade, convergência, hardware, tempo e implantação", subtitle))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>Resultado principal.</b> Os cinco tratamentos foram concluídos. O relatório preserva as ressalvas de validade: INT4-QAT é emulado, FP16 não produziu FlatBuffer LiteRT e a PTQ INT8 é um grafo híbrido com tensores float auxiliares.", normal))
    story.append(Paragraph("<b>Checagens de integridade.</b> " + ("Todas as checagens obrigatórias passaram." if quality.get("passed") else "Há checagens obrigatórias que falharam; consulte o JSON de qualidade."), normal))
    story.append(Paragraph("<b>Principais achados.</b>", heading))
    for item in overview.get("key_findings", []):
        story.append(Paragraph(item, bullet, bulletText="•"))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Qualidade de teste", heading))
    metric_rows = [[row["label"], value(row, "accuracy", percent=True), value(row, "macro_precision", percent=True), value(row, "macro_recall", percent=True), value(row, "macro_f1", percent=True), value(row, "macro_ovr_auc", percent=True), value(row, "logit_mse_vs_fp32", 6)] for row in summary]
    story.append(build_table(["Variante", "Acurácia", "Precisão macro", "Recall macro", "Macro-F1", "AUC macro", "MSE logits"], metric_rows, [4.0*cm, 2.25*cm, 2.7*cm, 2.55*cm, 2.2*cm, 2.35*cm, 2.35*cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Image(str(analysis / "01_test_metrics.png"), width=24.7*cm, height=7.75*cm))
    story.append(PageBreak())
    story.append(Paragraph("Treinamento e uso de hardware", heading))
    train_rows = [[row["label"], value(row, "epochs_completed", 0), value(row, "training_seconds", 1), value(row, "mean_epoch_seconds", 2), value(row, "train_examples_per_second", 1), value(row, "gpu_util_mean_percent", 1), value(row, "gpu_memory_peak_gib", 2), value(row, "process_rss_peak_gib", 2)] for row in summary if row["variant"] != "int8_ptq"]
    story.append(build_table(["Variante", "Épocas", "Tempo épocas (s)", "Média/época (s)", "Treino (ex/s)", "GPU média (%)", "VRAM pico (GiB)", "RSS pico (GiB)"], train_rows, [3.7*cm, 1.4*cm, 2.65*cm, 2.6*cm, 2.5*cm, 2.45*cm, 2.55*cm, 2.45*cm]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Image(str(analysis / "02_validation_convergence.png"), width=14.2*cm, height=7.1*cm))
    story.append(Image(str(analysis / "03_training_time_and_vram.png"), width=14.2*cm, height=7.1*cm))
    story.append(PageBreak())
    story.append(Paragraph("Implantação LiteRT/CPU e limites de validade", heading))
    deploy_rows = [[row["label"], value(row, "serialized_litert_mib"), value(row, "estimated_deployment_mib"), row.get("litert_status") or "—", row.get("litert_validity") or "—", value(row, "inference_latency_ms"), value(row, "inference_throughput_eps", 1), value(row, "inference_rss_peak_mib", 1)] for row in summary]
    story.append(build_table(["Variante", "LiteRT (MiB)", "Pesos estimados (MiB)", "Conversão", "Validade", "Latência/batch (ms)", "Throughput (ex/s)", "RSS infer. (MiB)"], deploy_rows, [3.5*cm, 2.25*cm, 2.8*cm, 2.15*cm, 2.8*cm, 2.8*cm, 2.7*cm, 2.6*cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Image(str(analysis / "04_litert_efficiency.png"), width=24.7*cm, height=8.9*cm))
    story.append(PageBreak())
    story.append(Paragraph("Deltas contra FP32 e interpretação", heading))
    delta_rows = [[row["label"], value(row, "accuracy_delta_pp"), value(row, "macro_f1_delta_pp"), value(row, "auc_delta_pp"), value(row, "logit_mse_vs_fp32", 6), value(row, "prediction_agreement_vs_fp32", percent=True)] for row in deltas]
    story.append(build_table(["Variante", "Δ acurácia (p.p.)", "Δ Macro-F1 (p.p.)", "Δ AUC (p.p.)", "MSE logits", "Acordo FP32"], delta_rows, [4.0*cm, 3.1*cm, 3.25*cm, 2.8*cm, 3.0*cm, 3.2*cm]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>Limites:</b> cada variante tem uma única execução com seed 42; os resultados são descritivos e não possuem intervalos de confiança. QAT treinada do zero não isola causalmente apenas o efeito da quantização. A inferência LiteRT é CPU; ela não deve ser comparada diretamente ao desempenho de treino na GPU.", normal))
    story.append(Paragraph("<b>Avisos de qualidade:</b>", heading))
    for warning in quality.get("warnings", []):
        story.append(Paragraph(warning, bullet, bulletText="•"))
    story.append(Paragraph("<b>Recomendação:</b> repetir com múltiplas seeds, consertar a rota FP16 LiteRT, tratar PTQ como híbrida até eliminar tensores float e medir INT4 apenas em um backend com suporte nativo.", normal))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("Artefatos auditáveis: relatório Markdown, notebook executado, `run_summary.csv`, `metric_deltas_vs_fp32.csv`, `training_epochs.csv` e `data_quality.json`.", centered))
    document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
