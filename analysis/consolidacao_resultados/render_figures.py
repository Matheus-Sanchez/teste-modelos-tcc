from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "analysis" / "consolidacao_resultados"
DATA = BASE / "data"
FIGURES = BASE / "figures"

WIDTH = 2000
BG = "#F7F8FA"
PANEL = "#FFFFFF"
TEXT = "#17212B"
MUTED = "#5F6B76"
GRID = "#DCE2E8"
WINDOWS = "#1F5A8A"
MAC = "#D97706"
POSITIVE = "#245B88"
NEGATIVE = "#C46A2B"
GOLD = "#C7A24B"

DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "kmnist": "KMNIST",
    "emnist_balanced": "EMNIST Balanced",
    "cifar10": "CIFAR-10",
    "cifar100_coarse": "CIFAR-100 coarse",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "fer2013": "FER2013",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    choices = [
        Path("C:/Windows/Fonts/seguisb.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in choices:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONTS = {
    "title": font(52, True),
    "subtitle": font(26),
    "section": font(30, True),
    "label": font(23),
    "label_bold": font(23, True),
    "small": font(19),
    "tiny": font(16),
    "value": font(22, True),
}


def fmt(value: float | int | None, decimals: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/d"
    text = f"{float(value):.{decimals}f}".replace(".", ",")
    return text + suffix


def pct(value: float | None, decimals: int = 1) -> str:
    return fmt((float(value) * 100) if value is not None and not pd.isna(value) else None, decimals, "%")


def wrap_lines(text: str, max_chars: int) -> list[str]:
    return textwrap.wrap(str(text), width=max_chars, break_long_words=False) or [""]


def new_canvas(title: str, subtitle: str, height: int = 1200) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    image = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((90, 64), title, fill=TEXT, font=FONTS["title"])
    y = 135
    for line in wrap_lines(subtitle, 125):
        draw.text((92, y), line, fill=MUTED, font=FONTS["subtitle"])
        y += 36
    return image, draw, max(y + 35, 215)


def footer(draw: ImageDraw.ImageDraw, height: int, source: str) -> None:
    draw.line((90, height - 58, WIDTH - 90, height - 58), fill=GRID, width=2)
    draw.text((92, height - 45), source, fill=MUTED, font=FONTS["tiny"])


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=22, fill=PANEL, outline="#E5E9ED", width=2)


def legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.ellipse((x, y, x + 20, y + 20), fill=WINDOWS)
    draw.text((x + 31, y - 4), "Windows", fill=TEXT, font=FONTS["small"])
    draw.rectangle((x + 150, y, x + 170, y + 20), fill=MAC)
    draw.text((x + 181, y - 4), "Mac", fill=TEXT, font=FONTS["small"])


def save(image: Image.Image, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    image.save(FIGURES / name, format="PNG", optimize=True, dpi=(180, 180))


def draw_coverage(runs: pd.DataFrame, evidence: pd.DataFrame, status: pd.DataFrame, reported_batch: pd.DataFrame, quant: pd.DataFrame) -> None:
    image, draw, top = new_canvas(
        "Cobertura dos resultados por ambiente",
        "Runs completos conhecidos, separados pela qualidade da evidência disponível no snapshot.",
        1180,
    )
    box = (90, top, WIDTH - 90, 1040)
    panel(draw, box)
    categories = [
        ("Windows", "Bruto completo", int((runs.platform.eq("Windows") & runs.evidence_level.eq("raw_complete")).sum()), WINDOWS),
        ("Mac", "Bruto completo", int((runs.platform.eq("Mac") & runs.evidence_level.eq("raw_complete")).sum()), MAC),
        ("Mac", "Compacto + consolidado", int(runs.evidence_level.eq("raw_compact_plus_consolidated").sum()), "#EAAE54"),
        ("Mac", "Consolidado apenas", int(runs.evidence_level.eq("consolidated_report_only").sum()), "#F2CE8B"),
        ("Mac", "Batch só em relatório", int((~reported_batch.raw_artifact_versioned.astype(bool)).sum()), "#B88B5C"),
        ("Mac", "Quantização só em relatório", int(quant.status.eq("completed").sum()), "#7F6549"),
    ]
    max_value = max(value for _, _, value, _ in categories)
    left, right = 485, WIDTH - 170
    y = top + 85
    row_h = 118
    for platform_name, label, value, color in categories:
        draw.text((130, y + 13), platform_name, fill=TEXT, font=FONTS["label_bold"])
        draw.text((235, y + 13), label, fill=MUTED, font=FONTS["label"])
        bar_w = int((right - left) * value / max_value)
        draw.rounded_rectangle((left, y, left + bar_w, y + 54), radius=10, fill=color)
        draw.text((left + bar_w + 18, y + 9), str(value), fill=TEXT, font=FONTS["value"])
        y += row_h
    pending_windows = int(status[(status.platform == "Windows") & (status.status != "completed")].shape[0])
    pending_mac = int(quant.status.ne("completed").sum())
    note = f"Fora das barras: {pending_windows} estados Windows ainda não concluídos; {pending_mac} jobs da quantização Mac parciais ou não iniciados."
    draw.text((130, 950), note, fill=MUTED, font=FONTS["small"])
    footer(draw, image.height, "Fonte: status_inventory.csv, evidence_coverage.csv e relatórios Mac de 14/09/2026.")
    save(image, "01_cobertura_resultados.png")


def draw_campaign_hours(campaigns: pd.DataFrame) -> None:
    campaigns = campaigns[campaigns.phase != "smoke"].copy()
    campaigns["label"] = campaigns.apply(lambda row: f"{row['platform']} | {row['campaign']} | {row['phase']}", axis=1)
    campaigns = campaigns.sort_values("training_hours")
    image, draw, top = new_canvas(
        "Tempo acumulado observado por campanha",
        "Soma dos tempos de treino persistidos. Campanhas Mac disponíveis apenas por relatório não entram no total.",
        1350,
    )
    panel(draw, (90, top, WIDTH - 90, 1210))
    left, right = 750, WIDTH - 170
    max_value = float(campaigns.training_hours.max())
    y = top + 65
    row_h = 102
    for _, row in campaigns.iterrows():
        color = WINDOWS if row.platform == "Windows" else MAC
        label = str(row.label)
        for i, line in enumerate(wrap_lines(label, 48)[:2]):
            draw.text((130, y + i * 28), line, fill=TEXT if i == 0 else MUTED, font=FONTS["small"])
        bar_w = int((right - left) * float(row.training_hours) / max_value)
        draw.rounded_rectangle((left, y + 5, left + bar_w, y + 48), radius=8, fill=color)
        draw.text((left + bar_w + 14, y + 8), fmt(row.training_hours, 1, " h"), fill=TEXT, font=FONTS["small"])
        draw.text((right - 100, y + 58), f"{int(row.completed_runs)} runs", fill=MUTED, font=FONTS["tiny"], anchor="ra")
        y += row_h
    footer(draw, image.height, "Fonte: campaign_summary.csv. Totais acumulados por run, não tempo de relógio com paralelismo.")
    save(image, "02_tempo_acumulado_campanhas.png")


def draw_activation_f1(pairs: pd.DataFrame) -> None:
    datasets = list(dict.fromkeys(pairs.dataset.tolist()))
    short_labels = {
        "CIFAR-10": "C10",
        "CIFAR-100 coarse": "C100",
        "EMNIST Balanced": "EMNIST",
        "Fashion-MNIST": "F-MN",
        "KMNIST": "KMNIST",
        "MNIST": "MNIST",
        "SVHN": "SVHN",
    }
    activations = ["relu", "sigmoid", "softmax"]
    image, draw, top = new_canvas(
        "Macro F1 nas ativações Windows e Mac",
        "Vinte e um pares descritivos. O batch difere entre os ambientes, portanto as diferenças não isolam o efeito do sistema operacional.",
        1440,
    )
    legend(draw, WIDTH - 430, top - 32)
    margin = 90
    gap = 28
    panel_w = (WIDTH - 2 * margin - 2 * gap) // 3
    bottom = 1300
    for panel_index, activation in enumerate(activations):
        x0 = margin + panel_index * (panel_w + gap)
        x1 = x0 + panel_w
        panel(draw, (x0, top, x1, bottom))
        draw.text((x0 + 28, top + 24), activation.capitalize(), fill=TEXT, font=FONTS["section"])
        plot_top, plot_bottom = top + 92, bottom - 95
        plot_left, plot_right = x0 + 185, x1 - 35
        for tick in np.linspace(0, 1, 6):
            y = int(plot_bottom - tick * (plot_bottom - plot_top))
            draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
            draw.text((plot_left - 18, y), fmt(tick * 100, 0, "%"), fill=MUTED, font=FONTS["tiny"], anchor="rm")
        subset = pairs[pairs.activation == activation].set_index("dataset")
        group_w = (plot_right - plot_left) / len(datasets)
        for idx, dataset in enumerate(datasets):
            if dataset not in subset.index:
                continue
            row = subset.loc[dataset]
            center = plot_left + (idx + 0.5) * group_w
            bar_w = max(12, int(group_w * 0.27))
            for offset, key, color in [(-bar_w - 3, "macro_f1_windows", WINDOWS), (3, "macro_f1_mac", MAC)]:
                value = float(row[key])
                y = int(plot_bottom - value * (plot_bottom - plot_top))
                draw.rounded_rectangle((int(center + offset), y, int(center + offset + bar_w), plot_bottom), radius=4, fill=color)
            draw.text((center, plot_bottom + 14), short_labels.get(dataset, dataset), fill=MUTED, font=FONTS["tiny"], anchor="ma")
    footer(draw, image.height, "Fonte: comparisons_activation_windows_mac.csv. C10=CIFAR-10; C100=CIFAR-100 coarse; F-MN=Fashion-MNIST.")
    save(image, "03_macro_f1_ativacoes_windows_mac.png")


def diverging_color(value: float, max_abs: float) -> tuple[int, int, int]:
    if max_abs <= 0:
        return (245, 245, 245)
    t = min(1.0, abs(value) / max_abs)
    start = np.array([248, 248, 246])
    end = np.array([36, 91, 136] if value >= 0 else [196, 106, 43])
    return tuple(int(v) for v in (start * (1 - t) + end * t))


def draw_activation_delta_heatmap(pairs: pd.DataFrame) -> None:
    datasets = list(dict.fromkeys(pairs.dataset.tolist()))
    activations = ["relu", "sigmoid", "softmax"]
    pivot = pairs.pivot(index="dataset", columns="activation", values="delta_macro_f1_pp_windows_minus_mac").reindex(index=datasets, columns=activations)
    image, draw, top = new_canvas(
        "Diferença de Macro F1 nas ativações",
        "Windows menos Mac em pontos percentuais. Valores positivos favorecem Windows; o batch diferente torna a leitura descritiva.",
        1160,
    )
    panel(draw, (90, top, WIDTH - 90, 1025))
    left, right = 580, WIDTH - 200
    cell_w = (right - left) // len(activations)
    cell_h = 100
    max_abs = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
    for col, activation in enumerate(activations):
        draw.text((left + col * cell_w + cell_w / 2, top + 40), activation.capitalize(), fill=TEXT, font=FONTS["label_bold"], anchor="ma")
    y0 = top + 95
    for row_idx, dataset in enumerate(datasets):
        y = y0 + row_idx * cell_h
        draw.text((left - 28, y + cell_h / 2), dataset, fill=TEXT, font=FONTS["label"], anchor="rm")
        for col, activation in enumerate(activations):
            value = float(pivot.loc[dataset, activation])
            x = left + col * cell_w
            fill = diverging_color(value, max_abs)
            draw.rectangle((x, y, x + cell_w - 6, y + cell_h - 6), fill=fill, outline=PANEL, width=2)
            ink = "#FFFFFF" if abs(value) / max_abs > 0.45 else TEXT
            draw.text((x + (cell_w - 6) / 2, y + (cell_h - 6) / 2), fmt(value, 2, " p.p."), fill=ink, font=FONTS["value"], anchor="mm")
    footer(draw, image.height, "Fonte: comparisons_activation_windows_mac.csv. Softmax coincide numericamente nos pares disponíveis.")
    save(image, "04_delta_macro_f1_ativacoes_heatmap.png")


def draw_batch_dumbbell(pairs: pd.DataFrame) -> None:
    pairs = pairs.sort_values("macro_f1_windows").reset_index(drop=True)
    image, draw, top = new_canvas(
        "Macro F1 nos seis pares de batch comparáveis",
        "Condições alinhadas em dataset, batch, augmentation, seed e split. Hardware, sistema e TensorFlow ainda diferem.",
        1120,
    )
    legend(draw, WIDTH - 430, top - 32)
    panel(draw, (90, top, WIDTH - 90, 990))
    left, right = 620, WIDTH - 170
    min_x, max_x = 0.88, 1.0
    for tick in np.linspace(min_x, max_x, 7):
        x = left + (tick - min_x) / (max_x - min_x) * (right - left)
        draw.line((x, top + 85, x, 900), fill=GRID, width=1)
        draw.text((x, 915), fmt(tick * 100, 0, "%"), fill=MUTED, font=FONTS["tiny"], anchor="ma")
    y = top + 130
    row_h = 115
    for _, row in pairs.iterrows():
        label = f"{row.dataset} | {row.variant}"
        draw.text((130, y), label, fill=TEXT, font=FONTS["label"], anchor="lm")
        win = float(row.macro_f1_windows)
        mac = float(row.macro_f1_mac)
        xw = left + (win - min_x) / (max_x - min_x) * (right - left)
        xm = left + (mac - min_x) / (max_x - min_x) * (right - left)
        draw.line((xm, y, xw, y), fill="#AAB4BE", width=5)
        draw.ellipse((xw - 12, y - 12, xw + 12, y + 12), fill=WINDOWS, outline=PANEL, width=2)
        draw.rectangle((xm - 11, y - 11, xm + 11, y + 11), fill=MAC, outline=PANEL, width=2)
        delta = float(row.delta_macro_f1_pp_windows_minus_mac)
        draw.text((right + 22, y), fmt(delta, 2, " p.p."), fill=POSITIVE if delta >= 0 else NEGATIVE, font=FONTS["small"], anchor="lm")
        y += row_h
    footer(draw, image.height, "Fonte: comparisons_batch_windows_mac.csv. Delta à direita = Windows menos Mac.")
    save(image, "05_macro_f1_pares_batch.png")


def draw_batch_time(pairs: pd.DataFrame) -> None:
    pairs = pairs.sort_values(["dataset", "batch_size_windows"]).reset_index(drop=True)
    image, draw, top = new_canvas(
        "Tempo médio por época nos pares de batch",
        "O Windows via WSL2 com RTX A2000 registrou épocas entre 5,1 e 6,7 vezes mais rápidas nesses seis pares.",
        1120,
    )
    legend(draw, WIDTH - 430, top - 32)
    panel(draw, (90, top, WIDTH - 90, 990))
    left, right = 590, WIDTH - 170
    max_value = float(max(pairs.mean_epoch_seconds_windows.max(), pairs.mean_epoch_seconds_mac.max())) * 1.06
    y = top + 100
    row_h = 118
    for _, row in pairs.iterrows():
        label = f"{row.dataset} | {row.variant}"
        draw.text((130, y + 25), label, fill=TEXT, font=FONTS["label"], anchor="lm")
        for j, (key, color) in enumerate([("mean_epoch_seconds_windows", WINDOWS), ("mean_epoch_seconds_mac", MAC)]):
            value = float(row[key])
            yy = y + j * 35
            bar_w = (right - left) * value / max_value
            draw.rounded_rectangle((left, yy, left + bar_w, yy + 24), radius=5, fill=color)
            draw.text((left + bar_w + 12, yy + 12), fmt(value, 1, " s"), fill=TEXT, font=FONTS["tiny"], anchor="lm")
        ratio = float(row.epoch_time_ratio_windows_over_mac)
        draw.text((right + 55, y + 25), f"{fmt(1 / ratio, 1)}x", fill=MUTED, font=FONTS["small"], anchor="lm")
        y += row_h
    footer(draw, image.height, "Fonte: comparisons_batch_windows_mac.csv. Multiplicador = Mac/Windows em segundos por época.")
    save(image, "06_tempo_epoca_pares_batch.png")


def draw_telemetry(pairs: pd.DataFrame) -> None:
    metrics = [
        ("gpu_util_pct_mean", "GPU média", "%", 1.0),
        ("ram_pct_mean", "RAM média", "%", 1.0),
        ("process_rss_bytes_mean", "RSS médio", "GiB", 1 / (1024**3)),
        ("gpu_memory_used_bytes_mean", "Memória GPU média", "GiB", 1 / (1024**3)),
    ]
    image, draw, top = new_canvas(
        "Telemetria de hardware nos pares de batch",
        "Médias das amostras por run. A memória GPU do Apple M4 é compartilhada; a NVIDIA usa VRAM dedicada.",
        1500,
    )
    legend(draw, WIDTH - 430, top - 32)
    margin, gap = 90, 30
    panel_w = (WIDTH - 2 * margin - gap) // 2
    panel_h = 540
    for idx, (prefix, title, unit, scale) in enumerate(metrics):
        row_idx, col_idx = divmod(idx, 2)
        x0 = margin + col_idx * (panel_w + gap)
        y0 = top + row_idx * (panel_h + gap)
        x1, y1 = x0 + panel_w, y0 + panel_h
        panel(draw, (x0, y0, x1, y1))
        draw.text((x0 + 28, y0 + 24), title, fill=TEXT, font=FONTS["section"])
        values = []
        for platform_name in ("windows", "mac"):
            values.extend((pd.to_numeric(pairs[f"{prefix}_{platform_name}"], errors="coerce") * scale).dropna().tolist())
        max_value = max(values) * 1.12 if values else 1
        plot_left, plot_right = x0 + 170, x1 - 45
        plot_top, plot_bottom = y0 + 95, y1 - 85
        group_w = (plot_right - plot_left) / len(pairs)
        for i, (_, run) in enumerate(pairs.iterrows()):
            center = plot_left + (i + 0.5) * group_w
            bw = max(10, int(group_w * 0.3))
            for offset, platform_name, color in [(-bw - 2, "windows", WINDOWS), (2, "mac", MAC)]:
                value = as_number(run.get(f"{prefix}_{platform_name}"))
                if value is None:
                    continue
                value *= scale
                y = plot_bottom - value / max_value * (plot_bottom - plot_top)
                draw.rectangle((center + offset, y, center + offset + bw, plot_bottom), fill=color)
            short = f"{str(run.dataset).split('-')[0]}\n{int(run.batch_size_windows)}"
            for line_idx, line in enumerate(short.split("\n")):
                draw.text((center, plot_bottom + 8 + line_idx * 18), line, fill=MUTED, font=FONTS["tiny"], anchor="ma")
        draw.text((x0 + 28, y1 - 45), f"Escala: 0 a {fmt(max_value, 1)} {unit}", fill=MUTED, font=FONTS["tiny"])
    footer(draw, image.height, "Fonte: telemetry/summary.json consolidado em comparisons_batch_windows_mac.csv.")
    save(image, "07_telemetria_pares_batch.png")


def as_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def draw_epoch_curves(pairs: pd.DataFrame, epochs: pd.DataFrame) -> None:
    image, draw, top = new_canvas(
        "Curvas de validação nos pares de batch",
        "Macro F1 de validação por época. A mesma escala de 80% a 100% é usada nos seis painéis.",
        1580,
    )
    legend(draw, WIDTH - 430, top - 32)
    margin, gap_x, gap_y = 90, 28, 30
    cols, rows = 2, 3
    panel_w = (WIDTH - 2 * margin - gap_x) // cols
    panel_h = 390
    for idx, (_, pair) in enumerate(pairs.sort_values(["dataset", "batch_size_windows"]).iterrows()):
        row_idx, col_idx = divmod(idx, cols)
        x0 = margin + col_idx * (panel_w + gap_x)
        y0 = top + row_idx * (panel_h + gap_y)
        x1, y1 = x0 + panel_w, y0 + panel_h
        panel(draw, (x0, y0, x1, y1))
        draw.text((x0 + 24, y0 + 18), f"{pair.dataset} | {pair.variant}", fill=TEXT, font=FONTS["label_bold"])
        plot_left, plot_right = x0 + 80, x1 - 35
        plot_top, plot_bottom = y0 + 75, y1 - 55
        y_min, y_max = 0.80, 1.0
        for tick in (0.8, 0.9, 1.0):
            y = plot_bottom - (tick - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
            draw.text((plot_left - 12, y), fmt(tick * 100, 0, "%"), fill=MUTED, font=FONTS["tiny"], anchor="rm")
        for uid, color in [(pair.windows_run_uid, WINDOWS), (pair.mac_run_uid, MAC)]:
            series = epochs[epochs.run_uid == uid].sort_values("epoch")
            series = series.dropna(subset=["val_macro_f1"])
            points = []
            for _, point in series.iterrows():
                x = plot_left + (float(point.epoch) - 1) / 99 * (plot_right - plot_left)
                value = min(y_max, max(y_min, float(point.val_macro_f1)))
                y = plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=color, width=4, joint="curve")
        draw.text((plot_left, plot_bottom + 12), "1", fill=MUTED, font=FONTS["tiny"], anchor="ma")
        draw.text((plot_right, plot_bottom + 12), "100 épocas", fill=MUTED, font=FONTS["tiny"], anchor="ma")
    footer(draw, image.height, "Fonte: epoch_metrics.csv. Valores abaixo de 80% são truncados visualmente para destacar a convergência final.")
    save(image, "08_curvas_validacao_pares_batch.png")


def draw_epoch_time_profiles(pairs: pd.DataFrame, epochs: pd.DataFrame) -> None:
    image, draw, top = new_canvas(
        "Tempo de cada época nos pares de batch",
        "Segundos por época ao longo de 100 épocas. Cada painel usa sua própria escala vertical, indicada no rodapé do painel.",
        1580,
    )
    legend(draw, WIDTH - 430, top - 32)
    margin, gap_x, gap_y = 90, 28, 30
    panel_w = (WIDTH - 2 * margin - gap_x) // 2
    panel_h = 390
    for idx, (_, pair) in enumerate(pairs.sort_values(["dataset", "batch_size_windows"]).iterrows()):
        row_idx, col_idx = divmod(idx, 2)
        x0 = margin + col_idx * (panel_w + gap_x)
        y0 = top + row_idx * (panel_h + gap_y)
        x1, y1 = x0 + panel_w, y0 + panel_h
        panel(draw, (x0, y0, x1, y1))
        draw.text((x0 + 24, y0 + 18), f"{pair.dataset} | {pair.variant}", fill=TEXT, font=FONTS["label_bold"])
        series_all = []
        for uid in (pair.windows_run_uid, pair.mac_run_uid):
            series_all.append(epochs[epochs.run_uid == uid].sort_values("epoch"))
        maxima = [pd.to_numeric(series.epoch_seconds, errors="coerce").quantile(0.98) for series in series_all if "epoch_seconds" in series]
        max_y = max([value for value in maxima if pd.notna(value)] + [1]) * 1.08
        plot_left, plot_right = x0 + 70, x1 - 35
        plot_top, plot_bottom = y0 + 75, y1 - 58
        for series, color in zip(series_all, (WINDOWS, MAC)):
            series = series.dropna(subset=["epoch_seconds"])
            points = []
            for _, point in series.iterrows():
                x = plot_left + (float(point.epoch) - 1) / 99 * (plot_right - plot_left)
                value = min(max_y, max(0, float(point.epoch_seconds)))
                y = plot_bottom - value / max_y * (plot_bottom - plot_top)
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
        draw.text((x0 + 24, y1 - 34), f"0 a {fmt(max_y, 0)} s", fill=MUTED, font=FONTS["tiny"])
        draw.text((plot_right, plot_bottom + 12), "100", fill=MUTED, font=FONTS["tiny"], anchor="ma")
    footer(draw, image.height, "Fonte: epoch_metrics.csv. O percentil 98 limita picos isolados de inicialização.")
    save(image, "09_tempo_por_epoca_pares_batch.png")


def draw_class_delta(pairs: pd.DataFrame, classes: pd.DataFrame) -> None:
    rows = []
    for _, pair in pairs.sort_values(["dataset", "batch_size_windows"]).iterrows():
        win = classes[classes.run_uid == pair.windows_run_uid].set_index("class_index")
        mac = classes[classes.run_uid == pair.mac_run_uid].set_index("class_index")
        joined = win[["f1"]].join(mac[["f1"]], lsuffix="_windows", rsuffix="_mac", how="inner")
        for class_index, item in joined.iterrows():
            rows.append({"pair": f"{pair.dataset} | {pair.variant}", "class_index": int(class_index), "delta": (item.f1_windows - item.f1_mac) * 100})
    data = pd.DataFrame(rows)
    pair_labels = list(dict.fromkeys(data.pair.tolist()))
    class_indices = sorted(data.class_index.unique())
    image, draw, top = new_canvas(
        "Diferença de F1 por classe nos pares de batch",
        "Windows menos Mac em pontos percentuais. As diferenças globais pequenas podem ocultar trocas entre classes.",
        1160,
    )
    panel(draw, (90, top, WIDTH - 90, 1025))
    left, right = 660, WIDTH - 130
    cell_w = (right - left) / len(class_indices)
    cell_h = 112
    max_abs = max(0.1, float(data.delta.abs().max()))
    for j, class_index in enumerate(class_indices):
        draw.text((left + (j + 0.5) * cell_w, top + 45), str(class_index), fill=TEXT, font=FONTS["small"], anchor="ma")
    y0 = top + 95
    for i, pair_label in enumerate(pair_labels):
        y = y0 + i * cell_h
        draw.text((left - 22, y + cell_h / 2), pair_label, fill=TEXT, font=FONTS["small"], anchor="rm")
        subset = data[data.pair == pair_label].set_index("class_index")
        for j, class_index in enumerate(class_indices):
            value = float(subset.loc[class_index, "delta"]) if class_index in subset.index else 0.0
            x = left + j * cell_w
            fill = diverging_color(value, max_abs)
            draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 5), fill=fill, outline=PANEL, width=1)
            ink = "#FFFFFF" if abs(value) / max_abs > 0.5 else TEXT
            draw.text((x + (cell_w - 4) / 2, y + (cell_h - 5) / 2), fmt(value, 1), fill=ink, font=FONTS["tiny"], anchor="mm")
    footer(draw, image.height, "Fonte: class_metrics.csv. Classes identificadas pelo índice 0 a 9; nomes completos constam no CSV.")
    save(image, "10_delta_f1_por_classe_batch.png")


def draw_confusion_pairs(pairs: pd.DataFrame, confusion: pd.DataFrame, runs: pd.DataFrame) -> None:
    run_lookup = runs.set_index("run_uid")
    for idx, (_, pair) in enumerate(pairs.sort_values(["dataset", "batch_size_windows"]).iterrows(), start=1):
        image, draw, top = new_canvas(
            f"Matrizes de confusão | {pair.dataset} | {pair.variant}",
            "Percentual por classe verdadeira. Cada linha soma 100%; os valores absolutos permanecem no arquivo consolidado.",
            1120,
        )
        margin, gap = 90, 40
        panel_w = (WIDTH - 2 * margin - gap) // 2
        for panel_idx, (platform_name, uid, color) in enumerate(
            [("Windows", pair.windows_run_uid, WINDOWS), ("Mac", pair.mac_run_uid, MAC)]
        ):
            x0 = margin + panel_idx * (panel_w + gap)
            x1 = x0 + panel_w
            y0, y1 = top, 980
            panel(draw, (x0, y0, x1, y1))
            metrics = run_lookup.loc[uid]
            draw.text((x0 + 30, y0 + 24), platform_name, fill=color, font=FONTS["section"])
            draw.text((x0 + 30, y0 + 63), f"Accuracy {pct(metrics.accuracy, 2)} | Macro F1 {pct(metrics.macro_f1, 2)}", fill=MUTED, font=FONTS["small"])
            subset = confusion[confusion.run_uid == uid]
            n = int(max(subset.true_index.max(), subset.pred_index.max()) + 1)
            matrix = np.zeros((n, n), dtype=float)
            for _, cell in subset.iterrows():
                matrix[int(cell.true_index), int(cell.pred_index)] = float(cell.row_share)
            grid_left, grid_top = x0 + 120, y0 + 130
            grid_size = min(panel_w - 170, y1 - grid_top - 80)
            cell = grid_size / n
            for r in range(n):
                for c in range(n):
                    value = matrix[r, c]
                    shade = int(248 - value * 170)
                    fill = (shade, min(248, shade + 20), min(255, shade + 45))
                    xx = grid_left + c * cell
                    yy = grid_top + r * cell
                    draw.rectangle((xx, yy, xx + cell, yy + cell), fill=fill, outline="#FFFFFF", width=1)
                    if n <= 10 and (value >= 0.01 or r == c):
                        ink = "#FFFFFF" if value > 0.55 else TEXT
                        draw.text((xx + cell / 2, yy + cell / 2), fmt(value * 100, 0), fill=ink, font=FONTS["tiny"], anchor="mm")
            for label_idx in range(n):
                draw.text((grid_left - 12, grid_top + (label_idx + 0.5) * cell), str(label_idx), fill=MUTED, font=FONTS["tiny"], anchor="rm")
                draw.text((grid_left + (label_idx + 0.5) * cell, grid_top + grid_size + 10), str(label_idx), fill=MUTED, font=FONTS["tiny"], anchor="ma")
            draw.text((grid_left + grid_size / 2, grid_top + grid_size + 40), "Classe predita", fill=MUTED, font=FONTS["small"], anchor="ma")
            draw.text((x0 + 35, grid_top + grid_size / 2), "Classe verdadeira", fill=MUTED, font=FONTS["small"], anchor="mm")
        footer(draw, image.height, "Fonte: confusion_matrices_long.csv. Valores completos por célula e contagens absolutas disponíveis no anexo de dados.")
        save(image, f"11_{idx:02d}_matriz_confusao_{str(pair.dataset_key)}_batch_{int(pair.batch_size_windows):03d}.png")


def draw_softmax(pairs: pd.DataFrame) -> None:
    data = pairs[pairs.activation == "softmax"].sort_values("dataset")
    image, draw, top = new_canvas(
        "Softmax interno produziu desempenho degenerado",
        "O Macro F1 ficou próximo ao acaso nos dois ambientes para todos os sete datasets comparáveis.",
        1080,
    )
    legend(draw, WIDTH - 430, top - 32)
    panel(draw, (90, top, WIDTH - 90, 940))
    left, right = 570, WIDTH - 180
    max_value = max(float(data.macro_f1_windows.max()), float(data.macro_f1_mac.max())) * 1.18
    y = top + 90
    row_h = 92
    for _, row in data.iterrows():
        draw.text((130, y + 18), row.dataset, fill=TEXT, font=FONTS["label"], anchor="lm")
        for j, (key, color) in enumerate([("macro_f1_windows", WINDOWS), ("macro_f1_mac", MAC)]):
            value = float(row[key])
            yy = y + j * 30
            bar_w = (right - left) * value / max_value if max_value else 0
            draw.rectangle((left, yy, left + bar_w, yy + 21), fill=color)
            draw.text((left + bar_w + 10, yy + 10), pct(value, 2), fill=TEXT, font=FONTS["tiny"], anchor="lm")
        y += row_h
    footer(draw, image.height, "Fonte: comparisons_activation_windows_mac.csv. Softmax é a ativação oculta testada, não a saída padrão da classificação.")
    save(image, "12_softmax_macro_f1.png")


def draw_best_batches(runs: pd.DataFrame, reported_batch: pd.DataFrame) -> None:
    windows = runs[(runs.platform == "Windows") & (runs.campaign == "controlled-augmentation05-batch-activation") & (runs.phase == "batch")]
    win_best = windows.loc[windows.groupby("dataset_key").macro_f1.idxmax()].copy()
    mac_best = reported_batch[reported_batch.is_reported_best.astype(bool)].copy()
    rows = win_best[["dataset_key", "dataset", "batch_size", "macro_f1"]].merge(
        mac_best[["dataset_key", "batch_size", "reported_macro_f1", "raw_artifact_versioned"]],
        on="dataset_key",
        how="outer",
        suffixes=("_windows", "_mac"),
    )
    image, draw, top = new_canvas(
        "Melhor batch conhecido por dataset",
        "Windows possui 9 datasets completos. O relatório Mac de 14/09 registra vencedores para 5 datasets; dois deles não têm artefatos brutos versionados.",
        1360,
    )
    panel(draw, (90, top, WIDTH - 90, 1215))
    headers = ["Dataset", "Windows batch", "Windows Macro F1", "Mac batch", "Mac Macro F1", "Evidência Mac"]
    xs = [130, 650, 900, 1210, 1440, 1680]
    for x, header in zip(xs, headers):
        draw.text((x, top + 45), header, fill=TEXT, font=FONTS["label_bold"])
    draw.line((120, top + 88, WIDTH - 120, top + 88), fill=GRID, width=2)
    y = top + 110
    row_h = 87
    for _, row in rows.sort_values("dataset").iterrows():
        draw.text((xs[0], y), str(row.get("dataset", DATASET_LABELS.get(row.dataset_key, row.dataset_key))), fill=TEXT, font=FONTS["small"])
        draw.text((xs[1], y), str(int(row.batch_size_windows)) if pd.notna(row.batch_size_windows) else "n/d", fill=TEXT, font=FONTS["small"])
        draw.text((xs[2], y), pct(row.macro_f1, 2) if pd.notna(row.macro_f1) else "n/d", fill=WINDOWS, font=FONTS["small"])
        draw.text((xs[3], y), str(int(row.batch_size_mac)) if pd.notna(row.batch_size_mac) else "n/d", fill=TEXT, font=FONTS["small"])
        draw.text((xs[4], y), pct(row.reported_macro_f1, 2) if pd.notna(row.reported_macro_f1) else "n/d", fill=MAC, font=FONTS["small"])
        evidence_text = "bruto" if row.get("raw_artifact_versioned") is True or row.get("raw_artifact_versioned") == 1 else "relatório" if pd.notna(row.get("raw_artifact_versioned")) else "ausente"
        draw.text((xs[5], y), evidence_text, fill=MUTED, font=FONTS["small"])
        draw.line((120, y + 48, WIDTH - 120, y + 48), fill="#EDF0F3", width=1)
        y += row_h
    footer(draw, image.height, "Fonte: run_results.csv e reported_mac_batch_cells.csv. Métricas Mac report-only foram publicadas com quatro casas decimais.")
    save(image, "13_melhores_batches_por_dataset.png")


def main() -> None:
    runs = pd.read_csv(DATA / "run_results.csv")
    campaigns = pd.read_csv(DATA / "campaign_summary.csv")
    activation_pairs = pd.read_csv(DATA / "comparisons_activation_windows_mac.csv")
    batch_pairs = pd.read_csv(DATA / "comparisons_batch_windows_mac.csv")
    evidence = pd.read_csv(DATA / "evidence_coverage.csv")
    status = pd.read_csv(DATA / "status_inventory.csv")
    reported_batch = pd.read_csv(DATA / "reported_mac_batch_cells.csv")
    quant = pd.read_csv(DATA / "reported_mac_quantization_status.csv")
    epochs = pd.read_csv(DATA / "epoch_metrics.csv")
    classes = pd.read_csv(DATA / "class_metrics.csv")
    confusion = pd.read_csv(DATA / "confusion_matrices_long.csv")

    draw_coverage(runs, evidence, status, reported_batch, quant)
    draw_campaign_hours(campaigns)
    draw_activation_f1(activation_pairs)
    draw_activation_delta_heatmap(activation_pairs)
    draw_batch_dumbbell(batch_pairs)
    draw_batch_time(batch_pairs)
    draw_telemetry(batch_pairs)
    draw_epoch_curves(batch_pairs, epochs)
    draw_epoch_time_profiles(batch_pairs, epochs)
    draw_class_delta(batch_pairs, classes)
    draw_confusion_pairs(batch_pairs, confusion, runs)
    draw_softmax(activation_pairs)
    draw_best_batches(runs, reported_batch)
    print(f"Generated {len(list(FIGURES.glob('*.png')))} figures in {FIGURES}")


if __name__ == "__main__":
    main()
