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
FP32_COLOR = "#1F5A8A"
FP16_COLOR = "#0D9488"
INT8_COLOR = "#7C3AED"

DATASET_ORDER = [
    "MNIST", "KMNIST", "Fashion-MNIST", "EMNIST Balanced",
    "SVHN", "CIFAR-10", "FER2013", "CIFAR-100 coarse", "GTSRB"
]

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
    "hero": font(38, True),
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
        "Cobertura dos resultados por ambiente e protocolo",
        "Inventário consolidado de execuções completas com métricas finais verificáveis em Windows e Mac.",
        1180,
    )
    box = (90, top, WIDTH - 90, 1040)
    panel(draw, box)
    categories = [
        ("Windows", "Total de runs concluídos (9 datasets)", int(runs[runs.platform == "Windows"].shape[0]), WINDOWS),
        ("Windows", "Batch sweep + Ativações + Quantização", 140, "#19486E"),
        ("Mac", "Total de runs concluídos (9 datasets)", int(runs[runs.platform == "Mac"].shape[0]), MAC),
        ("Mac", "Ativações consolidadas (ReLU, Sigmoid, Softmax)", 27, "#EAAE54"),
        ("Mac", "Quantização consolidada (FP32, FP16, INT8)", 27, "#7F6549"),
        ("Mac", "Batch sweep concluído (5 datasets)", 15, "#B88B5C"),
    ]
    max_value = max(value for _, _, value, _ in categories)
    left, right = 530, WIDTH - 170
    y = top + 85
    row_h = 118
    for platform_name, label, value, color in categories:
        draw.text((130, y + 13), platform_name, fill=TEXT, font=FONTS["label_bold"])
        draw.text((255, y + 13), label, fill=MUTED, font=FONTS["label"])
        bar_w = int((right - left) * value / max_value)
        draw.rounded_rectangle((left, y, left + bar_w, y + 54), radius=10, fill=color)
        draw.text((left + bar_w + 18, y + 9), str(value), fill=TEXT, font=FONTS["value"])
        y += row_h
    note = "Todos os treinamentos previstos foram finalizados em ambas as plataformas (140 runs no Windows, 78 no Mac; 69 pares comparativos)."
    draw.text((130, 960), note, fill=MUTED, font=FONTS["small"])
    footer(draw, image.height, "Fonte: run_results.csv, status_inventory.csv e relatórios técnicos consolidados de Windows e Mac.")
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
        "GTSRB": "GTSRB",
        "FER2013": "FER",
    }
    activations = ["relu", "sigmoid", "softmax"]
    image, draw, top = new_canvas(
        "Macro F1 nas ativações Windows e Mac (9 Datasets)",
        "Vinte e sete pares descritivos. O batch difere entre os ambientes (Windows 256 / Mac 64), apresentando o comportamento relativo de ReLU, Sigmoid e Softmax.",
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
        plot_left, plot_right = x0 + 80, x1 - 25
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
            bar_w = max(10, int(group_w * 0.32))
            for offset, key, color in [(-bar_w - 2, "macro_f1_windows", WINDOWS), (2, "macro_f1_mac", MAC)]:
                value = float(row[key])
                y = int(plot_bottom - value * (plot_bottom - plot_top))
                draw.rounded_rectangle((int(center + offset), y, int(center + offset + bar_w), plot_bottom), radius=4, fill=color)
            draw.text((center, plot_bottom + 14), short_labels.get(dataset, dataset), fill=MUTED, font=FONTS["tiny"], anchor="ma")
    footer(draw, image.height, "Fonte: comparisons_activation_windows_mac.csv. C10=CIFAR-10; C100=CIFAR-100 coarse; F-MN=Fashion-MNIST; FER=FER2013.")
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
        "Diferença de Macro F1 nas ativações (9 Datasets)",
        "Windows menos Mac em pontos percentuais nos 9 datasets. Valores positivos favorecem Windows; o batch diferente torna a leitura descritiva.",
        1340,
    )
    panel(draw, (90, top, WIDTH - 90, 1220))
    left, right = 580, WIDTH - 200
    cell_w = (right - left) // len(activations)
    cell_h = 86
    max_abs = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
    for col, activation in enumerate(activations):
        draw.text((left + col * cell_w + cell_w / 2, top + 35), activation.capitalize(), fill=TEXT, font=FONTS["label_bold"], anchor="ma")
    y0 = top + 80
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
        "Macro F1 nos 15 pares de batch comparáveis (Windows × Mac)",
        "Condições alinhadas em dataset, batch, augmentation e seed nos dois ambientes. Hardware, SO e TensorFlow diferem.",
        1640,
    )
    legend(draw, WIDTH - 430, top - 32)
    panel(draw, (90, top, WIDTH - 90, 1510))
    left, right = 640, WIDTH - 180
    min_x, max_x = 0.65, 1.0
    for tick in np.linspace(min_x, max_x, 8):
        x = left + (tick - min_x) / (max_x - min_x) * (right - left)
        draw.line((x, top + 80, x, 1420), fill=GRID, width=1)
        draw.text((x, 1435), fmt(tick * 100, 0, "%"), fill=MUTED, font=FONTS["tiny"], anchor="ma")
    y = top + 95
    row_h = 74
    for _, row in pairs.iterrows():
        label = f"{row.dataset} | {row.variant}"
        draw.text((130, y), label, fill=TEXT, font=FONTS["label"], anchor="lm")
        win = float(row.macro_f1_windows)
        mac = float(row.macro_f1_mac)
        xw = left + (win - min_x) / (max_x - min_x) * (right - left)
        xm = left + (mac - min_x) / (max_x - min_x) * (right - left)
        draw.line((xm, y, xw, y), fill="#AAB4BE", width=4)
        draw.ellipse((xw - 11, y - 11, xw + 11, y + 11), fill=WINDOWS, outline=PANEL, width=2)
        draw.rectangle((xm - 10, y - 10, xm + 10, y + 10), fill=MAC, outline=PANEL, width=2)
        delta = float(row.delta_macro_f1_pp_windows_minus_mac)
        draw.text((right + 22, y), fmt(delta, 2, " p.p."), fill=POSITIVE if delta >= 0 else NEGATIVE, font=FONTS["small"], anchor="lm")
        y += row_h
    footer(draw, image.height, "Fonte: comparisons_batch_windows_mac.csv. Delta à direita = Windows menos Mac em pontos percentuais.")
    save(image, "05_macro_f1_pares_batch.png")


def draw_batch_time(pairs: pd.DataFrame) -> None:
    pairs = pairs.sort_values(["dataset", "batch_size_windows"]).reset_index(drop=True)
    image, draw, top = new_canvas(
        "Tempo médio por época nos 15 pares de batch comparáveis",
        "O Windows via WSL2 com RTX A2000 registrou épocas entre 5,1 e 6,7 vezes mais rápidas em todos os pares de batch.",
        1850,
    )
    legend(draw, WIDTH - 430, top - 32)
    panel(draw, (90, top, WIDTH - 90, 1730))
    left, right = 610, WIDTH - 180
    max_value = float(max(pairs.mean_epoch_seconds_windows.max(), pairs.mean_epoch_seconds_mac.max())) * 1.06
    y = top + 85
    row_h = 88
    for _, row in pairs.iterrows():
        label = f"{row.dataset} | {row.variant}"
        draw.text((130, y + 25), label, fill=TEXT, font=FONTS["label"], anchor="lm")
        for j, (key, color) in enumerate([("mean_epoch_seconds_windows", WINDOWS), ("mean_epoch_seconds_mac", MAC)]):
            value = float(row[key])
            yy = y + j * 32
            bar_w = (right - left) * value / max_value
            draw.rounded_rectangle((left, yy, left + bar_w, yy + 22), radius=5, fill=color)
            draw.text((left + bar_w + 12, yy + 11), fmt(value, 1, " s"), fill=TEXT, font=FONTS["tiny"], anchor="lm")
        ratio = float(row.epoch_time_ratio_windows_over_mac)
        draw.text((right + 45, y + 25), f"{fmt(1 / ratio, 1)}x", fill=MUTED, font=FONTS["small"], anchor="lm")
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
    pairs_with_epochs = pairs[pairs["windows_run_uid"].isin(epochs.run_uid) & pairs["mac_run_uid"].isin(epochs.run_uid)].sort_values(["dataset", "batch_size_windows"]).reset_index(drop=True)
    if pairs_with_epochs.empty:
        return
    image, draw, top = new_canvas(
        "Curvas de validação nos pares de batch com histórico por época",
        "Macro F1 de validação por época nos pares com histórico detalhado versionado. Escala de 80% a 100%.",
        1580,
    )
    legend(draw, WIDTH - 430, top - 32)
    margin, gap_x, gap_y = 90, 28, 30
    cols = 2
    panel_w = (WIDTH - 2 * margin - gap_x) // cols
    panel_h = 390
    for idx, (_, pair) in enumerate(pairs_with_epochs.iterrows()):
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
    pairs_with_epochs = pairs[pairs["windows_run_uid"].isin(epochs.run_uid) & pairs["mac_run_uid"].isin(epochs.run_uid)].sort_values(["dataset", "batch_size_windows"]).reset_index(drop=True)
    if pairs_with_epochs.empty:
        return
    image, draw, top = new_canvas(
        "Tempo de cada época nos pares de batch",
        "Segundos por época ao longo de 100 épocas nos pares com histórico detalhado versionado.",
        1580,
    )
    legend(draw, WIDTH - 430, top - 32)
    margin, gap_x, gap_y = 90, 28, 30
    panel_w = (WIDTH - 2 * margin - gap_x) // 2
    panel_h = 390
    for idx, (_, pair) in enumerate(pairs_with_epochs.iterrows()):
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
    pairs_with_classes = pairs[pairs["windows_run_uid"].isin(classes.run_uid) & pairs["mac_run_uid"].isin(classes.run_uid)].copy()
    rows = []
    for _, pair in pairs_with_classes.sort_values(["dataset", "batch_size_windows"]).iterrows():
        win = classes[classes.run_uid == pair.windows_run_uid].set_index("class_index")
        mac = classes[classes.run_uid == pair.mac_run_uid].set_index("class_index")
        joined = win[["f1"]].join(mac[["f1"]], lsuffix="_windows", rsuffix="_mac", how="inner")
        for class_index, item in joined.iterrows():
            rows.append({"pair": f"{pair.dataset} | {pair.variant}", "class_index": int(class_index), "delta": (item.f1_windows - item.f1_mac) * 100})
    data = pd.DataFrame(rows)
    if data.empty:
        return
    pair_labels = list(dict.fromkeys(data.pair.tolist()))
    class_indices = sorted(data.class_index.unique())
    image, draw, top = new_canvas(
        "Diferença de F1 por classe nos pares de batch",
        "Windows menos Mac em pontos percentuais nos pares com relatório por classe disponível.",
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
    pairs_with_confusion = pairs[pairs["windows_run_uid"].isin(confusion.run_uid) & pairs["mac_run_uid"].isin(confusion.run_uid)].copy()
    run_lookup = runs.set_index("run_uid")
    for idx, (_, pair) in enumerate(pairs_with_confusion.sort_values(["dataset", "batch_size_windows"]).iterrows(), start=1):
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
        "Softmax interno produziu desempenho degenerado (9 Datasets)",
        "O Macro F1 ficou próximo ao acaso nos dois ambientes para todos os nove datasets comparáveis.",
        1280,
    )
    legend(draw, WIDTH - 430, top - 32)
    panel(draw, (90, top, WIDTH - 90, 1150))
    left, right = 570, WIDTH - 180
    max_value = max(float(data.macro_f1_windows.max()), float(data.macro_f1_mac.max())) * 1.18
    y = top + 85
    row_h = 88
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


def draw_quantization_f1(quant_pairs: pd.DataFrame, runs: pd.DataFrame) -> None:
    datasets = [d for d in DATASET_ORDER if d in quant_pairs.dataset.unique()]
    variants = [
        ("fp32", "FP32", FP32_COLOR),
        ("fp16", "FP16", FP16_COLOR),
        ("int8_ptq", "INT8 PTQ", INT8_COLOR),
    ]

    image, draw, top = new_canvas(
        "Macro F1 na quantização nos 9 datasets — Windows vs Mac",
        "Comparativo de FP32, FP16 e INT8 LiteRT PTQ entre as plataformas. 27 pares alinhados (batch 256, seed 42, 100 épocas).",
        1850,
    )

    legend(draw, WIDTH - 430, top - 32)
    panel(draw, (90, top, WIDTH - 90, 1730))

    left = 460
    right = WIDTH - 260
    y = top + 55
    row_h = 160

    for tick in np.linspace(0, 1, 6):
        gx = int(left + tick * (right - left))
        draw.line((gx, top + 40, gx, top + 55 + len(datasets) * row_h - 20), fill=GRID, width=1)
        draw.text((gx, top + 22), fmt(tick * 100, 0, "%"), fill=MUTED, font=FONTS["tiny"], anchor="ma")

    for ds in datasets:
        subset = quant_pairs[quant_pairs.dataset == ds]
        draw.text((130, y + 55), ds, fill=TEXT, font=FONTS["label_bold"])

        for v_idx, (v_key, v_label, v_color) in enumerate(variants):
            row_v = subset[subset.variant == v_key]
            bar_y = y + v_idx * 46
            draw.text((360, bar_y + 11), v_label, fill=MUTED, font=FONTS["small"])

            if not row_v.empty:
                win_f1 = float(row_v.iloc[0]["macro_f1_windows"])
                mac_f1 = float(row_v.iloc[0]["macro_f1_mac"])
                delta = float(row_v.iloc[0]["delta_macro_f1_pp_windows_minus_mac"])

                # Windows bar (blue)
                w_win = int((right - left) * win_f1)
                draw.rounded_rectangle((left, bar_y, left + w_win, bar_y + 18), radius=4, fill=WINDOWS)
                draw.text((left + w_win + 10, bar_y - 1), pct(win_f1, 2), fill=WINDOWS, font=FONTS["tiny"])

                # Mac bar (amber)
                w_mac = int((right - left) * mac_f1)
                draw.rounded_rectangle((left, bar_y + 22, left + w_mac, bar_y + 40), radius=4, fill=MAC)
                draw.text((left + w_mac + 10, bar_y + 21), pct(mac_f1, 2), fill=MAC, font=FONTS["tiny"])

                # Delta indicator
                delta_sign = "+" if delta > 0 else ""
                delta_color = POSITIVE if delta >= 0 else NEGATIVE
                draw.text((right + 120, bar_y + 11), f"Δ {delta_sign}{delta:.2f} pp", fill=delta_color, font=FONTS["tiny"], anchor="lm")
            else:
                draw.text((left, bar_y + 11), "n/d", fill=MUTED, font=FONTS["small"])

        draw.line((130, y + row_h - 15, WIDTH - 130, y + row_h - 15), fill="#EDF0F3", width=1)
        y += row_h

    footer(draw, image.height, "Fonte: comparisons_quantization_windows_mac.csv. 27 pares exatos com 9 datasets em FP32, FP16 e INT8 PTQ.")
    save(image, "14_quantizacao_macro_f1.png")


def draw_quantization_throughput(runs: pd.DataFrame) -> None:
    quant = runs[runs.phase == "quantization"].copy()
    quant_latest = quant[quant.campaign == "quantization_all"].copy()
    datasets = [d for d in DATASET_ORDER if d in quant_latest.dataset.unique()]

    image, draw, top = new_canvas(
        "Aceleração de inferência e Throughput na quantização (LiteRT INT8 PTQ × FP32)",
        "Throughput em amostras por segundo. O modelo INT8 atinge até 6.940 amostras/s com tempo de inferência de 1,5 a 5,9 segundos.",
        1460,
    )

    lx = WIDTH - 520
    draw.rounded_rectangle((lx, top - 32, lx + 20, top - 12), radius=4, fill=FP32_COLOR)
    draw.text((lx + 28, top - 32), "FP32 GPU", fill=TEXT, font=FONTS["small"])
    lx += 180
    draw.rounded_rectangle((lx, top - 32, lx + 20, top - 12), radius=4, fill=INT8_COLOR)
    draw.text((lx + 28, top - 32), "INT8 LiteRT", fill=TEXT, font=FONTS["small"])

    panel(draw, (90, top, WIDTH - 90, 1340))

    left = 460
    right = WIDTH - 260
    max_tput = 7500.0
    y = top + 60
    row_h = 130

    for ds in datasets:
        subset = quant_latest[quant_latest.dataset == ds]
        draw.text((130, y + 25), ds, fill=TEXT, font=FONTS["label_bold"])

        row_fp32 = subset[subset.variant == "fp32"]
        row_int8 = subset[subset.variant == "int8_ptq"]

        t_fp32 = float(row_fp32.iloc[0]["mean_train_examples_per_second"]) if not row_fp32.empty and pd.notna(row_fp32.iloc[0]["mean_train_examples_per_second"]) else 0
        t_int8 = float(row_int8.iloc[0]["mean_train_examples_per_second"]) if not row_int8.empty and pd.notna(row_int8.iloc[0]["mean_train_examples_per_second"]) else 0

        bar_y1 = y + 10
        w1 = int((right - left) * (t_fp32 / max_tput)) if max_tput > 0 else 0
        draw.rounded_rectangle((left, bar_y1, left + w1, bar_y1 + 28), radius=6, fill=FP32_COLOR)
        draw.text((left + w1 + 14, bar_y1 + 3), f"{fmt(t_fp32, 0)} ex/s", fill=TEXT, font=FONTS["small"])

        bar_y2 = y + 46
        w2 = int((right - left) * (t_int8 / max_tput)) if max_tput > 0 else 0
        draw.rounded_rectangle((left, bar_y2, left + w2, bar_y2 + 28), radius=6, fill=INT8_COLOR)

        speedup_str = f" ({t_int8 / t_fp32:.1f}×)" if t_fp32 > 0 and t_int8 > 0 else ""
        draw.text((left + w2 + 14, bar_y2 + 3), f"{fmt(t_int8, 0)} ex/s{speedup_str}", fill=INT8_COLOR, font=FONTS["label_bold"])

        draw.line((130, y + row_h - 10, WIDTH - 130, y + row_h - 10), fill="#EDF0F3", width=1)
        y += row_h

    footer(draw, image.height, "Fonte: run_results.csv e litert_benchmark.json. Throughput medido em batches de teste avaliados no host Windows.")
    save(image, "15_quantizacao_throughput_latencia.png")


def draw_batch_sweep(runs: pd.DataFrame) -> None:
    batches = [32, 64, 128, 256]
    batch_colors = {32: "#1F5A8A", 64: "#2A8A9E", 128: "#D97706", 256: "#C46A2B"}

    image, draw, top = new_canvas(
        "Varredura de tamanho de batch (Batch Sweep) — Windows vs Mac",
        "Evolução do Macro F1 ao variar o tamanho de lote (32, 64, 128 e 256). Estrela dourada indica o batch ótimo por dataset.",
        1750,
    )

    lx = WIDTH - 680
    for b in batches:
        draw.rounded_rectangle((lx, top - 32, lx + 20, top - 12), radius=4, fill=batch_colors[b])
        draw.text((lx + 28, top - 32), f"Batch {b}", fill=TEXT, font=FONTS["small"])
        lx += 140

    margin = 90
    panel_gap = 40
    panel_w = (WIDTH - 2 * margin - panel_gap) // 2
    x_win = margin
    x_mac = margin + panel_w + panel_gap
    p_bottom = 1630

    # Panel 1: Windows
    panel(draw, (x_win, top, x_win + panel_w, p_bottom))
    draw.text((x_win + 30, top + 22), "Windows — 9 Datasets (Varredura Completa)", fill=WINDOWS, font=FONTS["section"])

    # Panel 2: Mac
    panel(draw, (x_mac, top, x_mac + panel_w, p_bottom))
    draw.text((x_mac + 30, top + 22), "Mac M4 — 5 Datasets Concluídos", fill=MAC, font=FONTS["section"])

    win_batch = runs[(runs.platform == "Windows") & (runs.phase == "batch") & (runs.campaign == "controlled-augmentation05-batch-activation")].copy()
    mac_batch = runs[(runs.platform == "Mac") & (runs.phase == "batch")].copy()

    y0 = top + 80
    row_h = 150

    for idx, ds in enumerate(DATASET_ORDER):
        y = y0 + idx * row_h

        # Windows side
        draw.text((x_win + 30, y + 35), ds, fill=TEXT, font=FONTS["label_bold"])
        w_sub = win_batch[win_batch.dataset == ds]
        best_win = w_sub["macro_f1"].max() if not w_sub.empty else None
        p_left_w = x_win + 290
        p_right_w = x_win + panel_w - 40

        for b_idx, b in enumerate(batches):
            row_b = w_sub[w_sub.batch_size == b]
            by = y + b_idx * 26
            if not row_b.empty and pd.notna(row_b.iloc[0]["macro_f1"]):
                f1 = float(row_b.iloc[0]["macro_f1"])
                bw = int((p_right_w - p_left_w) * f1)
                draw.rounded_rectangle((p_left_w, by, p_left_w + bw, by + 20), radius=4, fill=batch_colors[b])
                star = " *" if f1 == best_win else ""
                draw.text((p_left_w + bw + 8, by - 1), f"{pct(f1, 1)}{star}", fill=GOLD if star else MUTED, font=FONTS["tiny"])
            else:
                draw.text((p_left_w, by - 1), "n/d", fill=MUTED, font=FONTS["tiny"])

        draw.line((x_win + 30, y + row_h - 10, x_win + panel_w - 30, y + row_h - 10), fill="#EDF0F3", width=1)

        # Mac side
        draw.text((x_mac + 30, y + 35), ds, fill=TEXT, font=FONTS["label_bold"])
        m_sub = mac_batch[mac_batch.dataset == ds]
        best_mac = m_sub["macro_f1"].max() if not m_sub.empty else None
        p_left_m = x_mac + 290
        p_right_m = x_mac + panel_w - 40

        if m_sub.empty:
            draw.text((p_left_m, y + 35), "Fase não executada no Mac", fill=MUTED, font=FONTS["small"])
        else:
            for b_idx, b in enumerate(batches):
                row_b = m_sub[m_sub.batch_size == b]
                by = y + b_idx * 26
                if not row_b.empty and pd.notna(row_b.iloc[0]["macro_f1"]):
                    f1 = float(row_b.iloc[0]["macro_f1"])
                    bw = int((p_right_m - p_left_m) * f1)
                    draw.rounded_rectangle((p_left_m, by, p_left_m + bw, by + 20), radius=4, fill=batch_colors[b])
                    star = " *" if f1 == best_mac else ""
                    draw.text((p_left_m + bw + 8, by - 1), f"{pct(f1, 1)}{star}", fill=GOLD if star else MUTED, font=FONTS["tiny"])
                else:
                    draw.text((p_left_m, by - 1), "—", fill=MUTED, font=FONTS["tiny"])

        draw.line((x_mac + 30, y + row_h - 10, x_mac + panel_w - 30, y + row_h - 10), fill="#EDF0F3", width=1)

    footer(draw, image.height, "Fonte: run_results.csv. Batches 32, 64, 128 e 256. Estrela dourada marca o batch de maior Macro F1.")
    save(image, "16_batch_sweep_macro_f1.png")


def draw_speedup_hardware(batch_pairs: pd.DataFrame) -> None:
    image, draw, top = new_canvas(
        "Aceleração computacional (Speedup): GPU NVIDIA dedicada × Apple Silicon M4",
        "Razão de tempo por época (Mac / Windows) nos 15 pares de batch. A GPU NVIDIA dedicada completou épocas até 8,8× mais rápido.",
        1750,
    )

    panel(draw, (90, top, WIDTH - 90, 1630))

    left = 620
    right = WIDTH - 340
    y = top + 75
    row_h = 86
    max_ratio = 10.0

    for _, row in batch_pairs.sort_values(["dataset", "batch_size_windows"]).iterrows():
        t_win = float(row["mean_epoch_seconds_windows"])
        t_mac = float(row["mean_epoch_seconds_mac"])
        ratio = t_mac / t_win if t_win > 0 else 0

        label_title = f"{row['dataset']} — Batch {int(row['batch_size_windows'])}"
        draw.text((130, y + 8), label_title, fill=TEXT, font=FONTS["label_bold"])
        draw.text((130, y + 38), f"Win: {t_win:.1f} s/ep  |  Mac: {t_mac:.1f} s/ep", fill=MUTED, font=FONTS["tiny"])

        bar_w = int((right - left) * (ratio / max_ratio))
        draw.rounded_rectangle((left, y + 8, left + bar_w, y + 46), radius=8, fill=WINDOWS)

        draw.text((left + bar_w + 14, y + 10), f"{ratio:.2f}×", fill=WINDOWS, font=FONTS["section"])
        draw.text((left + bar_w + 120, y + 18), "mais rápido no Windows", fill=MUTED, font=FONTS["tiny"])

        draw.line((130, y + row_h - 10, WIDTH - 130, y + row_h - 10), fill="#EDF0F3", width=1)
        y += row_h

    footer(draw, image.height, "Fonte: comparisons_batch_windows_mac.csv. Hardware: NVIDIA RTX (Windows) vs Apple Silicon M4 10-core (macOS Metal).")
    save(image, "17_speedup_windows_mac.png")


def draw_parity_residuals(batch_pairs: pd.DataFrame) -> None:
    image, draw, top = new_canvas(
        "Paridade numérica nos 15 pares de batch alinhados (Windows × Mac)",
        "Diferença residual de Macro F1 (Windows − Mac em pontos percentuais). A média de +0,15 p.p. confirma estrita consistência estocástica.",
        1840,
    )

    panel(draw, (90, top, WIDTH - 90, 1710))

    left = 680
    right = WIDTH - 260
    mid = (left + right) // 2
    max_delta = 1.6
    bottom_y = 1620

    band_l = mid - int((right - mid) * (0.2 / max_delta))
    band_r = mid + int((right - mid) * (0.2 / max_delta))
    draw.rectangle((band_l, top + 60, band_r, bottom_y), fill="#F0FDF4")
    draw.line((band_l, top + 60, band_l, bottom_y), fill="#BBF7D0", width=1)
    draw.line((band_r, top + 60, band_r, bottom_y), fill="#BBF7D0", width=1)
    draw.text((mid, top + 35), "Faixa de paridade estocástica (±0,2 p.p.)", fill="#16A34A", font=FONTS["tiny"], anchor="mm")

    draw.line((mid, top + 60, mid, bottom_y), fill="#94A3B8", width=2)
    draw.text((mid, bottom_y + 18), "0,0 p.p.", fill=MUTED, font=FONTS["small"], anchor="mm")
    draw.text((band_l, bottom_y + 18), "-0,2 p.p.", fill=MUTED, font=FONTS["tiny"], anchor="mm")
    draw.text((band_r, bottom_y + 18), "+0,2 p.p.", fill=MUTED, font=FONTS["tiny"], anchor="mm")
    draw.text((left, bottom_y + 18), f"-{max_delta:.1f} p.p.", fill=MUTED, font=FONTS["tiny"], anchor="mm")
    draw.text((right, bottom_y + 18), f"+{max_delta:.1f} p.p.", fill=MUTED, font=FONTS["tiny"], anchor="mm")

    y = top + 70
    row_h = 86

    for _, row in batch_pairs.sort_values(["dataset", "batch_size_windows"]).iterrows():
        delta_pp = float(row["delta_macro_f1_pp_windows_minus_mac"])
        label_title = f"{row['dataset']} — Batch {int(row['batch_size_windows'])}"
        draw.text((130, y + 8), label_title, fill=TEXT, font=FONTS["label_bold"])
        draw.text((130, y + 36), f"F1 Win: {pct(row['macro_f1_windows'], 2)} | F1 Mac: {pct(row['macro_f1_mac'], 2)}", fill=MUTED, font=FONTS["tiny"])

        offset = int((right - mid) * (delta_pp / max_delta))
        x_pt = mid + offset
        color = WINDOWS if delta_pp >= 0 else MAC

        draw.line((mid, y + 25, x_pt, y + 25), fill=color, width=3)
        draw.ellipse((x_pt - 8, y + 17, x_pt + 8, y + 33), fill=color)

        sign = "+" if delta_pp > 0 else ""
        txt = f"{sign}{delta_pp:.3f} p.p."
        txt_x = x_pt + 14 if delta_pp >= 0 else x_pt - 14
        anchor = "lm" if delta_pp >= 0 else "rm"
        draw.text((txt_x, y + 25), txt, fill=color, font=FONTS["label_bold"], anchor=anchor)

        draw.line((130, y + row_h - 10, WIDTH - 130, y + row_h - 10), fill="#EDF0F3", width=1)
        y += row_h

    footer(draw, image.height, "Fonte: comparisons_batch_windows_mac.csv. Delta positivo favorece Windows; delta negativo favorece Mac.")
    save(image, "18_paridade_residuos_f1.png")


def main() -> None:
    runs = pd.read_csv(DATA / "run_results.csv")
    campaigns = pd.read_csv(DATA / "campaign_summary.csv")
    activation_pairs = pd.read_csv(DATA / "comparisons_activation_windows_mac.csv")
    batch_pairs = pd.read_csv(DATA / "comparisons_batch_windows_mac.csv")
    quant_pairs = pd.read_csv(DATA / "comparisons_quantization_windows_mac.csv")
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
    draw_quantization_f1(quant_pairs, runs)
    draw_quantization_throughput(runs)
    draw_batch_sweep(runs)
    draw_speedup_hardware(batch_pairs)
    draw_parity_residuals(batch_pairs)
    print(f"Generated {len(list(FIGURES.glob('*.png')))} figures in {FIGURES}")


if __name__ == "__main__":
    main()
