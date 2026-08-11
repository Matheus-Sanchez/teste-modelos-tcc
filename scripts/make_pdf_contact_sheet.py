#!/usr/bin/env python3
"""Create a labeled contact sheet from rendered PDF pages for visual QA."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: make_pdf_contact_sheet.py PAGE_DIR OUTPUT_PNG")
    page_dir = Path(sys.argv[1])
    output = Path(sys.argv[2])
    pages = sorted(page_dir.glob("page-*.png"))
    if not pages:
        raise SystemExit("no rendered pages found")

    thumb_w, label_h, gap, columns = 255, 26, 14, 3
    with Image.open(pages[0]) as first:
        thumb_h = round(first.height * thumb_w / first.width)
    rows = (len(pages) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (gap + columns * (thumb_w + gap), gap + rows * (thumb_h + label_h + gap)),
        "#d8dee9",
    )
    draw = ImageDraw.Draw(sheet)
    for index, page in enumerate(pages):
        row, column = divmod(index, columns)
        x = gap + column * (thumb_w + gap)
        y = gap + row * (thumb_h + label_h + gap)
        with Image.open(page) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y + label_h))
        draw.text((x, y + 5), f"Página {index + 1}", fill="#111827")
    sheet.save(output)


if __name__ == "__main__":
    main()
