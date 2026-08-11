#!/usr/bin/env python3
"""Run the official report-to-google-doc helper with a rounding-safe preflight.

The upstream width calculator rounds every column independently to one decimal
place.  For some ten-column tables this can make the *reported* width 0.3 pt
wider than the document text column, although the DOCX writer itself uses the
exact page width.  This wrapper keeps the official parser, writer, renderer,
manifest, and checks, and only assigns the rounding remainder to the final
column during the width-cap check.
"""

from __future__ import annotations

import sys
from pathlib import Path


PLUGIN_SCRIPTS = Path(
    r"C:\Users\maths\.codex\plugins\cache\openai-curated-remote"
    r"\data-analytics\0.2.8-13ceeea1f599\skills\build-report"
    r"\report-to-google-doc\scripts"
)
sys.path.insert(0, str(PLUGIN_SCRIPTS))

from report_to_google_doc import quality, table_utils  # noqa: E402
from report_to_google_doc.cli import main  # noqa: E402


def rounding_safe_table_column_widths(table: dict, columns: int) -> list[float]:
    """Preserve upstream widths while preventing cumulative round-up overflow."""
    widths = table_utils.table_column_widths(table, columns)
    if not widths:
        return widths

    total_width = round(float(table.get("docs_width_pt", table_utils.DOC_CONTENT_WIDTH_PT)), 1)
    rounded_total = round(sum(widths), 1)
    if rounded_total > total_width:
        widths[-1] = round(widths[-1] - (rounded_total - total_width), 1)
    return widths


quality.table_column_widths = rounding_safe_table_column_widths


if __name__ == "__main__":
    main()
