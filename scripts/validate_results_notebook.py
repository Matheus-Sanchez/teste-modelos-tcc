"""Validate and execute the results notebook with the project Python environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "benchmark_results_windows.ipynb"


def main() -> int:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    namespace = {"__file__": str(NOTEBOOK_PATH)}
    for index, cell in enumerate(code_cells, start=1):
        source = "".join(cell["source"])
        compile(source, f"{NOTEBOOK_PATH}:cell-{index}", "exec")
        exec(compile(source, f"{NOTEBOOK_PATH}:cell-{index}", "exec"), namespace)
    print(f"Validated and executed {len(code_cells)} code cells from {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
