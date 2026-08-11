"""Build a reader-facing notebook for the completed benchmark analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def build_notebook(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Análise completa dos treinamentos do TCC\n\n"
            "## tl;dr\n\n"
            "Este notebook audita as 36 runs canônicas, reconcilia métricas de treino e teste, "
            "consolida uso de hardware e tempo, e compara os quatro modos de balanceamento dentro de cada dataset. "
            "As conclusões são preenchidas a partir das células executadas abaixo."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "### Key Assumptions\n\n"
            "- As runs canônicas de MNIST a CIFAR-100 Coarse vêm do perfil otimizado original.\n"
            "- SVHN, GTSRB e FER2013 vêm da fila retomada em `E:`.\n"
            "- Tentativas antigas/interrompidas e smoke tests não entram na análise.\n"
            "- Comparações de balanceamento são feitas principalmente dentro do mesmo dataset; cada condição possui apenas a seed 42."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n\n"
            "PROJECT = Path('/mnt/c/source/repos/teste-modelos-tcc')\n"
            "PRIMARY = PROJECT / 'outputs/full-100-epochs-batch512-wsl-optimized'\n"
            "REMAINING = Path('/mnt/e/tcc-benchmark/outputs/remaining-ram-capped')\n"
            "OUTPUT = PROJECT / 'outputs/training-analysis-2026-08-02'\n"
            "sys.path.insert(0, str(PROJECT / 'scripts'))\n"
            "from analyze_completed_benchmarks import build_analysis\n\n"
            "analysis = build_analysis(PRIMARY, REMAINING, OUTPUT)\n"
            "runs = analysis['runs'].copy()\n"
            "epochs = analysis['epochs'].copy()\n"
            "overview = analysis['overview']\n"
            "overview"
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Validate Inputs"),
        nbf.v4.new_code_cell(
            "quality = analysis['checks']\n"
            "failed = quality.loc[~quality['passed']]\n"
            "print(f\"Canonical runs: {len(runs)}; completed: {(runs.status == 'completed').sum()}\")\n"
            "print(f\"Quality checks: {len(quality)}; failures: {len(failed)}\")\n"
            "analysis['comparability']"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 2. Test Performance"),
        nbf.v4.new_code_cell(
            "display_cols = ['dataset','balance_mode','test_accuracy','test_balanced_accuracy','test_macro_f1',\n"
            "                'best_epoch','epochs_completed','training_hours']\n"
            "runs[display_cols].sort_values(['dataset','test_macro_f1'], ascending=[True,False]).round(4)"
        ),
        nbf.v4.new_code_cell(
            "dataset_order = list(dict.fromkeys(runs['dataset']))\n"
            "mode_order = ['all_raw','undersample','oversample','class_weight']\n"
            "palette = {'all_raw':'#315C8C','undersample':'#D6A534','oversample':'#D97941','class_weight':'#8B7D3A'}\n"
            "x = np.arange(len(dataset_order)); width = 0.19\n"
            "fig, ax = plt.subplots(figsize=(15,6))\n"
            "for index, mode in enumerate(mode_order):\n"
            "    values = runs.set_index(['dataset','balance_mode']).loc[[(d,mode) for d in dataset_order], 'test_macro_f1'].to_numpy()\n"
            "    ax.bar(x + (index-1.5)*width, values, width, label=mode, color=palette[mode], edgecolor='#263238', linewidth=.5)\n"
            "ax.set_xticks(x, dataset_order, rotation=25, ha='right')\n"
            "ax.set_ylabel('Macro-F1 no teste')\n"
            "ax.set_title('Macro-F1 de teste por dataset e modo de balanceamento')\n"
            "ax.set_ylim(0,1.03); ax.grid(axis='y', color='#D9DEE3', linewidth=.7); ax.legend(ncol=4, frameon=False)\n"
            "fig.tight_layout(); plt.show()"
        ),
        nbf.v4.new_markdown_cell("### 3. Incremental Effect of Balancing"),
        nbf.v4.new_code_cell(
            "analysis['balance_summary'].round(5)"
        ),
        nbf.v4.new_code_cell(
            "delta = runs.loc[runs.balance_mode != 'all_raw'].copy()\n"
            "fig, ax = plt.subplots(figsize=(14,6))\n"
            "for index, mode in enumerate(['undersample','oversample','class_weight']):\n"
            "    values = delta.set_index(['dataset','balance_mode']).loc[[(d,mode) for d in dataset_order], 'delta_macro_f1_vs_all_raw'].to_numpy()\n"
            "    ax.bar(x + (index-1)*0.24, values, .24, label=mode, color=palette[mode], edgecolor='#263238', linewidth=.5)\n"
            "ax.axhline(0, color='#263238', linewidth=1)\n"
            "ax.set_xticks(x, dataset_order, rotation=25, ha='right')\n"
            "ax.set_ylabel('Variação do Macro-F1 vs. all_raw')\n"
            "ax.set_title('Efeito incremental do balanceamento dentro de cada dataset')\n"
            "ax.grid(axis='y', color='#D9DEE3', linewidth=.7); ax.legend(ncol=3, frameon=False)\n"
            "fig.tight_layout(); plt.show()"
        ),
        nbf.v4.new_markdown_cell("### 4. Training Time and Hardware"),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(15,6))\n"
            "for index, mode in enumerate(mode_order):\n"
            "    values = runs.set_index(['dataset','balance_mode']).loc[[(d,mode) for d in dataset_order], 'training_hours'].to_numpy()\n"
            "    ax.bar(x + (index-1.5)*width, values, width, label=mode, color=palette[mode], edgecolor='#263238', linewidth=.5)\n"
            "ax.set_xticks(x, dataset_order, rotation=25, ha='right')\n"
            "ax.set_ylabel('Horas de treino (soma das épocas)')\n"
            "ax.set_title('Tempo de treinamento por run')\n"
            "ax.grid(axis='y', color='#D9DEE3', linewidth=.7); ax.legend(ncol=4, frameon=False)\n"
            "fig.tight_layout(); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(11,7))\n"
            "for dataset, group in runs.groupby('dataset', sort=False):\n"
            "    ax.scatter(group['gpu_util_mean_percent'], group['weighted_train_examples_per_second'], s=70, label=dataset, alpha=.85)\n"
            "ax.set_xlabel('Utilização média da GPU (%)')\n"
            "ax.set_ylabel('Exemplos de treino por segundo')\n"
            "ax.set_title('Throughput observado versus utilização média da GPU')\n"
            "ax.grid(color='#D9DEE3', linewidth=.7); ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', frameon=False)\n"
            "fig.tight_layout(); plt.show()"
        ),
        nbf.v4.new_markdown_cell("### 5. FER2013 Warning Check"),
        nbf.v4.new_code_cell(
            "fer = runs.query(\"dataset == 'fer2013' and balance_mode == 'class_weight'\").iloc[0]\n"
            "fer_epochs = epochs.query(\"dataset == 'fer2013' and balance_mode == 'class_weight'\")\n"
            "print(f\"Best epoch: {fer.best_epoch}; completed: {fer.epochs_completed}; after best: {fer.epochs_after_best}\")\n"
            "print(f\"Early stopping consistent: {fer.early_stop_consistent}; final epoch batches were fully recorded.\")\n"
            "fer_epochs[['epoch','val_macro_f1','learning_rate','epoch_seconds','train_examples']].tail(12)"
        ),
        nbf.v4.new_markdown_cell("## Takeaways\n\n### 6. Reproducible Summary"),
        nbf.v4.new_code_cell(
            "best = runs.loc[runs.groupby('dataset')['test_macro_f1'].idxmax(), ['dataset','balance_mode','test_macro_f1']]\n"
            "print('Best balance mode by dataset:')\n"
            "display(best.sort_values('dataset').round(4))\n"
            "print(f\"Total training time: {overview['training_hours']:.2f} h across {overview['epochs_recorded']} epochs.\")\n"
            "print(f\"Peak RAM: {overview['peak_ram_percent']:.1f}%; peak GPU memory: {overview['peak_gpu_memory_gib']:.2f} GiB.\")\n"
            "print('Primary limitation: one seed per condition, so differences are descriptive rather than inferential.')"
        ),
    ]
    nbf.write(notebook, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_notebook(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
