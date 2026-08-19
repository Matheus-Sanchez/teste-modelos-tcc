#!/usr/bin/env bash
# Install the supported WSL runtime and the five controlled-benchmark datasets.
# Run this inside Ubuntu WSL, at the repository root. It intentionally does
# not start training; use run_controlled_pipeline.py only after preflight passes.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -f /proc/version ]] || ! grep -qi microsoft /proc/version; then
  echo "Este bootstrap deve ser executado dentro de uma distribuição Ubuntu no WSL2." >&2
  exit 2
fi

sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/wsl-gpu.txt
python scripts/fetch_datasets.py --dataset mnist
python scripts/fetch_datasets.py --dataset fashion_mnist
python scripts/fetch_datasets.py --dataset kmnist
python scripts/fetch_datasets.py --dataset emnist_balanced
python scripts/fetch_datasets.py --dataset cifar10
python scripts/fetch_datasets.py --dataset cifar100_coarse
python scripts/fetch_datasets.py --dataset svhn
python scripts/fetch_datasets.py --dataset gtsrb
python scripts/fetch_datasets.py --dataset fer2013
PYTHON_BIN="$project_root/.venv/bin/python" scripts/wsl-gpu-env.sh \
  "$project_root/.venv/bin/tcc-benchmark" preflight --require-tensorflow \
  --output-root "$project_root/outputs/controlled-augmentation2" \
  --data-path "$project_root/datasets"
PYTHON_BIN="$project_root/.venv/bin/python" scripts/wsl-gpu-env.sh \
  "$project_root/.venv/bin/python" -c 'import tensorflow as tf; gpus = tf.config.list_physical_devices("GPU"); print(gpus); raise SystemExit(0 if gpus else "Nenhuma GPU TensorFlow foi detectada.")'
