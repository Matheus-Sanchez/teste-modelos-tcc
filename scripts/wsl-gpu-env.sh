#!/usr/bin/env bash
# Run a command with the CUDA libraries installed by tensorflow[and-cuda]
# visible to TensorFlow inside WSL. Invoke this only after activating the
# project's Python 3.10 virtual environment.
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Uso: $0 <comando> [argumentos...]" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
site_packages="$($python_bin -c 'import site; print(site.getsitepackages()[0])')"
nvidia_root="$site_packages/nvidia"

if [[ -d "$nvidia_root" ]]; then
  cuda_libs="$(find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -path '*/lib' -print | paste -sd: -)"
  if [[ -n "$cuda_libs" ]]; then
    export LD_LIBRARY_PATH="$cuda_libs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

# The runner also requests memory growth, but exporting this before TensorFlow
# starts prevents an eager full-VRAM allocation during any auxiliary command.
# Quantization runs are isolated and compare VRAM under the same policy.  Do
# not inherit a caller's false value, which would make TensorFlow reserve all
# VRAM before the benchmark can configure memory growth.
export TF_FORCE_GPU_ALLOW_GROWTH=true

exec "$@"
