#!/usr/bin/env bash
# Setup auf einer frischen Vast.ai GPU-Instanz.
# Voraussetzung: ein PyTorch-Image (z.B. "pytorch/pytorch" oder ein
# vastai/pytorch-Template) mit CUDA-faehigem torch bereits installiert.
set -euo pipefail

echo "== Python / Torch =="
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'keine GPU')"

echo "== Fehlende Deps installieren =="
pip install --no-cache-dir tiktoken datasets numpy

echo "== Fertig. Training starten mit: =="
echo "  bash run_124m.sh"
