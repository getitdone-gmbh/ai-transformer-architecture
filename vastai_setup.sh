#!/usr/bin/env bash
# Setup on a fresh Vast.ai GPU instance.
# Prerequisite: a PyTorch image (e.g. "pytorch/pytorch" or a
# vastai/pytorch template) with a CUDA-capable torch already installed.
set -euo pipefail

echo "== Python / Torch =="
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

echo "== Installing missing deps =="
pip install --no-cache-dir tiktoken datasets numpy

echo "== Done. Start training with: =="
echo "  bash run_124m.sh"
