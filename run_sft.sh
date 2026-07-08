#!/usr/bin/env bash
# SFT des 540M-Basismodells auf einer CUDA-Instanz (Vast.ai).
#
# Nimmt die fp16-Gewichte (halber Upload; sft.py castet beim Laden auf
# fp32). Micro-Batch 8 auf 24 GB VRAM: die Fixkosten (Params + Grads +
# Adam-Momente, alles fp32) sind ~8,6 GB, der Rest geht an Aktivierungen
# und den fp32-Softmax ueber die Logits — Batch 16 ist dafuer zu viel
# (empirisch: OOM bei 20,7/23,5 GB). Effektiver Batch bleibt 8*4 = 32.
set -euo pipefail

export WEIGHTS=${WEIGHTS:-weights_540m_fp16.pt}
export OUT_PATH=${OUT_PATH:-sft_540m.pt}
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-4}
# Puffer wachsen lassen statt neu allokieren: verhindert, dass die 8
# Bucket-Breiten des Collate den Allocator fragmentieren.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u sft.py 2>&1 | tee sft_540m.log
