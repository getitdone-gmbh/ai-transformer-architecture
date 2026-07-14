#!/usr/bin/env bash
# SFT of the 540M base model on a CUDA instance (Vast.ai).
#
# Uses the fp16 weights (half the upload; sft.py casts to fp32 on load).
# Micro-batch 8 on 24 GB VRAM: the fixed costs (params + grads + Adam
# moments, all fp32) are ~8.6 GB, the rest goes to activations and the
# fp32 softmax over the logits — batch 16 is too much for that
# (empirically: OOM at 20.7/23.5 GB). Effective batch stays 8*4 = 32.
set -euo pipefail

export WEIGHTS=${WEIGHTS:-weights_540m_fp16.pt}
export OUT_PATH=${OUT_PATH:-sft_540m.pt}
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-4}
# Grow buffers instead of re-allocating: prevents the collate's 8 bucket
# widths from fragmenting the allocator.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u sft.py 2>&1 | tee sft_540m.log
