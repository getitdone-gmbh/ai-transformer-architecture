#!/usr/bin/env bash
# Main run: ~355M parameters (GPT-2-medium class) on ~10 billion tokens.
#
# BEFORE this, run on the instance (once, takes several hours):
#   python prepare_data.py          # builds shards/ (~20 GB, FineWeb2 + Wiki)
#
# Architecture: d_model=1024, 24 layers, 16 heads (GPT-2 medium shape).
# D_FF=2752: SwiGLU convention (2/3)*4*d_model ~ 2731, rounded up to a
# multiple of 64 (GPU-friendly matrix sizes).
#
# DROPOUT=0.0: with ONE pass over 10 billion fresh tokens the model sees
# every example exactly once — overfitting is structurally impossible,
# dropout would only throw away capacity (Llama recipe).
#
# LR 3e-4: large models need smaller peak LRs (GPT-2 medium standard).
# WARMUP 2000: longer ramp-up for the larger parameter space.
#
# Effective: 8 x 8 = 64 sequences x 1024 tokens = 65,536 tokens/step.
# 10 billion tokens / 65k = ~150,000 steps -> depending on the host, ~4-6
# days on an RTX 4090. Rolling checkpoint every 2000 steps saves progress.
#
# On "CUDA out of memory": BATCH_SIZE=4 GRAD_ACCUM_STEPS=16.
set -euo pipefail

D_MODEL=1024 NUM_HEADS=16 NUM_LAYERS=24 D_FF=2752 \
SEQ_LENGTH=1024 \
BATCH_SIZE=8 GRAD_ACCUM_STEPS=8 \
SHARD_MANIFEST=shards/manifest.json \
NUM_EPOCHS=1 \
LEARNING_RATE=3e-4 MIN_LR=3e-5 WARMUP_STEPS=2000 \
DROPOUT=0.0 \
python train.py 2>&1 | tee train_350m.log
