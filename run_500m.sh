#!/usr/bin/env bash
# Main run: ~540M parameters on ~10 billion tokens (FineWeb2 + Wiki + Python).
#
# BEFORE this, run on the instance (once, several hours):
#   nohup python prepare_data.py > prepare.log 2>&1 &
#   (Default mix: 7B FineWeb2 German + 2B Wikipedia + 1B Python.
#    German only: CODE_TOKENS=0 FINEWEB_TOKENS=8e9 python prepare_data.py)
#
# GPU choice: H100 80GB (recommended) — a 540M model with large micro-batches
# keeps an H100 well utilized, and then it is cheaper per FLOP than a 4090:
#   H100 80GB:  ~1.5 days,  ~$60-75   <- default settings below
#   RTX 4090:   ~8 days,    ~$70-80   <- set BATCH_SIZE=3 GRAD_ACCUM_STEPS=10
#
# Architecture: d_model=1280, 24 layers, 20 heads (64-dim heads).
# D_FF=3456: SwiGLU convention (2/3)*4*1280 ~ 3413, rounded up to a
# multiple of 64. Adds up to ~540M parameters.
#
# LR 2.5e-4: the larger the model, the smaller the stable peak LR
# (GPT-3 table: 760M -> 2.5e-4). DROPOUT=0.0: one pass over fresh
# tokens, overfitting structurally impossible (see run_350m.sh).
#
# SEQ_LENGTH=2048: double the context window for ~+10% compute — the
# cheapest context you will ever buy (retrofitting it later = more
# expensive/worse). BATCH_SIZE halved (16 x 2048 = 32 x 1024 tokens)
# -> tokens/step identical, training math unchanged.
#
# Effective: 16 x 2 = 32 sequences x 2048 tokens = 65,536 tokens/step.
# 10 billion tokens -> ~150,000 steps. Rolling checkpoint every 2000 steps.
set -euo pipefail

D_MODEL=1280 NUM_HEADS=20 NUM_LAYERS=24 D_FF=3456 \
SEQ_LENGTH=2048 \
BATCH_SIZE=16 GRAD_ACCUM_STEPS=2 \
SHARD_MANIFEST=shards/manifest.json \
NUM_EPOCHS=1 \
LEARNING_RATE=2.5e-4 MIN_LR=2.5e-5 WARMUP_STEPS=2000 \
DROPOUT=0.0 \
python train.py 2>&1 | tee train_500m.log
