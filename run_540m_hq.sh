#!/usr/bin/env bash
# Continued pretraining: +5B tokens on top of the finished 540M base model.
# The experiment: "what does data QUALITY buy at fixed N?" — see IDEAS.md.
#
# Mix: 4.5B FineWeb2-HQ German (EPFL's model-based top-10% quality filter
# over FineWeb2) + 0.5B Python from codeparrot. The code is REPLAY, not a
# code push: continued training only preserves what the gradient keeps
# seeing — a German-only diet would slowly recruit the code circuits for
# German (catastrophic forgetting). Reading codeparrot from the START
# (files the model already saw) is fine for replay and guaranteed disjoint
# from the frozen code eval set (built from the LAST codeparrot file).
#
# BEFORE this, on the instance:
#   1. The starting weights: copy weights_540m_fp32.pt here.
#   2. The data (once, a few hours — note the fresh SHARD_DIR):
#        FINEWEB_TOKENS=0 WIKI_TOKENS=0 CODE_TOKENS=5e8 HQ_TOKENS=4.5e9 \
#        SHARD_DIR=shards_hq nohup python prepare_data.py > prepare_hq.log 2>&1 &
#      Requires eval_exclude_hashes.json (in the repo): the contamination
#      guard that keeps the frozen bpb eval set out of the training data.
#      prepare_data.py refuses to build HQ without it.
#
# GPU: H100 80GB, ~18-20 h, ~$35 — same batch geometry as run_500m.sh.
#
# LR 2.5e-5 = 10% of the original run's peak: a converged model hit with
# a full-height LR destroys structure faster than it learns new — low
# peak + short warmup is the standard continued-pretraining recipe.
# WARMUP_STEPS 200 (not 2000): the warmup only has to protect the fresh
# Adam moment estimates, not a random init.
#
# WARNING — fresh working directory: AUTO_RESUME picks up any
# checkpoint_latest.pt it finds. If a checkpoint from the ORIGINAL run
# lies around, the warm start silently becomes a resume of that run
# (INIT_FROM only applies when no resume checkpoint is found — which is
# exactly what a crashed HQ run needs to resume itself).
#
# 5B tokens / 65,536 tokens-per-step ≈ 76,000 optimizer steps.
#
# AFTER the run, locally: compare_bpb.py + cloze_eval.py against the
# baseline numbers in IDEAS.md — same frozen eval set, honest before/after.
set -euo pipefail

D_MODEL=1280 NUM_HEADS=20 NUM_LAYERS=24 D_FF=3456 \
SEQ_LENGTH=2048 \
BATCH_SIZE=16 GRAD_ACCUM_STEPS=2 \
SHARD_MANIFEST=shards_hq/manifest.json \
INIT_FROM=weights_540m_fp32.pt \
NUM_EPOCHS=1 \
LEARNING_RATE=2.5e-5 MIN_LR=2.5e-6 WARMUP_STEPS=200 \
DROPOUT=0.0 \
python train.py 2>&1 | tee train_540m_hq.log
