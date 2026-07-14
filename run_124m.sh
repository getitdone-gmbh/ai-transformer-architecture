#!/usr/bin/env bash
# Starts the 124M training run. All knobs are explicit ENV vars so the run
# is reproducible and easy to tweak (see VASTAI.md).
#
# On "CUDA out of memory": lower BATCH_SIZE (e.g. 8) and raise GRAD_ACCUM_STEPS
# (e.g. 6) so the effective batch (BATCH_SIZE*GRAD_ACCUM_STEPS) stays the
# same.
set -euo pipefail

D_MODEL=768 NUM_HEADS=12 NUM_LAYERS=12 D_FF=2048 \
SEQ_LENGTH=512 \
BATCH_SIZE=16 GRAD_ACCUM_STEPS=3 \
NUM_ARTICLES=400000 NUM_EPOCHS=1 \
LEARNING_RATE=6e-4 \
python train.py 2>&1 | tee train_124m.log
