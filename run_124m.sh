#!/usr/bin/env bash
# Startet den 124M-Trainingslauf. Alle Knoepfe explizit als ENV-Vars, damit
# der Lauf reproduzierbar ist und man sie leicht anpassen kann (siehe VASTAI.md).
#
# Bei "CUDA out of memory": BATCH_SIZE runter (z.B. 8) und GRAD_ACCUM_STEPS
# hoch (z.B. 6), damit die effektive Batch (BATCH_SIZE*GRAD_ACCUM_STEPS) gleich
# bleibt.
set -euo pipefail

D_MODEL=768 NUM_HEADS=12 NUM_LAYERS=12 D_FF=2048 \
SEQ_LENGTH=512 \
BATCH_SIZE=16 GRAD_ACCUM_STEPS=3 \
NUM_ARTICLES=400000 NUM_EPOCHS=1 \
LEARNING_RATE=6e-4 \
python train.py 2>&1 | tee train_124m.log
