#!/usr/bin/env bash
# Hauptlauf: ~355M Parameter (GPT-2-medium-Klasse) auf ~10 Mrd. Tokens.
#
# VORHER auf der Instanz ausfuehren (einmalig, dauert einige Stunden):
#   python prepare_data.py          # baut shards/ (~20 GB, FineWeb2 + Wiki)
#
# Architektur: d_model=1024, 24 Layer, 16 Heads (GPT-2 medium Shape).
# D_FF=2752: SwiGLU-Konvention (2/3)*4*d_model ~ 2731, aufgerundet auf ein
# Vielfaches von 64 (GPU-freundliche Matrixgroessen).
#
# DROPOUT=0.0: bei EINEM Pass ueber 10 Mrd. frische Tokens sieht das Modell
# jedes Beispiel genau einmal — Overfitting ist strukturell unmoeglich,
# Dropout wuerde nur Kapazitaet verschenken (Llama-Recipe).
#
# LR 3e-4: grosse Modelle brauchen kleinere Peak-LRs (GPT-2 medium Standard).
# WARMUP 2000: laengerer Anlauf fuer den groesseren Parameterraum.
#
# Effektiv: 8 x 8 = 64 Sequenzen x 1024 Tokens = 65.536 Tokens/Step.
# 10 Mrd. Tokens / 65k = ~150.000 Steps -> je nach Host ~4-6 Tage auf einer
# RTX 4090. Rolling-Checkpoint alle 2000 Steps sichert den Fortschritt.
#
# Bei "CUDA out of memory": BATCH_SIZE=4 GRAD_ACCUM_STEPS=16.
set -euo pipefail

D_MODEL=1024 NUM_HEADS=16 NUM_LAYERS=24 D_FF=2752 \
SEQ_LENGTH=1024 \
BATCH_SIZE=8 GRAD_ACCUM_STEPS=8 \
SHARD_MANIFEST=shards/manifest.json \
NUM_EPOCHS=1 \
LEARNING_RATE=3e-4 MIN_LR=3e-5 WARMUP_STEPS=2000 \
DROPOUT=0.0 \
python train.py 2>&1 | tee train_350m.log
