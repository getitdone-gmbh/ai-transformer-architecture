#!/usr/bin/env bash
# Hauptlauf: ~540M Parameter auf ~10 Mrd. Tokens (FineWeb2 + Wiki + Python).
#
# VORHER auf der Instanz ausfuehren (einmalig, mehrere Stunden):
#   nohup python prepare_data.py > prepare.log 2>&1 &
#   (Default-Mix: 7 Mrd. FineWeb2-Deutsch + 2 Mrd. Wikipedia + 1 Mrd. Python.
#    Reines Deutsch: CODE_TOKENS=0 FINEWEB_TOKENS=8e9 python prepare_data.py)
#
# GPU-Wahl: H100 80GB (empfohlen) — ein 540M-Modell mit grossen Micro-Batches
# lastet eine H100 gut aus, dann ist sie pro FLOP guenstiger als eine 4090:
#   H100 80GB:  ~1-1.5 Tage, ~55-70 $   <- Standard-Settings unten
#   RTX 4090:   ~7-8 Tage,   ~65-75 $   <- BATCH_SIZE=6 GRAD_ACCUM_STEPS=10 setzen
#
# Architektur: d_model=1280, 24 Layer, 20 Heads (64-dim Koepfe).
# D_FF=3456: SwiGLU-Konvention (2/3)*4*1280 ~ 3413, aufgerundet auf ein
# Vielfaches von 64. Macht zusammen ~540M Parameter.
#
# LR 2.5e-4: je groesser das Modell, desto kleiner die stabile Peak-LR
# (GPT-3-Tabelle: 760M -> 2.5e-4). DROPOUT=0.0: ein Pass ueber frische
# Tokens, Overfitting strukturell unmoeglich (siehe run_350m.sh).
#
# Effektiv: 32 x 2 = 64 Sequenzen x 1024 Tokens = 65.536 Tokens/Step.
# 10 Mrd. Tokens -> ~150.000 Steps. Rolling-Checkpoint alle 2000 Steps.
set -euo pipefail

D_MODEL=1280 NUM_HEADS=20 NUM_LAYERS=24 D_FF=3456 \
SEQ_LENGTH=1024 \
BATCH_SIZE=32 GRAD_ACCUM_STEPS=2 \
SHARD_MANIFEST=shards/manifest.json \
NUM_EPOCHS=1 \
LEARNING_RATE=2.5e-4 MIN_LR=2.5e-5 WARMUP_STEPS=2000 \
DROPOUT=0.0 \
python train.py 2>&1 | tee train_500m.log
