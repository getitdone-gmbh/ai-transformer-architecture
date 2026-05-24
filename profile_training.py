"""Kurzer Profiling-Run um die langsamsten Ops im Training zu finden.

Baut das Modell wie im Training auf, macht 5 Warmup-Batches (absorbiert
torch.compile JIT), dann 20 profilierte Batches. Druckt die Top-Ops
sortiert nach selbst-verbrauchter MPS-Zeit.

Laeuft in ~30-60 Sekunden, hat keinen Einfluss auf parallel laufendes
Training (separater Prozess).
"""

import time

import torch
import torch.nn as nn

from train import (
    GPTDecoder,
    create_causal_mask,
    D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, SEQ_LENGTH, LEARNING_RATE,
)

VOCAB_SIZE = 50257
NUM_WARMUP = 5
NUM_PROFILE = 20


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = GPTDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, DROPOUT).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Inputs simulieren
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), device=device)
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SEQ_LENGTH,), device=device)
    mask = create_causal_mask(SEQ_LENGTH, device)

    # Warmup — laesst torch.compile JIT-Kompilieren
    print(f"\nWarmup ({NUM_WARMUP} Batches, kompiliert)...")
    t0 = time.time()
    for i in range(NUM_WARMUP):
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits.view(-1, VOCAB_SIZE), target)
        loss.backward()
        optimizer.step()
    torch.mps.synchronize()
    print(f"  Dauer: {time.time()-t0:.2f}s (inkl. JIT-Kompilierung)")

    # Speed-Messung (ohne Profiler)
    print(f"\nReine Speed-Messung ({NUM_PROFILE} Batches, kein Profiler)...")
    t0 = time.time()
    for _ in range(NUM_PROFILE):
        optimizer.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits.view(-1, VOCAB_SIZE), target)
        loss.backward()
        optimizer.step()
    torch.mps.synchronize()
    elapsed = time.time() - t0
    print(f"  {NUM_PROFILE/elapsed:.2f} Batches/sec, {NUM_PROFILE*BATCH_SIZE*SEQ_LENGTH/elapsed:.0f} Tokens/sec")

    # Mit Profiler — auf MPS gibt es nur ProfilerActivity.CPU; der Dispatcher
    # sieht alle Ops und misst sie inklusive der Zeit die in MPS verbracht wird
    # (synchron mit den GPU-Calls).
    print(f"\nProfilieren ({NUM_PROFILE} Batches)...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        for _ in range(NUM_PROFILE):
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits.view(-1, VOCAB_SIZE), target)
            loss.backward()
            optimizer.step()
        torch.mps.synchronize()

    print("\n" + "=" * 70)
    print("TOP-OPS NACH SELBST-CPU-ZEIT (inkl. MPS-Sync)")
    print("=" * 70)
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20,
    ))

    print("\n" + "=" * 70)
    print("TOP-OPS NACH GESAMTZEIT (inkl. aufgerufene Subops)")
    print("=" * 70)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=15,
    ))


if __name__ == "__main__":
    main()
