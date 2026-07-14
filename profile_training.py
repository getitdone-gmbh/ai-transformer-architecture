"""Short profiling run to find the slowest ops in training.

Builds the model exactly as in training, runs 5 warmup batches (absorbs
the torch.compile JIT), then 20 profiled batches. Prints the top ops
sorted by self-consumed MPS time.

Runs in ~30-60 seconds and has no effect on a training run happening in
parallel (separate process).
"""

import time

import torch
import torch.nn as nn

from train import (
    GPTDecoder,
    configure_optimizer,
    D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, SEQ_LENGTH, LEARNING_RATE, WEIGHT_DECAY,
)

VOCAB_SIZE = 50257
NUM_WARMUP = 5
NUM_PROFILE = 20


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = GPTDecoder(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, DROPOUT).to(device)
    model = torch.compile(model)
    optimizer = configure_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Simulate inputs (causal masking happens inside the SDPA kernel itself)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), device=device)
    target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SEQ_LENGTH,), device=device)

    # Warmup — lets torch.compile do its JIT compilation
    print(f"\nWarmup ({NUM_WARMUP} batches, compiled)...")
    t0 = time.time()
    for i in range(NUM_WARMUP):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), target)
        loss.backward()
        optimizer.step()
    torch.mps.synchronize()
    print(f"  Duration: {time.time()-t0:.2f}s (incl. JIT compilation)")

    # Speed measurement (without the profiler)
    print(f"\nPure speed measurement ({NUM_PROFILE} batches, no profiler)...")
    t0 = time.time()
    for _ in range(NUM_PROFILE):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), target)
        loss.backward()
        optimizer.step()
    torch.mps.synchronize()
    elapsed = time.time() - t0
    print(f"  {NUM_PROFILE/elapsed:.2f} batches/sec, {NUM_PROFILE*BATCH_SIZE*SEQ_LENGTH/elapsed:.0f} tokens/sec")

    # With the profiler — on MPS only ProfilerActivity.CPU exists; the
    # dispatcher sees all ops and measures them including the time spent in
    # MPS (synchronous with the GPU calls).
    print(f"\nProfiling ({NUM_PROFILE} batches)...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        for _ in range(NUM_PROFILE):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), target)
            loss.backward()
            optimizer.step()
        torch.mps.synchronize()

    print("\n" + "=" * 70)
    print("TOP OPS BY SELF CPU TIME (incl. MPS sync)")
    print("=" * 70)
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=20,
    ))

    print("\n" + "=" * 70)
    print("TOP OPS BY TOTAL TIME (incl. called sub-ops)")
    print("=" * 70)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=15,
    ))


if __name__ == "__main__":
    main()
