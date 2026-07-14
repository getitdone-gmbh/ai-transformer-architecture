"""SFT: Supervised Fine-Tuning of the base model on German instructions.

Turns the text-continuer into an answer-giver. The same training loop
as train.py (forward -> cross-entropy -> backward -> AdamW), only three
things differ — and each one has a reason:

1. DATA: instead of 10 billion raw web tokens, ~49k question-answer pairs
   (FreedomIntelligence/alpaca-gpt4-deutsch) in a fixed template:

       ### Frage:
       {frage}

       ### Antwort:
       {antwort}<|endoftext|>

   The <|endoftext|> is crucial: it is where the model learns to STOP,
   instead of hallucinating the next question.

2. LOSS MASKING: the question part gets label -100 (= CrossEntropyLoss
   ignores it). Only what the model gets wrong on the ANSWER is
   penalized — we want to train an answer-giver, not a
   question-asker. (Padding is masked the same way.)

3. SMALL LEARNING RATE (2e-5 instead of 2.5e-4): the base sits in a good
   valley of the loss landscape; SFT should shift the position within the
   valley, not leap up the valley wall and overwrite the 10-billion-token
   base (catastrophic forgetting).

Runs on the M4 Max (MPS) in a few hours. ENV overrides as everywhere.

    python sft.py                          # takes weights_540m_fp32.pt
    WEIGHTS=andere.pt EPOCHS=1 python sft.py
"""

import math
import os
import time

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset

from train import (
    GPTDecoder, configure_optimizer, get_amp_dtype, get_device, get_lr, _env,
)
from contextlib import nullcontext

WEIGHTS = _env("WEIGHTS", "weights_540m_fp32.pt", str)
OUT_PATH = _env("OUT_PATH", "sft_540m.pt", str)
DATASET = _env("DATASET", "FreedomIntelligence/alpaca-gpt4-deutsch", str)
                                       # multiple datasets: comma-separated
LIMIT = _env("LIMIT", 0, int)          # 0 = all examples; >0 for smoke tests
MAX_LEN = _env("MAX_LEN", 512, int)    # examples longer than this: DISCARDED
                                       # (not truncated — otherwise the answer
                                       # loses its <|endoftext|> and we would
                                       # train stopping mid-sentence)
BATCH_SIZE = _env("BATCH_SIZE", 4, int)     # keep small: the logits + CE
                                            # ([B*T, 50257] in fp32) are the
                                            # memory peak of every step
GRAD_ACCUM = _env("GRAD_ACCUM", 8, int)     # effectively 4*8 = 32 examples/step
EPOCHS = _env("EPOCHS", 2, int)             # up to ~4 epochs is fine (Muennighoff)
LEARNING_RATE = _env("LEARNING_RATE", 2e-5, float)
MIN_LR = _env("MIN_LR", 2e-6, float)
WARMUP_STEPS = _env("WARMUP_STEPS", 100, int)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 0.0, float)  # no decay: we want to preserve
                                                 # the base, not shrink it
GRAD_CLIP = _env("GRAD_CLIP", 1.0, float)
VAL_FRACTION = _env("VAL_FRACTION", 0.02, float)

# DO NOT TRANSLATE: this prompt template is baked into the trained model
# weights. The "### Frage" / "### Antwort" markers and the {frage} field
# name must stay exactly as-is, or the fine-tuned model breaks.
PROMPT_TMPL = "### Frage:\n{frage}\n\n### Antwort:\n"


class InstructDataset(Dataset):
    """Tokenized (input_ids, labels) pairs with a loss mask.

    labels is a copy of input_ids in which the question part and padding
    are set to -100. During training logits[t] is compared against
    labels[t+1] (next-token objective as always) — thanks to the -100,
    only the answer part contributes to the gradient.

    Examples are sorted by length; the collate pads only up to the
    longest example in the batch (instead of rigidly to MAX_LEN) —
    saving half the compute time for short questions.
    """

    def __init__(self, examples, enc, max_len):
        self.enc = enc
        self.max_len = max_len
        self.items = []
        skipped = too_long = 0
        for ex in examples:
            conv = ex["conversations"]
            # Only the first question-answer pair (the dataset is almost
            # exclusively single-turn; multi-turn we save for later).
            if len(conv) < 2 or conv[0]["from"] != "human":
                skipped += 1
                continue
            prompt = PROMPT_TMPL.format(frage=conv[0]["value"].strip())
            answer = conv[1]["value"].strip()
            p_ids = enc.encode(prompt, disallowed_special=())
            a_ids = enc.encode(answer, disallowed_special=()) + [enc.eot_token]
            ids = p_ids + a_ids
            if len(ids) > max_len:
                too_long += 1
                continue
            self.items.append((ids, len(p_ids)))
        self.items.sort(key=lambda it: len(it[0]))
        if skipped:
            print(f"  {skipped} examples skipped (unexpected format)")
        if too_long:
            print(f"  {too_long} examples discarded (longer than {max_len} tokens)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, pad_id):
    """Dynamic padding to the batch's longest sequence + label mask.

    The width is rounded up to multiples of 64: MPS/Metal caches kernels
    and buffers PER TENSOR SHAPE. Without buckets, every batch width
    (20, 21, 23, ...) creates its own cache entries — the memory pool
    fragments and grows over hours until OOM. With buckets there are
    only 8 shapes at MAX_LEN=512, which get reused constantly.
    """
    width = max(len(ids) for ids, _ in batch)
    width = (width + 63) // 64 * 64
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, (ids, n_prompt) in enumerate(batch):
        t = torch.tensor(ids, dtype=torch.long)
        input_ids[i, : len(ids)] = t
        # Answer part (incl. EOT) is the learning target; question + padding are -100.
        labels[i, n_prompt: len(ids)] = t[n_prompt:]
    return input_ids, labels


def sample_answer(model, enc, device, frage, max_new_tokens=60):
    """Generate a sample answer in the chat template (for progress tracking)."""
    prompt = PROMPT_TMPL.format(frage=frage)
    ids = torch.tensor([enc.encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.7,
                         top_p=0.9, repetition_penalty=1.2,
                         eos_token_id=enc.eot_token)
    text = enc.decode(out[0].cpu().tolist())
    answer = text[len(prompt):].replace("<|endoftext|>", "").strip()
    return answer


def main():
    device = get_device()
    amp_dtype = get_amp_dtype(device)
    enc = tiktoken.get_encoding("gpt2")
    print(f"Device: {device} | Autocast: {amp_dtype or 'fp32'}")

    # --- Load base (config travels along inside the weights file) ---
    blob = torch.load(WEIGHTS, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = GPTDecoder(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"], dropout=0.0,
    ).to(device)
    state = {k: v.float() if v.is_floating_point() else v
             for k, v in blob["model_state_dict"].items()}
    model.load_state_dict(state)
    n = sum(p.numel() for p in model.parameters())
    print(f"Base loaded: {WEIGHTS} ({n / 1e6:.0f}M parameters)")

    # --- Data (multiple datasets in the same conversations format are
    # simply concatenated; the length-sorting in the Dataset then mixes
    # them together anyway) ---
    from datasets import concatenate_datasets
    parts = [load_dataset(n.strip(), split="train")
             for n in DATASET.split(",") if n.strip()]
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    if LIMIT:
        ds = ds.select(range(min(LIMIT, len(ds))))
    print(f"Dataset: {DATASET} ({len(ds):,} examples)")
    full = InstructDataset(ds, enc, MAX_LEN)
    n_val = max(8, int(len(full) * VAL_FRACTION))
    # Val = take examples evenly distributed across the length-sorting,
    # so that both splits have the same length mixture.
    val_idx = set(range(0, len(full), max(1, len(full) // n_val)))
    train_items = [full.items[i] for i in range(len(full)) if i not in val_idx]
    val_items = [full.items[i] for i in val_idx]
    full.items = train_items
    val_ds = InstructDataset.__new__(InstructDataset)
    val_ds.items, val_ds.enc, val_ds.max_len = val_items, enc, MAX_LEN

    pad_id = enc.eot_token
    coll = lambda b: collate(b, pad_id)
    # shuffle=True does NOT mix the length-sorted examples at the batch
    # level — instead we shuffle the BATCH order ourselves, so that
    # batches stay length-homogeneous (little padding) while the order
    # is random.
    train_batches = [full.items[i:i + BATCH_SIZE]
                     for i in range(0, len(full.items), BATCH_SIZE)]
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=coll)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = configure_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_batches) / GRAD_ACCUM)
    max_steps = EPOCHS * steps_per_epoch
    print(f"Training: {len(full.items):,} train / {len(val_ds.items):,} val | "
          f"eff. batch {BATCH_SIZE * GRAD_ACCUM} | max_steps={max_steps}\n")

    amp_ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype else nullcontext())
    test_fragen = ["Was ist die Hauptstadt von Frankreich?",
                   "Erkläre kurz, was ein Vulkan ist."]

    def run_loss(input_ids, labels):
        logits = model(input_ids)
        # Shift: logits at position t predict token t+1.
        return criterion(logits[:, :-1].reshape(-1, cfg["vocab_size"]),
                         labels[:, 1:].reshape(-1))

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(train_batches)).tolist()
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        for bi, batch_idx in enumerate(order):
            input_ids, labels = coll(train_batches[batch_idx])
            input_ids, labels = input_ids.to(device), labels.to(device)
            with amp_ctx:
                loss = run_loss(input_ids, labels)
            (loss / GRAD_ACCUM).backward()

            if (bi + 1) % GRAD_ACCUM == 0 or bi == len(order) - 1:
                lr = get_lr(global_step, WARMUP_STEPS, max_steps,
                            LEARNING_RATE, MIN_LR)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                # MPS never releases cached buffers on its own; without
                # this the pool grows monotonically over hours (the
                # overnight OOM). Costs ~ms, saves the training.
                if device.type == "mps" and global_step % 25 == 0:
                    torch.mps.empty_cache()
                if global_step % 20 == 0:
                    rate = (bi + 1) * BATCH_SIZE / (time.time() - t0)
                    print(f"  Step {global_step}/{max_steps}, "
                          f"loss={loss.item():.4f}, lr={lr:.2e}, "
                          f"{rate:.1f} examples/s")

        # --- Val + sample answers per epoch ---
        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                with amp_ctx:
                    vloss += run_loss(input_ids, labels).item()
                nb += 1
        print(f"\nEpoch {epoch + 1}: val_loss={vloss / max(1, nb):.4f} "
              f"(only answer tokens measured)")
        for frage in test_fragen:
            print(f"  Q: {frage}")
            print(f"  A: {sample_answer(model, enc, device, frage)}\n")

        torch.save({
            "config": cfg,
            "arch_version": blob.get("arch_version"),
            "model_state_dict": model.state_dict(),
            "sft_dataset": DATASET,
            "sft_epochs": epoch + 1,
        }, OUT_PATH)
        print(f"Saved: {OUT_PATH}\n{'-' * 60}")


if __name__ == "__main__":
    main()
