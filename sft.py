"""SFT: Supervised Fine-Tuning des Basismodells auf deutsche Instruktionen.

Macht aus dem Textfortsetzer einen Antwortgeber. Derselbe Trainings-
Kreislauf wie train.py (Forward -> CrossEntropy -> Backward -> AdamW),
nur drei Dinge sind anders — und jedes hat einen Grund:

1. DATEN: statt 10 Mrd. roher Web-Tokens ~49k Frage-Antwort-Paare
   (FreedomIntelligence/alpaca-gpt4-deutsch) im festen Template:

       ### Frage:
       {frage}

       ### Antwort:
       {antwort}<|endoftext|>

   Das <|endoftext|> ist entscheidend: daran lernt das Modell AUFZUHOEREN,
   statt die naechste Frage zu halluzinieren.

2. LOSS-MASKIERUNG: der Frage-Teil bekommt Label -100 (= CrossEntropyLoss
   ignoriert ihn). Bestraft wird nur, was das Modell bei der ANTWORT
   falsch macht — wir wollen einen Antwortgeber trainieren, keinen
   Fragensteller. (Padding wird genauso maskiert.)

3. KLEINE LERNRATE (2e-5 statt 2.5e-4): die Basis sitzt in einem guten
   Tal der Loss-Landschaft; SFT soll die Position im Tal verschieben,
   nicht die Talwand hochspringen und die 10-Mrd.-Token-Basis
   ueberschreiben (Catastrophic Forgetting).

Laeuft auf dem M4 Max (MPS) in wenigen Stunden. ENV-Overrides wie ueberall.

    python sft.py                          # nimmt weights_540m_fp32.pt
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
LIMIT = _env("LIMIT", 0, int)          # 0 = alle Beispiele; >0 fuer Smoke-Tests
MAX_LEN = _env("MAX_LEN", 512, int)    # Beispiele laenger als das: abgeschnitten
BATCH_SIZE = _env("BATCH_SIZE", 8, int)
GRAD_ACCUM = _env("GRAD_ACCUM", 4, int)     # effektiv 8*4 = 32 Beispiele/Step
EPOCHS = _env("EPOCHS", 2, int)             # bis ~4 Epochen ok (Muennighoff)
LEARNING_RATE = _env("LEARNING_RATE", 2e-5, float)
MIN_LR = _env("MIN_LR", 2e-6, float)
WARMUP_STEPS = _env("WARMUP_STEPS", 100, int)
WEIGHT_DECAY = _env("WEIGHT_DECAY", 0.0, float)  # kein Decay: wir wollen die
                                                 # Basis erhalten, nicht schrumpfen
GRAD_CLIP = _env("GRAD_CLIP", 1.0, float)
VAL_FRACTION = _env("VAL_FRACTION", 0.02, float)

PROMPT_TMPL = "### Frage:\n{frage}\n\n### Antwort:\n"


class InstructDataset(Dataset):
    """Tokenisierte (input_ids, labels)-Paare mit Loss-Maske.

    labels ist eine Kopie von input_ids, in der Frage-Teil und Padding
    auf -100 stehen. Beim Training werden logits[t] gegen labels[t+1]
    verglichen (naechstes-Token-Ziel wie immer) — durch die -100 traegt
    nur der Antwort-Teil zum Gradienten bei.

    Beispiele sind nach Laenge sortiert; der Collate padded nur bis zum
    laengsten Beispiel des Batches (statt stur auf MAX_LEN) — spart bei
    kurzen Fragen die Haelfte der Rechenzeit.
    """

    def __init__(self, examples, enc, max_len):
        self.enc = enc
        self.max_len = max_len
        self.items = []
        skipped = 0
        for ex in examples:
            conv = ex["conversations"]
            # Nur das erste Frage-Antwort-Paar (Datensatz ist fast
            # ausschliesslich single-turn; Multi-Turn heben wir uns auf).
            if len(conv) < 2 or conv[0]["from"] != "human":
                skipped += 1
                continue
            prompt = PROMPT_TMPL.format(frage=conv[0]["value"].strip())
            answer = conv[1]["value"].strip()
            p_ids = enc.encode(prompt)
            a_ids = enc.encode(answer) + [enc.eot_token]
            ids = (p_ids + a_ids)[:max_len]
            n_prompt = min(len(p_ids), max_len)
            self.items.append((ids, n_prompt))
        self.items.sort(key=lambda it: len(it[0]))
        if skipped:
            print(f"  {skipped} Beispiele uebersprungen (unerwartetes Format)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, pad_id):
    """Dynamisches Padding auf die Batch-laengste Sequenz + Label-Maske."""
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, (ids, n_prompt) in enumerate(batch):
        t = torch.tensor(ids, dtype=torch.long)
        input_ids[i, : len(ids)] = t
        # Antwort-Teil (inkl. EOT) ist Lernziel; Frage + Padding sind -100.
        labels[i, n_prompt: len(ids)] = t[n_prompt:]
    return input_ids, labels


def sample_answer(model, enc, device, frage, max_new_tokens=60):
    """Eine Probe-Antwort im Chat-Template generieren (fuer den Fortschritt)."""
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

    # --- Basis laden (config reist in der Gewichte-Datei mit) ---
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
    print(f"Basis geladen: {WEIGHTS} ({n / 1e6:.0f}M Parameter)")

    # --- Daten ---
    ds = load_dataset(DATASET, split="train")
    if LIMIT:
        ds = ds.select(range(min(LIMIT, len(ds))))
    print(f"Datensatz: {DATASET} ({len(ds):,} Beispiele)")
    full = InstructDataset(ds, enc, MAX_LEN)
    n_val = max(8, int(len(full) * VAL_FRACTION))
    # Val = gleichmaessig ueber die Laengen-Sortierung verteilt entnehmen,
    # damit beide Splits die gleiche Laengen-Mischung haben.
    val_idx = set(range(0, len(full), max(1, len(full) // n_val)))
    train_items = [full.items[i] for i in range(len(full)) if i not in val_idx]
    val_items = [full.items[i] for i in val_idx]
    full.items = train_items
    val_ds = InstructDataset.__new__(InstructDataset)
    val_ds.items, val_ds.enc, val_ds.max_len = val_items, enc, MAX_LEN

    pad_id = enc.eot_token
    coll = lambda b: collate(b, pad_id)
    # shuffle=True mischt die laengen-sortierten Beispiele auf Batch-Ebene
    # NICHT — wir shuffeln stattdessen die BATCH-Reihenfolge selbst, damit
    # Batches laengen-homogen bleiben (wenig Padding), aber die Reihenfolge
    # zufaellig ist.
    train_batches = [full.items[i:i + BATCH_SIZE]
                     for i in range(0, len(full.items), BATCH_SIZE)]
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=coll)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = configure_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_batches) / GRAD_ACCUM)
    max_steps = EPOCHS * steps_per_epoch
    print(f"Training: {len(full.items):,} train / {len(val_ds.items):,} val | "
          f"eff. Batch {BATCH_SIZE * GRAD_ACCUM} | max_steps={max_steps}\n")

    amp_ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype else nullcontext())
    test_fragen = ["Was ist die Hauptstadt von Frankreich?",
                   "Erkläre kurz, was ein Vulkan ist."]

    def run_loss(input_ids, labels):
        logits = model(input_ids)
        # Shift: logits an Position t sagen Token t+1 vorher.
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
                if global_step % 20 == 0:
                    rate = (bi + 1) * BATCH_SIZE / (time.time() - t0)
                    print(f"  Step {global_step}/{max_steps}, "
                          f"loss={loss.item():.4f}, lr={lr:.2e}, "
                          f"{rate:.1f} Beispiele/s")

        # --- Val + Probe-Antworten pro Epoche ---
        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                with amp_ctx:
                    vloss += run_loss(input_ids, labels).item()
                nb += 1
        print(f"\nEpoche {epoch + 1}: val_loss={vloss / max(1, nb):.4f} "
              f"(nur Antwort-Tokens gemessen)")
        for frage in test_fragen:
            print(f"  F: {frage}")
            print(f"  A: {sample_answer(model, enc, device, frage)}\n")

        torch.save({
            "config": cfg,
            "arch_version": blob.get("arch_version"),
            "model_state_dict": model.state_dict(),
            "sft_dataset": DATASET,
            "sft_epochs": epoch + 1,
        }, OUT_PATH)
        print(f"Gespeichert: {OUT_PATH}\n{'-' * 60}")


if __name__ == "__main__":
    main()
