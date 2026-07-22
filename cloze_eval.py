"""German cloze knowledge test — measures facts, not fluency.

The bpb comparison (compare_bpb.py) averages over every token, so it is
dominated by the many EASY tokens (grammar, word endings, common phrases)
— a model can win it on language form alone. Knowledge lives in a few
HARD tokens: after "Die Hauptstadt von Frankreich heißt", exactly one
token separates knowing from guessing. This eval scores ONLY those
answer tokens.

Each item in cloze_de.jsonl is {prefix, answer}; the answer starts with
a space so that no tokenizer merges it into the prefix. Two metrics:

  - answer-bpb: nats the model spends on the answer tokens, per answer
    byte. Tokenizer-independent (same trick as compare_bpb) — the fair
    headline number.
  - top-1 hit rate: is the FIRST answer token the model's argmax?
    Intuitive ("how often does greedy decoding start correctly"), but
    mildly tokenizer-dependent — quote it as color, not as the result.

Prediction registered before the first run (2026-07-22): our 540M wins
the fineweb bpb, but Qwen wins THIS eval despite its worse German —
knowledge transfers across languages, form does not.

OUTCOME (same day): the prediction was WRONG. Ours 0.5556 / 57% top-1,
Qwen 0.8602 / 37%. Two honest readings: (a) these are COMMON facts,
saturated in German Wikipedia — which our model read in full; the
long-tail (rare facts) might still flip it. (b) the answers are German
surface forms ("Warschau", not "Warsaw") — even a model that knows the
fact must also know its German name, so the test measures knowledge AND
its German binding, not knowledge alone. Kept as a lesson: register
predictions before measuring, and read your own eval's fine print.

Usage:
    python cloze_eval.py                    # our checkpoint + Qwen
    SKIP_HF=1 python cloze_eval.py          # ours only
    CHECKPOINT=... HF_MODEL=... CLOZE_FILE=... SHOW_SAMPLES=8
"""

import json
import math
import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import tiktoken
import torch
import torch.nn.functional as F

import train
from compare_bpb import HF_MODEL, SKIP_HF, SKIP_OURS, load_hf, load_ours

CLOZE_FILE = os.environ.get("CLOZE_FILE", "cloze_de.jsonl")
SHOW_SAMPLES = int(os.environ.get("SHOW_SAMPLES", "8"))


@torch.no_grad()
def score_cloze(name, encode, decode, logits_fn, device, items):
    total_nats, total_bytes, hits, rows = 0.0, 0, 0, []
    for it in items:
        ids_p = encode(it["prefix"])
        ids_f = encode(it["prefix"] + it["answer"])
        ans = ids_f[len(ids_p):]
        if not ans or ids_f[:len(ids_p)] != ids_p:
            # Tokenizer merged across the prefix/answer boundary — the
            # leading space in the answers prevents this in practice, but
            # a silent wrong split would corrupt the metric, so: skip loud.
            print(f"  ! boundary merge, skipping: {it['prefix']!r}")
            continue
        x = torch.tensor([ids_f], dtype=torch.long, device=device)
        # float() before log_softmax: summing many logs in fp16/bf16
        # accumulates rounding exactly where we measure hundredths.
        logp = F.log_softmax(logits_fn(x)[0].float(), dim=-1)
        # Answer token k sits at position len(ids_p)+k and is predicted
        # from the position before it.
        nats = -sum(logp[len(ids_p) + k - 1, t].item()
                    for k, t in enumerate(ans))
        n_bytes = len(it["answer"].encode("utf-8"))
        top_id = int(logp[len(ids_p) - 1].argmax())
        hit = top_id == ans[0]
        total_nats += nats
        total_bytes += n_bytes
        hits += hit
        rows.append((it, nats / (math.log(2) * n_bytes), hit,
                     decode([top_id])))

    n = len(rows)
    print(f"\n=== {name} ===")
    print(f"  answer-bpb: {total_nats / (math.log(2) * total_bytes):.4f} "
          f"bits/byte   top-1: {hits}/{n} ({100 * hits / n:.0f}%)")
    # The most instructive rows: where the model was most sure and most
    # lost. rows sorted by answer difficulty (bpb ascending).
    rows.sort(key=lambda r: r[1])
    show = rows[:SHOW_SAMPLES // 2] + rows[-SHOW_SAMPLES // 2:]
    for it, bpb, hit, top in show:
        mark = "✓" if hit else "✗"
        print(f"  {mark} {bpb:6.2f} bpb  {it['prefix']}[{it['answer']}]"
              f"  -> greedy: {top!r}")
    return {"answer_bpb": total_nats / (math.log(2) * total_bytes),
            "top1": hits / n, "n": n}


def main():
    with open(CLOZE_FILE) as f:
        items = [json.loads(line) for line in f]
    print(f"{len(items)} cloze items from {CLOZE_FILE}")
    device = train.get_device()
    results = {}

    if not SKIP_OURS:
        enc = tiktoken.get_encoding("gpt2")
        name, encode, _, model = load_ours(device)
        results[name] = score_cloze(name, encode, enc.decode, model,
                                    device, items)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    if not SKIP_HF:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(HF_MODEL)
        name, encode, _, logits_fn = load_hf(device)
        results[name] = score_cloze(name, encode, tok.decode, logits_fn,
                                    device, items)

    print("\n" + "=" * 64)
    print(f"{'model':<40} {'answer-bpb':>10} {'top-1':>7}")
    for name, r in results.items():
        print(f"{name:<40} {r['answer_bpb']:>10.4f} {r['top1']:>6.0%}")
    print("=" * 64)
    print("answer-bpb: bits per byte spent on the ANSWER tokens only — "
          "low = the model knew it. Fluency does not help here.")


if __name__ == "__main__":
    main()
