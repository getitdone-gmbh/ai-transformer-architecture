"""Bits per byte of a (transplanted) Gemma on our eval axes.

The yardstick for every phase-2 step. A new tokenizer changes the token
COUNT, so loss per token is meaningless across tokenizers (bigger tokens
are harder to predict). Bits per byte normalizes to the raw text, exactly
as eval/compare_bpb.py does for our own models:

    bpb = sum_of_loss_in_nats / (ln(2) * utf8_bytes)

The goal of the transplant in one line: same bpb as the original Gemma
(nothing lost) with ~18% fewer tokens (the win).

Scoring follows Gemma's own setup: every document starts with <bos>,
documents longer than the window are scored in windows overlapping by one
token, so every token after <bos> is predicted exactly once. With an
extended tokenizer a text has several possible tokenizations; we score the
canonical one, which makes bpb an upper bound — fair to the transplant,
never flattering.

Usage:
    MODEL=google/gemma-4-E2B python gemma_de/bpb.py
    MODEL=~/gemma_de_work/models/E2B_B64k_init python gemma_de/bpb.py
    MAX_DOCS=50 ...   # quick check

The tokenizer is read from MODEL (its tokenizer.json), so a transplanted
model always gets measured with the tokenizer it was built for.
"""

import json
import math
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FILES = {"de": "fineweb_hq.jsonl", "code": "code.jsonl", "en": "fineweb_en.jsonl"}
WINDOW = int(os.environ.get("WINDOW", 2048))
MAX_DOCS = int(os.environ.get("MAX_DOCS", 0))


def resolve(name):
    path = os.path.expanduser(name)
    return path if os.path.isdir(path) else name


def load(name, dtype=torch.bfloat16):
    name = resolve(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype,
                                                 device_map="cuda")
    model.eval()
    if os.path.isdir(name):
        tok = Tokenizer.from_file(os.path.join(name, "tokenizer.json"))
    else:
        tok = Tokenizer.from_pretrained(name)
    return model, tok


def read_docs(fname):
    with open(os.path.join(ROOT, "eval_data", fname), encoding="utf-8") as f:
        docs = [json.loads(line)["text"] for line in f]
    return docs[:MAX_DOCS] if MAX_DOCS else docs


@torch.inference_mode()
def doc_nats(model, ids):
    """Sum of -ln p over every token after <bos>."""
    total, start = 0.0, 0
    while start < len(ids) - 1:
        chunk = torch.tensor([ids[start:start + WINDOW]], device="cuda")
        logits = model(input_ids=chunk).logits[0, :-1].float()
        total += F.cross_entropy(logits, chunk[0, 1:], reduction="sum").item()
        start += WINDOW - 1
    return total


def measure(model, tok, label=""):
    bos = tok.token_to_id("<bos>")
    out = {}
    for axis, fname in FILES.items():
        docs = read_docs(fname)
        nats = nbytes = ntok = 0
        t0 = time.time()
        for text in docs:
            ids = [bos] + tok.encode(text, add_special_tokens=False).ids
            nats += doc_nats(model, ids)
            nbytes += len(text.encode("utf-8"))
            ntok += len(ids) - 1
        out[axis] = dict(bpb=nats / (math.log(2) * nbytes), tokens=ntok,
                         nats_per_token=nats / ntok)
        print(f"{label:28s} {axis:5s} bpb {out[axis]['bpb']:.4f}  "
              f"tokens {ntok:9,d}  nats/tok {nats / ntok:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    name = os.environ.get("MODEL", "google/gemma-4-E2B")
    model, tok = load(name)
    res = measure(model, tok, os.path.basename(name.rstrip("/")))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump({"model": name, **res}, f, indent=2)
