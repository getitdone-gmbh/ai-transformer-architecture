"""Phase 2, step 1: give Gemma the extended tokenizer (variant B).

WHAT CHANGES: only the vocabulary. Gemma's 262,144 rows stay untouched;
B_64k appends 65,536 German tokens, so every embedding table gets 65,536
new rows. In Gemma 4 that is TWO tables:
  embed_tokens            (vocab x hidden)       — also the LM head (tied)
  embed_tokens_per_layer  (vocab x layers*256)   — E2B/E4B only: a small
                          extra vector per token for EVERY layer ("PLE")

HOW THE NEW ROWS START — the whole trick. A random row would be noise
the layers have never seen. But we know exactly what each new token IS:
'▁Hauptstadt' was learned as the merge of Gemma pieces, e.g.
'▁Haupt' + 'stadt'. Its new row starts as the MEAN of those pieces' rows.
The layers then see something close to what they saw before, just in one
position instead of two. The recursion follows the merge list: a new
token made of a new token and an old one expands to all its original
Gemma pieces.

What this cannot fix by itself: the tied LM head. The output score for
'▁Hauptstadt' becomes h · mean(rows) — a guess, not a learned value.
bpb.py measures how much that costs; step 2 (training only the new rows)
is there to repair it.

BASE, NOT -it: the instruct models are trained on the chat format and
score raw text badly (E2B-it: 4.3 nats/token on German web text, base:
2.05). A tokenizer transplant is a pretraining-style repair, so it starts
from the base model; instruction tuning comes after, on top.

Usage:
    MODEL=google/gemma-4-E2B TOKENIZER=~/gemma_de_work/B_64k.json \\
    OUT=~/gemma_de_work/models/E2B_B64k_init python gemma_de/transplant.py
"""

import fnmatch
import json
import os
import shutil

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

MODEL = os.environ.get("MODEL", "google/gemma-4-E2B")
TOKENIZER = os.path.expanduser(
    os.environ.get("TOKENIZER", "~/gemma_de_work/B_64k.json"))
OUT = os.path.expanduser(
    os.environ.get("OUT", "~/gemma_de_work/models/E2B_B64k_init"))

KEEP = ["tokenizer*", "*.jinja", "processor_config.json", "chat_template*"]


def components(tok_json, n_old):
    """New token id -> list of original Gemma ids it is made of."""
    vocab = tok_json["model"]["vocab"]
    comps = {}
    for a, b in tok_json["model"]["merges"]:
        new = vocab[a + b]
        if new < n_old or new in comps:
            continue
        comps[new] = ([vocab[a]] if vocab[a] < n_old else comps[vocab[a]]) + \
                     ([vocab[b]] if vocab[b] < n_old else comps[vocab[b]])
    return comps


@torch.no_grad()
def main():
    with open(TOKENIZER, encoding="utf-8") as f:
        tok_json = json.load(f)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    tables = [model.get_input_embeddings()]
    if hasattr(model, "get_per_layer_input_embeddings"):
        tables.append(model.get_per_layer_input_embeddings())
    n_old = tables[0].weight.shape[0]
    n_new = max(tok_json["model"]["vocab"].values()) + 1
    comps = components(tok_json, n_old)
    assert len(comps) == n_new - n_old, (len(comps), n_new - n_old)

    model.resize_token_embeddings(n_new, mean_resizing=False)
    tables = [model.get_input_embeddings()]
    if hasattr(model, "get_per_layer_input_embeddings"):
        tables.append(model.get_per_layer_input_embeddings())
    for table in tables:
        w = table.weight
        for new, parts in comps.items():
            w[new] = w[parts].float().mean(0).to(w.dtype)
        print(f"init {tuple(w.shape)}: {len(comps):,} rows from "
              f"{sum(map(len, comps.values())) / len(comps):.2f} pieces avg")
    # Tied head must follow the resized input table.
    assert model.lm_head.weight.data_ptr() == tables[0].weight.data_ptr()

    model.save_pretrained(OUT)
    # Tokenizer files: Gemma's (chat template, special tokens) + our vocab.
    # Filter by name ourselves: the snapshot folder also holds the cached
    # ORIGINAL weights and config, which must not overwrite ours.
    src = snapshot_download(MODEL, allow_patterns=KEEP)
    for fname in os.listdir(src):
        if any(fnmatch.fnmatch(fname, pat) for pat in KEEP):
            shutil.copyfile(os.path.join(src, fname), os.path.join(OUT, fname))
    shutil.copyfile(TOKENIZER, os.path.join(OUT, "tokenizer.json"))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
