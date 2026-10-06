"""What does German cost? Tokens per text across tokenizers.

THE QUESTION: our de_bpe packs ~1.7x more German into a token than GPT-2.
But GPT-2 is a straw man — modern models (Gemma, Qwen, Kolibri) ship
vocabularies of 150k-260k entries that already cover German. Before we
invest in a German tokenizer for a big model (tokenizer transplantation),
measure whether there is anything left to win.

WHY IT MATTERS: every token costs compute (one forward step per generated
token), context window (2048 tokens hold less German text if German
shreds) and money (APIs bill per token). Fewer tokens for the same text =
cheaper, faster and longer context for free.

METRICS, per tokenizer and eval file:
  chars/token   human-readable ("how much text is in a token")
  bytes/token   UTF-8 bytes; what Aleph Alpha reports. Umlauts and ß are
                2 bytes, so German bytes/token > chars/token.
  tokens rel.   token count relative to de_bpe on the same text
                (1.30 = needs 30% more tokens for the same German)

FAIRNESS: same raw documents for all, encoded one document at a time,
no special tokens (BOS/EOS) — we measure the text, not the wrapper.

CONTAMINATION CAVEAT: the checked-in data/de_bpe.json was LEARNED on
eval_data/fineweb.jsonl (see its .meta.json). On that file de_bpe is
graded on its own training text and looks too good — the honest held-out
number is fineweb_hq.jsonl (a different FineWeb2-HQ file). The script
reads the meta file and flags the overlap.

VOCAB SIZE IS NOT FREE: a bigger vocab compresses better but costs
embedding + LM-head parameters (vocab x d_model, twice if untied) and
a softmax over more entries. Printed alongside so the comparison stays
honest.

Usage:
    python eval/tokenizer_efficiency.py              # all tokenizers
    TOKENIZERS=gpt2,de_bpe,google/gemma-4-31B-it python eval/tokenizer_efficiency.py
    TOKENIZERS=google/gemma-4-31B-it,~/gemma_de_work/B_32k.json ...  # local files

Needs: tokenizers, tiktoken, huggingface_hub (only tokenizer.json is
downloaded, never the weights).
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from paths import data_path  # noqa: E402
from tokenizer import get_tokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EVAL_FILES = ["fineweb_hq.jsonl", "fineweb.jsonl", "code.jsonl",
              "fineweb_en.jsonl"]   # en: written by gemma_de/build_corpus.py

# Ungated repos only (meta-llama is gated -> unsloth's mirror, same file).
DEFAULT = [
    "gpt2",
    "de_bpe",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.5-27B",
    "Aleph-Alpha/Kolibri-1",
    "unsloth/Llama-3.1-8B",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]


def load(name):
    """Return (count_tokens(text) -> int, vocab_size)."""
    if name in ("gpt2", "de_bpe"):
        tok = get_tokenizer(name)
        return (lambda t: len(tok.encode_ordinary(t))), tok.n_vocab
    from tokenizers import Tokenizer
    tok = (Tokenizer.from_file(os.path.expanduser(name))
           if name.endswith(".json") else Tokenizer.from_pretrained(name))
    return ((lambda t: len(tok.encode(t, add_special_tokens=False).ids)),
            tok.get_vocab_size(with_added_tokens=True))


def read_docs(fname):
    with open(os.path.join(ROOT, "eval_data", fname), encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def de_bpe_trained_on():
    try:
        with open(data_path("de_bpe.json.meta.json")) as f:
            return json.load(f).get("trained_on", "")
    except OSError:
        return ""


def main():
    names = os.environ.get("TOKENIZERS")
    names = names.split(",") if names else DEFAULT
    corpora = {f: read_docs(f) for f in EVAL_FILES
               if os.path.exists(os.path.join(ROOT, "eval_data", f))}
    sizes = {f: (sum(len(d) for d in docs),
                 sum(len(d.encode("utf-8")) for d in docs))
             for f, docs in corpora.items()}

    results = {}  # name -> {file: tokens}, vocab
    for name in names:
        try:
            count, vocab = load(name)
        except Exception as e:  # gated, missing tokenizer.json, offline
            print(f"skip {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        results[name] = (vocab, {f: sum(count(d) for d in docs)
                                 for f, docs in corpora.items()})
        print(f"done {name}", file=sys.stderr)

    leaked = de_bpe_trained_on()
    for f in corpora:
        chars, nbytes = sizes[f]
        ref = results.get("de_bpe", (None, {}))[1].get(f)
        print(f"\n== {f}  ({len(corpora[f])} docs, {chars:,} chars, "
              f"{nbytes:,} bytes)")
        print(f"{'tokenizer':44s} {'vocab':>8s} {'tokens':>10s} "
              f"{'chars/tok':>9s} {'bytes/tok':>9s} {'vs de_bpe':>9s}")
        rows = sorted(results.items(), key=lambda kv: kv[1][1][f])
        for name, (vocab, toks) in rows:
            name = os.path.basename(name) if name.endswith(".json") else name
            n = toks[f]
            rel = f"{n / ref:.2f}x" if ref else "-"
            flag = " *" if name == "de_bpe" and f in leaked else ""
            print(f"{name + flag:44s} {vocab:8,d} {n:10,d} "
                  f"{chars / n:9.2f} {nbytes / n:9.2f} {rel:>9s}")
    if leaked:
        print(f"\n* de_bpe was trained on {leaked} — not held-out there.")


if __name__ == "__main__":
    main()
