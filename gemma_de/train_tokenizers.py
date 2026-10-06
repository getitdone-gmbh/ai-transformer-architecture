"""Phase 1, step 2: build the two candidate tokenizers for a German Gemma.

VARIANT A — REPLACE: a fresh byte-level BPE on the mix from build_corpus.py
(70% German, 20% Python, 10% English), 64k and 128k vocab. Every token ID
is new, so a transplant must re-initialize the whole embedding matrix. In
exchange, the vocabulary spends its budget on OUR languages and the
embedding shrinks (Gemma: 262k rows).

VARIANT B — EXTEND: keep all 262,144 Gemma tokens and their merge order,
and APPEND German merges at the end. BPE applies merges in rank order, so
appended merges only fire once no Gemma merge applies anymore — they
can glue Gemma's pieces into bigger German tokens but never split
anything. Consequence by construction: no text gets MORE tokens, code
and English stay as they are, and only the new rows need training.

How B is trained — "continued BPE" with a trick. A BPE trainer starts
from single characters. We want it to start from GEMMA'S tokens instead.
So: tokenize every German word with Gemma, map each distinct Gemma token
to one private-use Unicode character, and hand the trainer these
"words of Gemma-token characters". Its merges are then merges of Gemma
tokens; mapping the characters back gives merges in Gemma's vocabulary.
One run with the largest size gives all smaller sizes too: BPE merges are
ordered, so the first 16k merges of a 64k run ARE the 16k tokenizer.

Output (WORK_DIR): A_64k.json, A_128k.json, B_16k.json, B_32k.json,
B_64k.json — all loadable with tokenizers.Tokenizer.from_file and
measurable with eval/tokenizer_efficiency.py.

Usage:
    python gemma_de/train_tokenizers.py           # both variants
    ONLY=B python gemma_de/train_tokenizers.py
"""

import json
import os
import re
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
from tokenizers import (Regex, Tokenizer, decoders, models,  # noqa: E402
                        pre_tokenizers, trainers)

WORK_DIR = os.path.expanduser(os.environ.get("WORK_DIR", "~/gemma_de_work"))
GEMMA = os.environ.get("GEMMA", "google/gemma-4-31B-it")
ONLY = os.environ.get("ONLY", "AB")
A_VOCABS = [65536, 131072]
B_MERGES = [16384, 32768, 65536]
MIN_WORD_FREQ = 2   # B: words seen once can't teach a merge worth having
WORD = re.compile(r" ?[^\W\d_]+| ?[^\w\s]+")   # B: letter runs | punctuation

# Pre-tokenization for A: GPT-4-style split, but digits one by one (like
# Gemma and Kolibri — arithmetic sees every digit) and runs of spaces kept
# together (Python indentation becomes a few tokens, not dozens).
SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
         r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def docs(name):
    with open(os.path.join(WORK_DIR, f"{name}.txt"), encoding="utf-8") as f:
        buf = ""
        while chunk := f.read(1 << 24):
            parts = (buf + chunk).split("\0")
            buf = parts.pop()
            yield from parts
        if buf:
            yield buf


def gemma_specials(gemma):
    return [t["content"] for t in json.loads(gemma.to_str())["added_tokens"]]


def train_a(gemma):
    for vocab in A_VOCABS:
        tok = Tokenizer(models.BPE(byte_fallback=False))
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tok.decoder = decoders.ByteLevel()

        def mix():
            for name in ("de", "code", "en"):
                yield from docs(name)

        # Gemma's special tokens come along so the chat template keeps
        # working after a transplant (they get IDs 0..23 here).
        tok.train_from_iterator(mix(), trainers.BpeTrainer(
            vocab_size=vocab, special_tokens=gemma_specials(gemma),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False))
        out = os.path.join(WORK_DIR, f"A_{vocab // 1024}k.json")
        tok.save(out)
        print(f"saved {out} ({tok.get_vocab_size()} tokens)", flush=True)


def private_use_chars():
    """Code points the trainer can treat as opaque symbols (planes 15+16)."""
    for base in (0xF0000, 0x100000):
        for cp in range(base, base + 0xFFFE):
            yield chr(cp)


def train_b(gemma):
    # 1. German word counts, in Gemma's own surface form: a word with its
    #    leading space, exactly as Gemma sees it after replacing ' ' by '▁'.
    #    Letters and punctuation are counted separately — otherwise BPE
    #    learns '▁Paris.' as its own token, and 'Paris' gets two unrelated
    #    embeddings depending on whether a period follows.
    words = Counter()
    for text in docs("de"):
        words.update(WORD.findall(text))
    words = {w: c for w, c in words.items() if c >= MIN_WORD_FREQ}
    print(f"B: {len(words):,} distinct German words (freq >= "
          f"{MIN_WORD_FREQ})", flush=True)

    # 2. Gemma-tokenize every word, map each Gemma token to one symbol.
    wlist = list(words)
    enc = gemma.encode_batch(wlist, add_special_tokens=False)
    sym_of, tok_of = {}, {}
    pool = private_use_chars()
    mapped = Counter()
    byte_fallback = 0
    for w, e in zip(wlist, enc):
        if any(t.startswith("<0x") for t in e.tokens):
            byte_fallback += 1    # raw-byte pieces don't concatenate as text
            continue
        if len(e.tokens) < 2:
            continue              # already a single Gemma token: nothing to win
        s = []
        for t in e.tokens:
            if t not in sym_of:
                sym_of[t] = next(pool)
                tok_of[sym_of[t]] = t
            s.append(sym_of[t])
        mapped["".join(s)] += words[w]
    print(f"B: {len(mapped):,} multi-token words over {len(sym_of):,} "
          f"distinct Gemma tokens ({byte_fallback} byte-fallback words "
          "skipped)", flush=True)

    # 3. Ordinary BPE over the symbol strings. Frequencies are fed by
    #    repetition (the trainer has no weighted-input API); chunks keep
    #    single strings small.
    def weighted():
        for s, c in mapped.items():
            while c > 0:
                n = min(c, 4096)
                yield (s + " ") * n
                c -= n

    tr = Tokenizer(models.BPE())
    tr.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tr.train_from_iterator(weighted(), trainers.BpeTrainer(
        vocab_size=len(sym_of) + max(B_MERGES), show_progress=False))
    sym_merges = json.loads(tr.to_str())["model"]["merges"]
    print(f"B: trained {len(sym_merges):,} merges", flush=True)

    def text(s):
        return "".join(tok_of[c] for c in s)

    # 4. Append to Gemma: new merges at the END (lowest priority), new
    #    vocab entries after the last Gemma ID.
    base = json.loads(gemma.to_str())
    have_merges = {tuple(m) for m in base["model"]["merges"]}
    for n in B_MERGES:
        d = json.loads(gemma.to_str())
        vocab, merges = d["model"]["vocab"], d["model"]["merges"]
        next_id = max(max(vocab.values()),
                      max(t["id"] for t in d["added_tokens"])) + 1
        added = 0
        for m in sym_merges[:n]:
            a, b = (m if isinstance(m, list) else m.split(" "))
            a, b = text(a), text(b)
            if (a, b) in have_merges:
                continue
            merges.append([a, b])
            if a + b not in vocab:
                vocab[a + b] = next_id
                next_id += 1
                added += 1
        out = os.path.join(WORK_DIR, f"B_{n // 1024}k.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        print(f"saved {out} (+{added:,} tokens, vocab {next_id:,})",
              flush=True)


def main():
    gemma = Tokenizer.from_pretrained(GEMMA)
    if "B" in ONLY:
        train_b(gemma)
    if "A" in ONLY:
        train_a(gemma)


if __name__ == "__main__":
    main()
