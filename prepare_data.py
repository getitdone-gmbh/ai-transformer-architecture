"""Streaming data preparation: sources -> tokenized uint16 shards.

For the large run (350M @ ~10 billion tokens) the old approach
(download everything -> one token tensor in RAM) no longer works:
  - Raw data (~40-60 GB of text) should never sit fully on disk.
  - 10 billion tokens fit in no RAM if you build them as a list.

Hence the standard approach (nanoGPT / GPT-2 replications):
  1. STREAM the sources (HuggingFace streaming=True loads in chunks),
  2. tokenize on the fly (tiktoken, multi-threaded),
  3. write as flat uint16 binary shards (~100M tokens = ~200 MB each).

Why uint16: the GPT-2 vocab has 50,257 IDs < 65,536 -> 2 bytes/token instead
of 4. 10 billion tokens = ~20 GB on disk. The training later reads the shards
via np.memmap — so it NEVER loads everything into RAM (see ShardDataset in
train.py).

New compared to the old pipeline: the <|endoftext|> token sits between
documents. Without a separator, articles stick together seamlessly and the
model never learns that a context reset exists.

Configuration via ENV (all optional):
  FINEWEB_TOKENS=8e9   token budget from FineWeb2-German (0 = source off)
  WIKI_TOKENS=2e9      token budget from Wikipedia-DE    (0 = source off)
  SHARD_TOKENS=1e8     tokens per shard file
  SHARD_DIR=shards     output directory

Resume: the progress lives in shards/manifest.json (incl. document counter
per shard). After an abort, finished shards are kept and the stream is
fast-forwarded via dataset.skip(n_docs) — only the download is redone,
not the tokenization.
"""

import json
import os
import time

import numpy as np
import tiktoken
from datasets import load_dataset


def _env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return cast(raw)


# int(float(...)): allows "8e9" notation in the ENV var.
FINEWEB_TOKENS = int(_env("FINEWEB_TOKENS", 7e9, float))
WIKI_TOKENS = int(_env("WIKI_TOKENS", 2e9, float))
CODE_TOKENS = int(_env("CODE_TOKENS", 1e9, float))
SHARD_TOKENS = int(_env("SHARD_TOKENS", 1e8, float))
SHARD_DIR = _env("SHARD_DIR", "shards")
DOCS_PER_BATCH = 512   # documents per encode_ordinary_batch call

# Source definitions: (name, token_budget, load_dataset kwargs, text_field).
#   - FineWeb2: quality-filtered, deduplicated web German — the modern
#     pretraining standard. Diversity (dialogue, instruction, narrative).
#   - Wikipedia: clean and fact-rich, but stylistically monotonous.
#   - Python code (~10% of the mix): code is the logically densest text —
#     nested structures, exact references. It demonstrably improves
#     language/structure abilities too AND keeps the door open for later
#     coding efforts. CODE_TOKENS=0 turns the source off.
SOURCES = [
    ("fineweb", FINEWEB_TOKENS,
     dict(path="HuggingFaceFW/fineweb-2", name="deu_Latn", split="train"), "text"),
    ("wiki", WIKI_TOKENS,
     dict(path="wikimedia/wikipedia", name="20231101.de", split="train"), "text"),
    ("code", CODE_TOKENS,
     dict(path="codeparrot/codeparrot-clean", split="train"), "content"),
]


class ShardWriter:
    """Collects token arrays and writes full shards + manifest.

    The manifest is rewritten atomically after EVERY shard (tmp +
    os.replace) — if the job aborts, the last state is consistent and
    the resume knows exactly how many documents of each source have
    been processed.
    """

    def __init__(self, out_dir, shard_tokens):
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        os.makedirs(out_dir, exist_ok=True)
        self.manifest_path = os.path.join(out_dir, "manifest.json")
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {
                "tokenizer": "gpt2",
                "dtype": "uint16",
                "eot_between_docs": True,
                "shards": [],
            }
        self._buffer = []       # list of np.uint16 arrays
        self._buffered = 0      # tokens in the buffer
        self._buffered_docs = 0

    def source_progress(self, source):
        """(tokens, docs) that already lie in shards for this source."""
        toks = sum(s["num_tokens"] for s in self.manifest["shards"]
                   if s["source"] == source)
        docs = sum(s["num_docs"] for s in self.manifest["shards"]
                   if s["source"] == source)
        return toks, docs

    def add(self, arr):
        self._buffer.append(arr)
        self._buffered += arr.size
        self._buffered_docs += 1

    def flush_if_full(self, source):
        while self._buffered >= self.shard_tokens:
            self._write_shard(source)

    def finish_source(self, source):
        """Write the remaining buffer as a (smaller) closing shard."""
        if self._buffered > 0:
            self._write_shard(source, partial_ok=True)

    def _write_shard(self, source, partial_ok=False):
        # Cut off exactly shard_tokens; the rest stays in the buffer.
        # (For the closing shard: everything that is there.)
        take = self._buffered if partial_ok else self.shard_tokens
        chunks, got, docs = [], 0, 0
        while self._buffer and got < take:
            arr = self._buffer.pop(0)
            if got + arr.size > take:
                # Document exceeds the shard boundary: split it, put the
                # rest back into the buffer. The doc counter attributes the
                # document to the shard in which it ENDS — only that way is
                # the sum correct for the resume skip.
                head, tail = arr[: take - got], arr[take - got:]
                chunks.append(head)
                got += head.size
                self._buffer.insert(0, tail)
            else:
                chunks.append(arr)
                got += arr.size
                docs += 1
        self._buffered -= got
        self._buffered_docs -= docs

        idx = len(self.manifest["shards"])
        fname = f"shard_{idx:04d}_{source}.bin"
        np.concatenate(chunks).tofile(os.path.join(self.out_dir, fname))
        self.manifest["shards"].append(
            {"file": fname, "source": source, "num_tokens": got, "num_docs": docs}
        )
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=1)
        os.replace(tmp, self.manifest_path)
        total = sum(s["num_tokens"] for s in self.manifest["shards"])
        print(f"  Shard written: {fname} ({got:,} tokens, {docs:,} docs) "
              f"— total {total:,} tokens")


def build_source(writer, source, target_tokens, ds_kwargs, encoding, eot,
                 text_key="text"):
    done_tokens, done_docs = writer.source_progress(source)
    if done_tokens >= target_tokens:
        print(f"[{source}] already finished ({done_tokens:,} tokens) — skip")
        return
    print(f"[{source}] target {target_tokens:,} tokens "
          f"(available: {done_tokens:,}) — streaming...")

    ds = load_dataset(streaming=True, **ds_kwargs)
    if done_docs > 0:
        # Resume: skip documents that already lie in shards.
        # skip() fast-forwards the stream — the download runs through
        # again, but the expensive tokenization does not.
        print(f"[{source}] Resume: skipping {done_docs:,} documents")
        ds = ds.skip(done_docs)

    n_threads = os.cpu_count() or 4
    produced = done_tokens
    t0 = time.time()
    next_log = produced + 10_000_000
    batch = []

    def process(texts):
        nonlocal produced
        for toks in encoding.encode_ordinary_batch(texts, num_threads=n_threads):
            # Budget check per DOCUMENT, not per batch: a 512-article
            # batch can be millions of tokens large — without this check
            # the budget would be overshot by up to a whole batch.
            if produced >= target_tokens:
                break
            arr = np.asarray(toks + [eot], dtype=np.uint16)
            writer.add(arr)
            produced += arr.size
        writer.flush_if_full(source)

    for doc in ds:
        batch.append(doc[text_key])
        if len(batch) >= DOCS_PER_BATCH:
            process(batch)
            batch = []
            if produced >= target_tokens:
                break
            if produced >= next_log:
                rate = (produced - done_tokens) / max(1e-9, time.time() - t0)
                eta_min = (target_tokens - produced) / max(1e-9, rate) / 60
                print(f"  [{source}] {produced:,}/{target_tokens:,} tokens "
                      f"({rate / 1e6:.1f}M tok/s, ETA {eta_min:.0f} min)")
                next_log = produced + 10_000_000
    else:
        # Stream ended before the budget was reached (e.g. all of
        # Wikipedia < WIKI_TOKENS): process the rest and log it honestly.
        if batch:
            process(batch)
        print(f"  [{source}] source exhausted at {produced:,} tokens "
              f"(target was {target_tokens:,})")

    writer.finish_source(source)
    print(f"[{source}] finished: {writer.source_progress(source)[0]:,} tokens")


def main():
    encoding = tiktoken.get_encoding("gpt2")
    eot = encoding.eot_token  # <|endoftext|>, ID 50256

    planned = sum(t for _, t, _, _ in SOURCES)
    print(f"Plan: {planned:,} tokens -> ~{planned * 2 / 1024**3:.1f} GB "
          f"in '{SHARD_DIR}/' (uint16)")

    writer = ShardWriter(SHARD_DIR, SHARD_TOKENS)
    for source, target, ds_kwargs, text_key in SOURCES:
        if target <= 0:
            continue
        build_source(writer, source, target, ds_kwargs, encoding, eot,
                     text_key=text_key)

    total = sum(s["num_tokens"] for s in writer.manifest["shards"])
    print(f"\nDone: {total:,} tokens in {len(writer.manifest['shards'])} shards.")
    print(f"Training starts with: SHARD_MANIFEST={SHARD_DIR}/manifest.json")


if __name__ == "__main__":
    main()
