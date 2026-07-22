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

import hashlib
import json
import os
import time

# A stale token in the local keychain breaks even anonymous HF access
# (401 on public repos). Everything we pull is public, so force anonymous
# mode. Must happen BEFORE the hub libraries are imported. On a rented
# instance (no stored token) this is a no-op.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

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
HQ_TOKENS = int(_env("HQ_TOKENS", 0, float))  # FineWeb2-HQ German, default off
SHARD_TOKENS = int(_env("SHARD_TOKENS", 1e8, float))
SHARD_DIR = _env("SHARD_DIR", "shards")
DOCS_PER_BATCH = 512   # documents per encode_ordinary_batch call

# Contamination guard: md5 hashes of documents that must NEVER enter the
# training data — the frozen bpb eval set (compare_bpb.py writes this file).
# FineWeb2-HQ is filtered from ALL of FineWeb2, so it can contain the very
# documents our eval set was built from; training on them would silently
# inflate every future before/after measurement. Exact-hash matching
# catches exactly that case (HQ keeps the text verbatim).
EXCLUDE_HASHES = _env("EXCLUDE_HASHES", "eval_exclude_hashes.json")

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
    #   - FineWeb2-HQ: model-based top-10% quality filter over FineWeb2
    #     (EPFL) — the "textbook density" bet for continued pretraining.
    #     No named config on the hub, hence the raw parquet glob.
    ("fineweb_hq", HQ_TOKENS,
     dict(path="parquet",
          data_files="hf://datasets/epfml/FineWeb2-HQ/deu_Latn/*.parquet",
          split="train"), "text"),
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
        """(tokens, stream position) already covered by shards for this source.

        The second value is what a resume must feed to dataset.skip(). With
        hash filtering it is NOT the number of docs in the shards: filtered
        docs were consumed from the stream but never written. Each shard
        therefore records "stream_docs" — the stream position after the last
        document that ENDS in it. Old manifests (written before filtering
        existed) lack the field; there stream position == docs written.
        """
        shards = [s for s in self.manifest["shards"] if s["source"] == source]
        toks = sum(s["num_tokens"] for s in shards)
        if any("stream_docs" in s for s in shards):
            pos = max(s.get("stream_docs", 0) for s in shards)
        else:
            pos = sum(s["num_docs"] for s in shards)
        return toks, pos

    def add(self, arr, stream_idx):
        self._buffer.append((arr, stream_idx))
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
        last_stream_idx = None
        while self._buffer and got < take:
            arr, stream_idx = self._buffer.pop(0)
            if got + arr.size > take:
                # Document exceeds the shard boundary: split it, put the
                # rest back into the buffer. The doc counter attributes the
                # document to the shard in which it ENDS — only that way is
                # the sum correct for the resume skip.
                head, tail = arr[: take - got], arr[take - got:]
                chunks.append(head)
                got += head.size
                self._buffer.insert(0, (tail, stream_idx))
            else:
                chunks.append(arr)
                got += arr.size
                docs += 1
                last_stream_idx = stream_idx
        self._buffered -= got
        self._buffered_docs -= docs

        idx = len(self.manifest["shards"])
        fname = f"shard_{idx:04d}_{source}.bin"
        np.concatenate(chunks).tofile(os.path.join(self.out_dir, fname))
        entry = {"file": fname, "source": source, "num_tokens": got, "num_docs": docs}
        if last_stream_idx is not None:
            # Stream position after the last doc that ends here — the
            # resume skip target (includes hash-filtered docs, see
            # source_progress).
            entry["stream_docs"] = last_stream_idx + 1
        self.manifest["shards"].append(entry)
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=1)
        os.replace(tmp, self.manifest_path)
        total = sum(s["num_tokens"] for s in self.manifest["shards"])
        print(f"  Shard written: {fname} ({got:,} tokens, {docs:,} docs) "
              f"— total {total:,} tokens")


def build_source(writer, source, target_tokens, ds_kwargs, encoding, eot,
                 text_key="text", exclude=frozenset()):
    done_tokens, done_docs = writer.source_progress(source)
    if done_tokens >= target_tokens:
        print(f"[{source}] already finished ({done_tokens:,} tokens) — skip")
        return
    print(f"[{source}] target {target_tokens:,} tokens "
          f"(available: {done_tokens:,}) — streaming...")

    ds = load_dataset(streaming=True, **ds_kwargs)
    if done_docs > 0:
        # Resume: skip to the stream position covered by existing shards.
        # skip() fast-forwards the stream — the download runs through
        # again, but the expensive tokenization does not.
        print(f"[{source}] Resume: skipping {done_docs:,} documents")
        ds = ds.skip(done_docs)

    n_threads = os.cpu_count() or 4
    produced = done_tokens
    stream_idx = done_docs   # position in the source stream, incl. filtered
    n_filtered = 0
    t0 = time.time()
    next_log = produced + 10_000_000
    batch = []               # list of (text, stream_idx)

    def process(items):
        nonlocal produced
        texts = [t for t, _ in items]
        encoded = encoding.encode_ordinary_batch(texts, num_threads=n_threads)
        for toks, (_, idx) in zip(encoded, items):
            # Budget check per DOCUMENT, not per batch: a 512-article
            # batch can be millions of tokens large — without this check
            # the budget would be overshot by up to a whole batch.
            if produced >= target_tokens:
                break
            arr = np.asarray(toks + [eot], dtype=np.uint16)
            writer.add(arr, idx)
            produced += arr.size
        writer.flush_if_full(source)

    for doc in ds:
        text = doc[text_key]
        idx = stream_idx
        stream_idx += 1
        if exclude and hashlib.md5(text.encode("utf-8")).hexdigest() in exclude:
            n_filtered += 1
            continue
        batch.append((text, idx))
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
    if n_filtered:
        print(f"  [{source}] contamination filter: {n_filtered:,} eval-set "
              f"documents excluded")
    print(f"[{source}] finished: {writer.source_progress(source)[0]:,} tokens")


def main():
    encoding = tiktoken.get_encoding("gpt2")
    eot = encoding.eot_token  # <|endoftext|>, ID 50256

    planned = sum(t for _, t, _, _ in SOURCES)
    print(f"Plan: {planned:,} tokens -> ~{planned * 2 / 1024**3:.1f} GB "
          f"in '{SHARD_DIR}/' (uint16)")

    exclude = frozenset()
    if os.path.exists(EXCLUDE_HASHES):
        with open(EXCLUDE_HASHES) as f:
            exclude = frozenset(json.load(f)["md5"])
        print(f"Contamination guard: {len(exclude)} eval-set hashes "
              f"from '{EXCLUDE_HASHES}'")
    elif HQ_TOKENS > 0:
        # HQ overlaps the ORIGINAL FineWeb2 by construction — preparing it
        # without the guard would poison the frozen eval set. Refuse.
        raise RuntimeError(
            f"HQ_TOKENS is set but '{EXCLUDE_HASHES}' is missing — "
            "run compare_bpb.py once (it writes the hash file) or set "
            "EXCLUDE_HASHES to its location."
        )

    writer = ShardWriter(SHARD_DIR, SHARD_TOKENS)
    for source, target, ds_kwargs, text_key in SOURCES:
        if target <= 0:
            continue
        build_source(writer, source, target, ds_kwargs, encoding, eot,
                     text_key=text_key, exclude=exclude)

    total = sum(s["num_tokens"] for s in writer.manifest["shards"])
    print(f"\nDone: {total:,} tokens in {len(writer.manifest['shards'])} shards.")
    print(f"Training starts with: SHARD_MANIFEST={SHARD_DIR}/manifest.json")


if __name__ == "__main__":
    main()
