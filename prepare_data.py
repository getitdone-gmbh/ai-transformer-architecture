"""Streaming-Datenaufbereitung: Quellen -> tokenisierte uint16-Shards.

Fuer den grossen Lauf (350M @ ~10 Mrd. Tokens) funktioniert der alte Weg
(alles herunterladen -> ein Token-Tensor im RAM) nicht mehr:
  - Rohdaten (~40-60 GB Text) sollen nie komplett auf der Disk liegen.
  - 10 Mrd. Tokens passen in keinen RAM, wenn man sie als Liste baut.

Deshalb der Standard-Ansatz (nanoGPT / GPT-2-Replikationen):
  1. Quellen STREAMEN (HuggingFace streaming=True laedt haeppchenweise),
  2. on-the-fly tokenisieren (tiktoken, multi-threaded),
  3. als flache uint16-Binaer-Shards schreiben (~100M Tokens = ~200 MB je).

Warum uint16: GPT-2-Vocab hat 50.257 IDs < 65.536 -> 2 Bytes/Token statt 4.
10 Mrd. Tokens = ~20 GB auf Disk. Das Training liest die Shards spaeter per
np.memmap — es laedt also NIE alles in den RAM (siehe ShardDataset in
train.py).

Neu gegenueber der alten Pipeline: zwischen Dokumenten steht das
<|endoftext|>-Token. Ohne Trenner kleben Artikel uebergangslos aneinander
und das Modell lernt nie, dass ein Kontext-Reset existiert.

Konfiguration per ENV (alle optional):
  FINEWEB_TOKENS=8e9   Token-Budget aus FineWeb2-Deutsch (0 = Quelle aus)
  WIKI_TOKENS=2e9      Token-Budget aus Wikipedia-DE     (0 = Quelle aus)
  SHARD_TOKENS=1e8     Tokens pro Shard-Datei
  SHARD_DIR=shards     Ausgabe-Verzeichnis

Resume: der Fortschritt steht in shards/manifest.json (inkl. Dokument-
Zaehler pro Shard). Nach einem Abbruch werden fertige Shards behalten und
der Stream per dataset.skip(n_docs) vorgespult — es wird nur neu
heruntergeladen, nicht neu tokenisiert.
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


# int(float(...)): erlaubt "8e9"-Schreibweise in der ENV-Var.
FINEWEB_TOKENS = int(_env("FINEWEB_TOKENS", 8e9, float))
WIKI_TOKENS = int(_env("WIKI_TOKENS", 2e9, float))
SHARD_TOKENS = int(_env("SHARD_TOKENS", 1e8, float))
SHARD_DIR = _env("SHARD_DIR", "shards")
DOCS_PER_BATCH = 512   # Dokumente pro encode_ordinary_batch-Aufruf

# Quellen-Definitionen. FineWeb2 = qualitaetsgefiltertes, dedupliziertes
# Web-Deutsch (der moderne Pretraining-Standard); Wikipedia = sauber, aber
# stilistisch eintoenig. Die Mischung gibt Diversitaet UND Faktendichte.
SOURCES = [
    ("fineweb", FINEWEB_TOKENS,
     dict(path="HuggingFaceFW/fineweb-2", name="deu_Latn", split="train")),
    ("wiki", WIKI_TOKENS,
     dict(path="wikimedia/wikipedia", name="20231101.de", split="train")),
]


class ShardWriter:
    """Sammelt Token-Arrays und schreibt volle Shards + Manifest.

    Das Manifest wird nach JEDEM Shard atomar neu geschrieben (tmp +
    os.replace) — bricht der Job ab, ist der letzte Stand konsistent und
    der Resume weiss exakt, wie viele Dokumente jeder Quelle verarbeitet
    sind.
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
        self._buffer = []       # Liste von np.uint16-Arrays
        self._buffered = 0      # Tokens im Buffer
        self._buffered_docs = 0

    def source_progress(self, source):
        """(tokens, docs) die fuer diese Quelle schon in Shards liegen."""
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
        """Rest-Buffer als (kleineren) Abschluss-Shard schreiben."""
        if self._buffered > 0:
            self._write_shard(source, partial_ok=True)

    def _write_shard(self, source, partial_ok=False):
        # Genau shard_tokens abschneiden; der Rest bleibt im Buffer.
        # (Beim Abschluss-Shard: alles was da ist.)
        take = self._buffered if partial_ok else self.shard_tokens
        chunks, got, docs = [], 0, 0
        while self._buffer and got < take:
            arr = self._buffer.pop(0)
            if got + arr.size > take:
                # Dokument ueberschreitet die Shard-Grenze: splitten, Rest
                # zurueck in den Buffer. Der Doc-Zaehler bucht das Dokument
                # dem Shard zu, in dem es ENDET — nur so stimmt die Summe
                # fuer den Resume-Skip.
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
        print(f"  Shard geschrieben: {fname} ({got:,} Tokens, {docs:,} Docs) "
              f"— gesamt {total:,} Tokens")


def build_source(writer, source, target_tokens, ds_kwargs, encoding, eot):
    done_tokens, done_docs = writer.source_progress(source)
    if done_tokens >= target_tokens:
        print(f"[{source}] bereits fertig ({done_tokens:,} Tokens) — skip")
        return
    print(f"[{source}] Ziel {target_tokens:,} Tokens "
          f"(vorhanden: {done_tokens:,}) — streame...")

    ds = load_dataset(streaming=True, **ds_kwargs)
    if done_docs > 0:
        # Resume: Dokumente ueberspringen, die schon in Shards liegen.
        # skip() spult den Stream vor — Download laeuft nochmal durch,
        # aber die teure Tokenisierung nicht.
        print(f"[{source}] Resume: ueberspringe {done_docs:,} Dokumente")
        ds = ds.skip(done_docs)

    n_threads = os.cpu_count() or 4
    produced = done_tokens
    t0 = time.time()
    next_log = produced + 10_000_000
    batch = []

    def process(texts):
        nonlocal produced
        for toks in encoding.encode_ordinary_batch(texts, num_threads=n_threads):
            # Budget-Check pro DOKUMENT, nicht pro Batch: ein 512-Artikel-
            # Batch kann Millionen Tokens gross sein — ohne diesen Check
            # wuerde das Budget um bis zu einen ganzen Batch ueberschossen.
            if produced >= target_tokens:
                break
            arr = np.asarray(toks + [eot], dtype=np.uint16)
            writer.add(arr)
            produced += arr.size
        writer.flush_if_full(source)

    for doc in ds:
        batch.append(doc["text"])
        if len(batch) >= DOCS_PER_BATCH:
            process(batch)
            batch = []
            if produced >= target_tokens:
                break
            if produced >= next_log:
                rate = (produced - done_tokens) / max(1e-9, time.time() - t0)
                eta_min = (target_tokens - produced) / max(1e-9, rate) / 60
                print(f"  [{source}] {produced:,}/{target_tokens:,} Tokens "
                      f"({rate / 1e6:.1f}M tok/s, ETA {eta_min:.0f} min)")
                next_log = produced + 10_000_000
    else:
        # Stream zu Ende, bevor das Budget erreicht war (z.B. ganze
        # Wikipedia < WIKI_TOKENS): Rest verarbeiten und ehrlich loggen.
        if batch:
            process(batch)
        print(f"  [{source}] Quelle erschoepft bei {produced:,} Tokens "
              f"(Ziel war {target_tokens:,})")

    writer.finish_source(source)
    print(f"[{source}] fertig: {writer.source_progress(source)[0]:,} Tokens")


def main():
    encoding = tiktoken.get_encoding("gpt2")
    eot = encoding.eot_token  # <|endoftext|>, ID 50256

    planned = sum(t for _, t, _ in SOURCES)
    print(f"Plan: {planned:,} Tokens -> ~{planned * 2 / 1024**3:.1f} GB "
          f"in '{SHARD_DIR}/' (uint16)")

    writer = ShardWriter(SHARD_DIR, SHARD_TOKENS)
    for source, target, ds_kwargs in SOURCES:
        if target <= 0:
            continue
        build_source(writer, source, target, ds_kwargs, encoding, eot)

    total = sum(s["num_tokens"] for s in writer.manifest["shards"])
    print(f"\nFertig: {total:,} Tokens in {len(writer.manifest['shards'])} Shards.")
    print(f"Training startet mit: SHARD_MANIFEST={SHARD_DIR}/manifest.json")


if __name__ == "__main__":
    main()
