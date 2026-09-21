"""Einen deutschen BPE-Tokenizer trainieren — der Gegenentwurf zu GPT-2.

WARUM: unser Modell benutzt den GPT-2-BPE. Der ist auf englischem
Web-Text gelernt und zerlegt Deutsch in Krümel (2,6 Zeichen/Token gegen
5,7 auf Englisch). Dieses Skript baut denselben Algorithmus — byte-level
BPE, identische Vokabulargröße — nur auf deutschem Text gelernt, damit
der Unterschied MESSBAR wird statt behauptet.

Die Vokabulargröße ist absichtlich gleich (50.257): sonst vergleicht man
Vokabulargrößen statt Sprachen. Und sie muss ohnehin unter 65.536
bleiben, weil prepare_data.py die Shards als uint16 schreibt.

MESSMETHODIK: trainiert wird auf fineweb.jsonl, gemessen auf
fineweb_hq.jsonl — verschiedene Dokumente. Ein Tokenizer auf seinem
eigenen Trainingstext gemessen sieht immer zu gut aus.

WAS DAS SKRIPT NICHT TUT: das Modell umtrainieren. Ein neuer Tokenizer
vergibt alle Token-IDs neu, und eine ID ist nichts als die Zeilennummer
in der Embedding-Matrix — mit neuen IDs zeigt jedes gelernte Muster auf
die falsche Zeile. Der Tokenizer hier kostet Minuten; ihn im Modell zu
NUTZEN kostet ein vollständiges Pretraining (~1,5 Tage H100, ~$60-75).
Deshalb ist das ein Werkzeug zum Zeigen und Messen, nicht zum Umstellen.

DATENMENGE — die wichtigste Einschränkung: der Standardlauf nimmt die
~3,4 MB aus eval_data/, weil sie lokal liegen und das Ergebnis in zwei
Minuten da ist. Für einen BPE ist das WENIG (üblich: Gigabytes); die
seltenen Merges sind dann Rauschen. Die gemessene Kompression ist damit
eine konservative UNTERGRENZE — mit mehr Text wird sie besser, nicht
schlechter. Für einen echten Tokenizer:

    BYTES=1e9 python explain/train_tokenizer.py     # streamt FineWeb2 vom Hub

Usage:
    python explain/train_tokenizer.py                 # schnell, aus eval_data/
    BYTES=1e9 python explain/train_tokenizer.py       # echter Lauf, HF-Stream
    VOCAB=32000 python explain/train_tokenizer.py     # andere Vokabulargröße
    OUT=de_bpe.json python explain/train_tokenizer.py

Danach zeigt tokenize_de.py den neuen Tokenizer automatisch als
dritte Spalte:
    python explain/tokenize_de.py "Fußgängerüberweg"
"""

import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import tiktoken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# model/ on sys.path — the repo layout lives in model/paths.py, and
# anchoring to __file__ is what lets this script run from any directory.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
from paths import EVAL_DATA, data_path  # noqa: E402

VOCAB = int(float(os.environ.get("VOCAB", "50257")))
OUT = data_path(os.environ.get("OUT", "de_bpe.json"))
BYTES = float(os.environ.get("BYTES", "0"))   # 0 = lokal aus eval_data/

# Muss zu prepare_data.py passen: uint16-Shards können nur IDs < 65.536.
if VOCAB >= 65536:
    raise SystemExit(
        f"VOCAB={VOCAB} passt nicht in uint16 — prepare_data.py schreibt "
        "die Shards als uint16 (< 65.536). Größeres Vokabular würde die "
        "Shards auf der Platte verdoppeln.")


def local_texts(path):
    with open(path) as f:
        return [json.loads(line)["text"] for line in f]


def hub_texts(budget_bytes):
    """FineWeb2-Deutsch vom Hub streamen, bis das Byte-Budget voll ist.

    Dieselbe Quelle wie prepare_data.py — der Tokenizer lernt damit auf
    derselben Textverteilung, auf der das Modell später trainiert.
    """
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="deu_Latn",
                      split="train", streaming=True)
    total = 0
    for doc in ds:
        text = doc["text"]
        total += len(text.encode("utf-8"))
        yield text
        if total >= budget_bytes:
            break


if BYTES > 0:
    print(f"Trainingstext: FineWeb2-Stream, Budget {BYTES/1e9:.2f} GB")
    train_iter = hub_texts(BYTES)
    trained_on = f"fineweb-2 stream, {BYTES/1e9:.2f} GB"
else:
    train_texts = local_texts(os.path.join(EVAL_DATA, "fineweb.jsonl"))
    mb = sum(len(t) for t in train_texts) / 1e6
    print(f"Trainingstext: eval_data/fineweb.jsonl, {mb:.1f} MB")
    print("  (klein für einen BPE — das Ergebnis ist eine Untergrenze; "
          "BYTES=1e9 für einen echten Lauf)")
    train_iter = train_texts
    trained_on = f"eval_data/fineweb.jsonl, {mb:.1f} MB"

# Byte-level BPE, genau wie GPT-2: kann jedes Byte darstellen, also
# gibt es kein <UNK> und keine kaputten Umlaute. Der einzige Unterschied
# zu GPT-2 ist der Text, auf dem gelernt wird.
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(train_iter, trainers.BpeTrainer(
    vocab_size=VOCAB,
    special_tokens=["<|endoftext|>"],   # gleicher Name wie bei GPT-2
    show_progress=True,
))
tok.save(OUT)
print(f"\nGespeichert: {OUT}  ({tok.get_vocab_size()} Tokens)")

# Held-out-Messung: andere Dokumente als der Trainingstext.
test_texts = local_texts(os.path.join(EVAL_DATA, "fineweb_hq.jsonl"))
chars = sum(len(t) for t in test_texts)
gpt2 = tiktoken.get_encoding("gpt2")
n_gpt2 = sum(len(gpt2.encode(t)) for t in test_texts)
n_de = sum(len(tok.encode(t).ids) for t in test_texts)

print(f"\nHeld-out ({chars/1e6:.1f} MB aus fineweb_hq.jsonl, "
      f"NICHT im Tokenizer-Training):")
print(f"  GPT-2  50k (englisch gelernt): {n_gpt2:>8} Tokens  "
      f"{chars/n_gpt2:.2f} Zeichen/Token")
print(f"  DE-BPE {tok.get_vocab_size()//1000}k (selbst gelernt): "
      f"{n_de:>8} Tokens  {chars/n_de:.2f} Zeichen/Token")
print(f"  -> {n_gpt2/n_de:.2f}x weniger Tokens für denselben Text")
print(f"\nWas das WERT ist: bei gleichem GPU-Budget {n_gpt2/n_de:.2f}x mehr "
      f"Text pro Lauf,\nund ein 2048-Token-Kontext hält "
      f"{n_gpt2/n_de:.2f}x so viel Inhalt. Was es KOSTET:\n"
      "ein vollständiges Pretraining, weil alle Token-IDs neu sind.")

with open(OUT + ".meta.json", "w") as f:
    json.dump({"trained_on": trained_on, "vocab_size": tok.get_vocab_size(),
               "heldout_chars_per_token": round(chars / n_de, 3),
               "gpt2_chars_per_token": round(chars / n_gpt2, 3)}, f, indent=2)
