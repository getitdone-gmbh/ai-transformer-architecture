"""Phase 1, step 1: stream a tokenizer-training corpus to local text files.

A tokenizer is a compressor learned on a text MIX — whatever the mix
over-represents gets the cheap (short) codes. Our target: German first,
but code and English must stay cheap, because a Gemma that speaks German
in fewer tokens but suddenly writes Python in 2x the tokens is no win.

    de    FineWeb2 deu_Latn          (same source as our pretraining)
    code  codeparrot-clean (Python)  (same source as our code eval axis)
    en    FineWeb sample-10BT        (first file; eval uses the last)

Budgets in bytes, env-overridable: DE_BYTES=7e8 CODE_BYTES=2e8 EN_BYTES=1e8
(= 70/20/10 of 1 GB). Output: one text file per source in WORK_DIR,
documents separated by a NUL byte (never occurs in web text or code).

Contamination guard: documents whose md5 is in data/eval_exclude_hashes.json
(the frozen eval set) are skipped — same rule as prepare_data.py.

Also writes the held-out English eval file eval_data/fineweb_en.jsonl
(~1.5 MB from the LAST sample-10BT file, never in the training stream),
so tokenizer_efficiency.py can check that English does not get worse.
"""

import hashlib
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
from datasets import load_dataset  # noqa: E402

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from paths import EVAL_DATA, data_path  # noqa: E402

WORK_DIR = os.path.expanduser(os.environ.get("WORK_DIR", "~/gemma_de_work"))
SOURCES = [
    ("de", float(os.environ.get("DE_BYTES", 7e8)),
     dict(path="HuggingFaceFW/fineweb-2", name="deu_Latn", split="train"), "text"),
    ("code", float(os.environ.get("CODE_BYTES", 2e8)),
     dict(path="codeparrot/codeparrot-clean", split="train"), "content"),
    ("en", float(os.environ.get("EN_BYTES", 1e8)),
     dict(path="HuggingFaceFW/fineweb", data_files="sample/10BT/000_00000.parquet",
          split="train"), "text"),
]
EN_EVAL = dict(path="HuggingFaceFW/fineweb",
               data_files="sample/10BT/014_00000.parquet", split="train")
EN_EVAL_BYTES = 1.5e6
MIN_DOC, MAX_DOC = 200, 50_000   # same document filter as the frozen eval set


def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def stream(ds_kwargs, field):
    for doc in load_dataset(streaming=True, **ds_kwargs):
        yield doc[field]


def build_en_eval(exclude):
    out = os.path.join(EVAL_DATA, "fineweb_en.jsonl")
    if os.path.exists(out):
        print(f"exists: {out}")
        return
    total, docs = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for text in stream(EN_EVAL, "text"):
            n = len(text.encode("utf-8"))
            if not MIN_DOC <= n <= MAX_DOC or md5(text) in exclude:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total, docs = total + n, docs + 1
            if total >= EN_EVAL_BYTES:
                break
    print(f"wrote {out}: {docs} docs, {total/1e6:.2f} MB")


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(data_path("eval_exclude_hashes.json")) as f:
        exclude = frozenset(json.load(f)["md5"])
    build_en_eval(exclude)

    for name, budget, ds_kwargs, field in SOURCES:
        out = os.path.join(WORK_DIR, f"{name}.txt")
        if os.path.exists(out) and os.path.getsize(out) >= budget:
            print(f"exists: {out}")
            continue
        total, skipped = 0, 0
        with open(out, "w", encoding="utf-8") as f:
            for text in stream(ds_kwargs, field):
                if md5(text) in exclude or "\0" in text:
                    skipped += 1
                    continue
                f.write(text + "\0")
                total += len(text.encode("utf-8"))
                if total >= budget:
                    break
        print(f"wrote {out}: {total/1e6:.0f} MB ({skipped} docs skipped)")


if __name__ == "__main__":
    main()
