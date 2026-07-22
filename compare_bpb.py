"""Compare our model against HF reference models in bits per byte (bpb).

THE PROBLEM: loss per token is not comparable across models with different
tokenizers. A tokenizer that cuts text into more pieces makes every single
piece easier to predict — Qwen's 152k vocab vs our GPT-2 50k vocab would
make a raw loss comparison meaningless.

THE FIX: normalize to the raw text. Every language model is, at its core,
a text compressor; how many BITS it needs per BYTE of raw text is
tokenizer-independent:

    bpb = sum_of_loss_in_nats / (ln(2) * utf8_bytes_of_text)

Both models read the SAME raw documents (each through its own tokenizer),
we sum the cross-entropy over everything each model predicts, and divide
by the same byte count. Lower = better compression = better model.

EVAL DATA — the one thing that must be right: text our model has NEVER
seen. prepare_data.py consumed the sources sequentially from the start
(fineweb: ~7B of hundreds of B tokens = the first handful of 181 files;
code: 1B of ~50B = roughly the first file). So we take documents from the
LAST file of each source — guaranteed behind the training cutoff, same
distribution. Wikipedia is deliberately absent: its 2B-token budget may
have exhausted the whole dump, so no article is safely unseen.
Honest caveat that remains: Qwen has likely seen similar web text in its
own pretraining — contamination, if any, favors the reference model.

FAIRNESS RULES (identical treatment for both models):
  - Same raw documents, same order, same byte denominator.
  - Same sequence length (default 2048 = our training length; Qwen could
    use 32k context, which would flatter it — we measure model quality,
    not context length).
  - Documents joined by each model's own <|endoftext|> token (matches how
    both were pretrained). Separator losses count in the numerator but
    the separator adds no bytes — a small symmetric cost.
  - Windows overlap by 1 token, so every token of the stream except the
    very first is predicted exactly once.

Usage:
    python compare_bpb.py                          # both models, full run
    SKIP_HF=1 python compare_bpb.py                # only our checkpoint
    CHECKPOINT=sft_540m_v2.pt python compare_bpb.py
    HF_MODEL=meta-llama/Llama-3.2-1B python compare_bpb.py

Config via ENV (all optional):
    CHECKPOINT=weights_540m_fp32.pt   our checkpoint (base model = fair,
                                      since Qwen2.5-0.5B base is no-SFT)
    HF_MODEL=Qwen/Qwen2.5-0.5B        reference model (NOT -Instruct!)
    FINEWEB_EVAL_BYTES=3.5e6          eval text budget, German web
    CODE_EVAL_BYTES=0.5e6             eval text budget, Python (7:1 like
                                      the training mix fineweb:code)
    SEQ_LEN=2048  BATCH_SIZE=2        BATCH_SIZE=2 keeps the fp32 logits
                                      tensor <3 GB even at Qwen's vocab
    EVAL_DIR=eval_data                frozen eval set, built once and
                                      reused for every later model
                                      (post-pruning, post-quantization...)

The eval set is FROZEN on first run (eval_data/ + manifest). Delete the
directory to rebuild. Keeping it fixed is what makes numbers comparable
across experiments weeks apart.
"""

import hashlib
import json
import math
import os
import time

# A stale token in the local keychain breaks even anonymous HF access
# (401 on public repos). Everything we pull is public, so force
# anonymous mode. Must happen BEFORE the hub libraries are imported.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import tiktoken
import torch
import torch.nn.functional as F

import train


def _env(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return cast(raw)


CHECKPOINT = _env("CHECKPOINT", "weights_540m_fp32.pt")
HF_MODEL = _env("HF_MODEL", "Qwen/Qwen2.5-0.5B")
SEQ_LEN = int(_env("SEQ_LEN", 2048, float))
BATCH_SIZE = int(_env("BATCH_SIZE", 2, float))
EVAL_DIR = _env("EVAL_DIR", "eval_data")
SKIP_HF = _env("SKIP_HF", "0") in ("1", "true")
SKIP_OURS = _env("SKIP_OURS", "0") in ("1", "true")

# (source, byte budget, hub glob for the file list, format, text field)
EVAL_SOURCES = [
    ("fineweb", int(_env("FINEWEB_EVAL_BYTES", 3.5e6, float)),
     "datasets/HuggingFaceFW/fineweb-2/data/deu_Latn/train/*.parquet",
     "parquet", "text"),
    ("code", int(_env("CODE_EVAL_BYTES", 0.5e6, float)),
     "datasets/codeparrot/codeparrot-clean/*.json.gz",
     "json", "content"),
]

# Per-document size limits: drop trivially short docs (boilerplate) and
# oversized ones (a single 300 KB code file would dominate its source's
# whole budget). Applied while BUILDING the eval set, i.e. identically
# for every model that will ever be scored against it.
MIN_DOC_BYTES = 200
MAX_DOC_BYTES = 50_000


# ---------------------------------------------------------------------------
# Eval set: build once, freeze, reuse
# ---------------------------------------------------------------------------

def build_eval_set():
    """Fill EVAL_DIR with one .jsonl per source + a provenance manifest."""
    manifest_path = os.path.join(EVAL_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Eval set exists ({EVAL_DIR}/) — reusing frozen set.")
        return manifest

    from datasets import load_dataset
    from huggingface_hub import HfFileSystem

    os.makedirs(EVAL_DIR, exist_ok=True)
    fs = HfFileSystem(token=False)
    manifest = {"seq_len_note": "eval set is model-independent", "sources": []}

    for source, budget, pattern, fmt, text_key in EVAL_SOURCES:
        files = sorted(fs.glob(pattern))
        if not files:
            raise RuntimeError(f"[{source}] no files match {pattern}")
        last_file = files[-1]
        print(f"[{source}] {len(files)} files on the hub — streaming the "
              f"last one: {last_file.rsplit('/', 1)[-1]}")

        ds = load_dataset(fmt, data_files=f"hf://{last_file}",
                          split="train", streaming=True)
        out_path = os.path.join(EVAL_DIR, f"{source}.jsonl")
        got_bytes, got_docs = 0, 0
        with open(out_path, "w") as out:
            for doc in ds:
                text = doc[text_key]
                n = len(text.encode("utf-8"))
                if not (MIN_DOC_BYTES <= n <= MAX_DOC_BYTES):
                    continue
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                got_bytes += n
                got_docs += 1
                if got_bytes >= budget:
                    break
        print(f"[{source}] frozen: {got_docs:,} docs, {got_bytes:,} bytes")
        manifest["sources"].append({
            "source": source, "file": last_file, "docs": got_docs,
            "bytes": got_bytes, "min_doc_bytes": MIN_DOC_BYTES,
            "max_doc_bytes": MAX_DOC_BYTES,
        })

    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, manifest_path)
    return manifest


def write_exclude_hashes():
    """md5 of every eval doc -> guard file for prepare_data.py.

    FineWeb2-HQ (the continued-pretraining source) is filtered from ALL of
    FineWeb2 and may contain the very documents this eval set was built
    from. prepare_data.py drops any doc whose hash appears here — training
    must never see the measuring stick. The file is small and belongs in
    git: the vast.ai instance has the repo but not eval_data/.
    Only written for the default EVAL_DIR, so an ad-hoc eval set (e.g. a
    smoke test) can't silently replace the committed guard.
    """
    if EVAL_DIR != "eval_data":
        return
    hashes = []
    for source, *_ in EVAL_SOURCES:
        with open(os.path.join(EVAL_DIR, f"{source}.jsonl")) as f:
            for line in f:
                text = json.loads(line)["text"]
                hashes.append(hashlib.md5(text.encode("utf-8")).hexdigest())
    with open("eval_exclude_hashes.json", "w") as f:
        json.dump({
            "purpose": "md5 of frozen eval docs — prepare_data.py excludes "
                       "these from training data (contamination guard)",
            "md5": hashes,
        }, f, indent=0)
    print(f"Contamination guard written: eval_exclude_hashes.json "
          f"({len(hashes)} hashes)")


def load_docs(source):
    docs = []
    with open(os.path.join(EVAL_DIR, f"{source}.jsonl")) as f:
        for line in f:
            docs.append(json.loads(line)["text"])
    return docs


# ---------------------------------------------------------------------------
# Scoring: sum of nats over a token stream, one prediction per token
# ---------------------------------------------------------------------------

@torch.no_grad()
def sum_nats(logits_fn, ids, device):
    """Total cross-entropy (nats) + number of predicted tokens.

    Windows of SEQ_LEN overlap by exactly 1 token: the last token of
    window k reappears as the first token of window k+1, where it serves
    as context only (a window predicts positions 1..T-1 from 0..T-2).
    Net effect: every token in the stream except ids[0] is predicted
    exactly once — no token skipped, none counted twice.
    """
    stride = SEQ_LEN - 1
    windows = [ids[i:i + SEQ_LEN] for i in range(0, len(ids) - 1, stride)]
    windows = [w for w in windows if len(w) >= 2]

    total_nats, total_pred = 0.0, 0
    full = [w for w in windows if len(w) == SEQ_LEN]
    tail = [w for w in windows if len(w) < SEQ_LEN]
    t0, done = time.time(), 0

    def run_batch(batch):
        nonlocal total_nats, total_pred, done
        x = torch.tensor(batch, dtype=torch.long, device=device)
        logits = logits_fn(x)  # [B, T, vocab]
        # reduction="sum": we want total nats, not a mean — means over
        # differently sized batches would weight tokens unequally.
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nats += loss.item()
        total_pred += x.size(0) * (x.size(1) - 1)
        done += x.size(0)
        if done % (BATCH_SIZE * 20) == 0:
            rate = total_pred / max(1e-9, time.time() - t0)
            print(f"    {done}/{len(windows)} windows "
                  f"({rate / 1000:.1f}k tok/s, "
                  f"running loss {total_nats / total_pred:.4f} nats/tok)")

    for i in range(0, len(full), BATCH_SIZE):
        run_batch(full[i:i + BATCH_SIZE])
    for w in tail:  # the final short window runs alone (no padding needed)
        run_batch([w])
    return total_nats, total_pred


def score_model(name, encode, eot_id, logits_fn, device, sources_docs):
    """bpb per source + combined. encode: raw text -> list of token ids."""
    print(f"\n=== {name} ===")
    results = {}
    for source, docs in sources_docs.items():
        raw_bytes = sum(len(d.encode("utf-8")) for d in docs)
        ids = []
        for d in docs:
            ids.extend(encode(d))
            ids.append(eot_id)  # doc boundary, exactly like in pretraining
        print(f"  [{source}] {len(docs):,} docs -> {len(ids):,} tokens "
              f"({raw_bytes / len(ids):.2f} bytes/token)")
        nats, pred = sum_nats(logits_fn, ids, device)
        results[source] = {
            "nats": nats, "pred_tokens": pred, "bytes": raw_bytes,
            "nats_per_token": nats / pred,
            "bpb": nats / (math.log(2) * raw_bytes),
        }
        print(f"  [{source}] {results[source]['nats_per_token']:.4f} nats/tok "
              f"-> {results[source]['bpb']:.4f} bits/byte")
    total_nats = sum(r["nats"] for r in results.values())
    total_bytes = sum(r["bytes"] for r in results.values())
    results["combined"] = {"bpb": total_nats / (math.log(2) * total_bytes)}
    print(f"  [combined] {results['combined']['bpb']:.4f} bits/byte")
    return results


# ---------------------------------------------------------------------------
# The two contestants
# ---------------------------------------------------------------------------

def load_ours(device):
    print(f"Loading {CHECKPOINT} on {device}...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = train.GPTDecoder(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"], dropout=0.0,
    ).to(device)
    sd = {k.removeprefix("_orig_mod."): v
          for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    return (f"ours ({CHECKPOINT}, "
            f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M)",
            enc.encode_ordinary, enc.eot_token, model)


def load_hf(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {HF_MODEL} on {device} (first run downloads ~1 GB)...")
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    # fp32 for both contestants: we are measuring hundredths of a bit,
    # so neither model gets rounding noise the other doesn't have.
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL, dtype=torch.float32).to(device)
    model.eval()

    def encode(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    n = sum(p.numel() for p in model.parameters())
    return (f"{HF_MODEL} ({n / 1e6:.0f}M)", encode, tok.eos_token_id,
            lambda x: model(x).logits)


def main():
    build_eval_set()
    write_exclude_hashes()
    if SKIP_OURS and SKIP_HF:
        return  # eval set + guard file only
    sources_docs = {s: load_docs(s) for s, *_ in EVAL_SOURCES}
    device = train.get_device()
    all_results = {}

    if not SKIP_OURS:
        name, encode, eot, model = load_ours(device)
        all_results[name] = score_model(
            name, encode, eot, model, device, sources_docs)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    if not SKIP_HF:
        name, encode, eot, logits_fn = load_hf(device)
        all_results[name] = score_model(
            name, encode, eot, logits_fn, device, sources_docs)

    print("\n" + "=" * 64)
    print(f"{'model':<40} {'fineweb':>7} {'code':>7} {'total':>7}  (bits/byte)")
    for name, res in all_results.items():
        print(f"{name:<40} "
              f"{res['fineweb']['bpb']:>7.4f} "
              f"{res['code']['bpb']:>7.4f} "
              f"{res['combined']['bpb']:>7.4f}")
    print("=" * 64)
    print("Lower = better compression of unseen text. This is the "
          "tokenizer-independent height line on the scaling map.")


if __name__ == "__main__":
    main()
