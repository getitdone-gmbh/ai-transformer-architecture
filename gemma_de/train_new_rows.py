"""Phase 2, step 2: teach Gemma its new tokens — and touch nothing else.

After transplant.py the 65,536 new rows hold the MEAN of their Gemma
pieces. That is a good guess for the input side and a poor one for the
output side (the tied LM head scores '▁Hauptstadt' as h · mean(rows)).
Here we train exactly those rows, with every other weight frozen:

    trainable   new rows of embed_tokens (= input AND output, tied)
                new rows of embed_tokens_per_layer (E2B/E4B only)
    frozen      all 262,144 original rows, every layer, every norm

Why that is the right first lever: Gemma's knowledge and skills live in
the layers and the old rows. With those frozen, nothing Gemma could do
before can get lost — the only question is how well the layers can USE
the new tokens. One lever, one answer.

HOW ONLY SOME ROWS TRAIN: the new rows live in separate fp32 parameters
(small enough for Adam: ~65k x (1536 + 8960)), and the embedding lookups
and the LM head are patched to read old rows from the frozen bf16 table and
new rows from these parameters. At the end they are written back into the
tables and saved as a normal model folder.

DATA: the phase-1 corpus mix (70% German, 20% Python, 10% English). The
non-German share matters even though the new tokens are German: it
teaches the new output rows to stay QUIET where they don't belong — an
untrained '▁Hauptstadt' row can steal probability in Python code.

LORA_RANK > 0 — THE SECOND LEVER. Rows-only training stalled 9% above the
original on German: the layers were trained on '▁Haupt' + 'stadt' over
two positions, and now see one '▁Hauptstadt'. The new rows alone cannot
move work between positions — the layers have to adapt a little. LoRA
lets them, cheaply and reversibly: every attention and FFN projection W
gets a trainable low-rank detour,

    W x  ->  W x + (alpha/r) * B (A x)      A: r x in, B: out x r, B = 0 at start

so at step 0 the model is exactly the transplanted one. W itself stays
frozen; at the end B @ A is merged into W and the saved model is a plain
Gemma again. Same data, same budget as the rows-only run, so the
difference between the two runs is LoRA and nothing else.

Usage:
    MODEL=~/gemma_de_work/models/E2B_B64k_init TOKENS=5e7 \\
    OUT=~/gemma_de_work/models/E2B_B64k_rows python gemma_de/train_new_rows.py
    LORA_RANK=16 OUT=~/gemma_de_work/models/E2B_B64k_lora ...   # + LoRA
"""

import fnmatch
import math
import os
import random
import shutil
import sys
import time
import types

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpb  # noqa: E402
from bpb import load, measure  # noqa: E402
from transplant import KEEP  # noqa: E402

N_OLD = 262_144   # Gemma 4's vocabulary; everything above is ours
MODEL = os.environ.get("MODEL", "~/gemma_de_work/models/E2B_B64k_init")
OUT = os.path.expanduser(
    os.environ.get("OUT", "~/gemma_de_work/models/E2B_B64k_rows"))
WORK_DIR = os.path.expanduser(os.environ.get("WORK_DIR", "~/gemma_de_work"))
TOKENS = float(os.environ.get("TOKENS", 5e7))
SEQ = int(os.environ.get("SEQ", 2048))
MICRO = int(os.environ.get("MICRO", 4))
ACCUM = int(os.environ.get("ACCUM", 4))
LR = float(os.environ.get("LR", 1e-3))
WARMUP = int(os.environ.get("WARMUP", 50))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 200))
MIX = [("de", 0.7), ("code", 0.2), ("en", 0.1)]
LORA_RANK = int(os.environ.get("LORA_RANK", 0))   # 0 = rows only
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", 32))
LORA_LR = float(os.environ.get("LORA_LR", 2e-4))
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


def docs(name):
    with open(os.path.join(WORK_DIR, f"{name}.txt"), encoding="utf-8") as f:
        buf = ""
        while chunk := f.read(1 << 22):
            parts = (buf + chunk).split("\0")
            buf = parts.pop()
            yield from parts


def batches(tok, bos, eos):
    """Packed SEQ-token rows, documents drawn by the MIX weights."""
    rng = random.Random(0)
    streams = {name: docs(name) for name, _ in MIX}
    names, weights = zip(*MIX)
    buf, rows = [], []
    while True:
        text = next(streams[rng.choices(names, weights)[0]])
        buf += [bos] + tok.encode(text, add_special_tokens=False).ids + [eos]
        while len(buf) >= SEQ:
            rows.append(buf[:SEQ])
            buf = buf[SEQ:]
            if len(rows) == MICRO:
                yield torch.tensor(rows, device="cuda")
                rows = []


def split_rows(model):
    """Move the new rows into fp32 parameters and patch the lookups."""
    emb = model.get_input_embeddings()
    tables = [emb]
    if hasattr(model, "get_per_layer_input_embeddings"):
        tables.append(model.get_per_layer_input_embeddings())
    for p in model.parameters():
        p.requires_grad_(False)

    def lookup(self, ids):
        new = ids >= N_OLD
        old = F.embedding(ids.clamp(max=N_OLD - 1), self.weight[:N_OLD])
        add = F.embedding((ids - N_OLD).clamp(min=0),
                          self.new_rows).to(old.dtype)
        out = torch.where(new[..., None], add, old)
        return out * self.embed_scale.to(out.dtype)

    for t in tables:
        t.new_rows = torch.nn.Parameter(t.weight[N_OLD:].float().clone())
        t.forward = types.MethodType(lookup, t)

    head = model.lm_head

    def logits(self, h):
        return torch.cat([F.linear(h, self.weight[:N_OLD]),
                          F.linear(h, emb.new_rows.to(h.dtype))], dim=-1)
    head.forward = types.MethodType(logits, head)
    return tables


class LoRA(torch.nn.Module):
    """A frozen Linear plus a trainable low-rank detour B @ A."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = torch.nn.Parameter(
            torch.randn(r, base.in_features, device=base.weight.device) /
            math.sqrt(base.in_features))
        self.B = torch.nn.Parameter(
            torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        detour = F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))
        return self.base(x) + self.scale * detour

    @torch.no_grad()
    def merged(self):
        self.base.weight += (self.scale * self.B @ self.A).to(
            self.base.weight.dtype)
        return self.base


def add_lora(model):
    wrapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if (child_name in LORA_TARGETS and ".language_model." in
                    f".{name}." and isinstance(child, torch.nn.Linear)):
                lora = LoRA(child, LORA_RANK, LORA_ALPHA)
                setattr(mod, child_name, lora)
                wrapped.append((mod, child_name, lora))
    return wrapped


@torch.no_grad()
def merge_rows(tables):
    for t in tables:
        t.weight[N_OLD:] = t.new_rows.to(t.weight.dtype)
        del t.new_rows   # else save_pretrained writes them as extra keys


def main():
    model, tok = load(MODEL)
    tables = split_rows(model)
    params = [t.new_rows for t in tables]
    groups = [dict(params=params, lr=LR)]
    wrapped = add_lora(model) if LORA_RANK else []
    if wrapped:
        lora_params = [p for _, _, m in wrapped for p in (m.A, m.B)]
        groups.append(dict(params=lora_params, lr=LORA_LR))
        params = params + lora_params
        print(f"LoRA r={LORA_RANK} on {len(wrapped)} projections: "
              f"{sum(p.numel() for p in lora_params) / 1e6:.1f}M params",
              flush=True)
    print(f"trainable: {sum(p.numel() for p in params) / 1e6:.0f}M params "
          f"of {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B",
          flush=True)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0)
    steps = int(TOKENS // (SEQ * MICRO * ACCUM))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        1, (s + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(
            math.pi * min(s, steps) / steps))))

    data = batches(tok, tok.token_to_id("<bos>"), tok.token_to_id("<eos>"))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    t0 = time.time()
    for step in range(steps):
        total = 0.0
        for _ in range(ACCUM):
            x = next(data)
            loss = model(input_ids=x, labels=x).loss / ACCUM
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            done = (step + 1) * SEQ * MICRO * ACCUM
            print(f"step {step:5d}/{steps}  loss {total:.4f}  lr "
                  f"{sched.get_last_lr()[0]:.2e}  "
                  f"{done / (time.time() - t0):,.0f} tok/s", flush=True)
        if EVAL_EVERY and step and step % EVAL_EVERY == 0:
            model.eval()
            bpb.MAX_DOCS = 100   # quick look; the full eval runs on the saved model
            measure(model, tok, f"step {step}")
            model.train()

    merge_rows(tables)
    for mod, child_name, lora in wrapped:
        setattr(mod, child_name, lora.merged())
    model.save_pretrained(OUT)
    src = os.path.expanduser(MODEL)
    for fname in os.listdir(src):
        if any(fnmatch.fnmatch(fname, pat) for pat in KEEP):
            shutil.copyfile(os.path.join(src, fname), os.path.join(OUT, fname))
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
