"""GPT-style transformer for German Wikipedia.

Decoder-only architecture, trained with causal language modeling.
Runs on MPS (Apple Silicon), CUDA, and CPU.
"""

import bisect
import glob
import json
import math
import os
import re
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import tiktoken
from datasets import load_dataset


# ============================================================
# Hyperparameters
# ============================================================
#
# ALL central hyperparameters can be overridden via environment variables
# (see _env below). That way the same code runs small locally (quick test
# on MPS) and large on a Vast.ai GPU (124M) WITHOUT editing the file —
# which keeps remote runs reproducible and diff-free.
#
#   Example:  D_MODEL=768 NUM_LAYERS=12 python train.py

def _env(name, default, cast=str):
    """Reads env var `name` and casts it; otherwise falls back to `default`."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    return cast(raw)


# Model — default = GPT-2-small class (~124M parameters).
D_MODEL = _env("D_MODEL", 768, int)
NUM_HEADS = _env("NUM_HEADS", 12, int)
D_FF = _env("D_FF", 2048, int)         # SwiGLU (3 matrices): (2/3)*4*d_model ~ 2048
NUM_LAYERS = _env("NUM_LAYERS", 12, int)
DROPOUT = _env("DROPOUT", 0.1, float)  # 0.0 would be Llama-style; 0.1 guards against
                                       # overfitting on our still data-limited corpus

# Training
SEQ_LENGTH = _env("SEQ_LENGTH", 512, int)   # 128 -> 512: longer context, more realistic
# RoPE buffer size (position tables). Deliberately LARGER than SEQ_LENGTH:
# RoPE encodes relative distances, so the model can extrapolate moderately
# beyond the training length — but only as far as the buffer reaches.
# 4096 gives headroom for extrapolation experiments and later long-context
# finetuning, and costs only a few MB (not part of the checkpoint).
ROPE_MAX_SEQ = _env("ROPE_MAX_SEQ", 4096, int)
BATCH_SIZE = _env("BATCH_SIZE", 16, int)    # micro-batch (what fits on one GPU)
GRAD_ACCUM_STEPS = _env("GRAD_ACCUM_STEPS", 3, int)  # effectively 16*3 = 48 sequences/update
LEARNING_RATE = _env("LEARNING_RATE", 6e-4, float)   # peak LR — the GPT-2-small standard
MIN_LR = _env("MIN_LR", 6e-5, float)        # LR floor after the full cosine decay
WEIGHT_DECAY = _env("WEIGHT_DECAY", 0.1, float)  # AdamW decay, applied to matrices only
WARMUP_STEPS = _env("WARMUP_STEPS", 500, int)    # linear warmup over N optimizer steps
GRAD_CLIP = _env("GRAD_CLIP", 1.0, float)        # global L2-norm bound for gradients
NUM_EPOCHS = _env("NUM_EPOCHS", 1, int)
USE_TORCH_COMPILE = _env("USE_TORCH_COMPILE", True, bool)  # big speedup on CUDA
USE_AMP = _env("USE_AMP", True, bool)       # bf16 autocast if the hardware supports it

# Data
NUM_ARTICLES = _env("NUM_ARTICLES", 400000, int)
VAL_FRACTION = _env("VAL_FRACTION", 0.02, float)  # fraction of tokens for the validation split
DATA_CACHE_DIR = _env("DATA_CACHE_DIR", "data_cache", str)  # token cache for re-runs
# Shard mode (large runs): path to shards/manifest.json produced by
# prepare_data.py. When set, the uint16 shards are read via memmap
# instead of the small in-RAM token cache above.
SHARD_MANIFEST = _env("SHARD_MANIFEST", "", str)
# DataLoader processes: default 4 on CUDA (shard reads run in parallel with
# the GPU), 0 on MPS/CPU — there, macOS' spawn start would COPY the in-RAM
# token tensor once per worker (expensive) while the gain is minimal.
NUM_WORKERS = _env("NUM_WORKERS", 4 if torch.cuda.is_available() else 0, int)

# Checkpoints
CHECKPOINT_DIR = "."
CHECKPOINT_EVERY_N_EPOCHS = 1     # full epochs are long now -> save every one
CHECKPOINT_EVERY_N_STEPS = 2000   # plus a rolling 'checkpoint_latest.pt'
AUTO_RESUME = True                # automatically load the newest checkpoint
RESUME_FROM = None                # overrides AUTO_RESUME when set
# Warm start (continued pretraining): load ONLY the model weights from this
# checkpoint — optimizer moments, step counter and LR schedule start fresh.
# This is different from resume: resume continues an interrupted run
# (including its cosine schedule); warm start begins a NEW run on top of
# finished weights, typically with a much lower peak LR. A found resume
# checkpoint takes precedence, so a crashed warm-start run resumes itself.
INIT_FROM = _env("INIT_FROM", "", str)
# Architecture marker in the checkpoint so old files can be detected
ARCH_VERSION = "pre_ln_tied_swiglu_rope_init_rms_v1"


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_dtype(device):
    """bf16 for autocast if the hardware supports it — otherwise None (= fp32).

    Why bf16 instead of fp16: same exponent range as fp32, i.e. no gradient
    underflow and NO GradScaler needed. The parameters stay fp32 (autocast
    only casts the ops in the forward pass); the optimizer step and
    gradient clipping run unchanged in fp32.

    On MPS, older chips (M1) can't do bf16 — instead of guessing by version
    we run a tiny probe computation and fall back cleanly to fp32.
    """
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    if device.type == "mps":
        try:
            x = torch.zeros(2, device=device, dtype=torch.bfloat16)
            (x * x).sum().item()
            return torch.bfloat16
        except Exception:
            return None
    return None


# ============================================================
# Data
# ============================================================

def get_or_build_tokens(num_articles, encoding, cache_dir=DATA_CACHE_DIR):
    """Cached tokenization: loads tokens from the cache, or builds + saves them.

    On the first call with a given num_articles: the Wikipedia snapshot is
    read article by article, batch-tokenized in chunks of 1000 (using
    tiktoken's multi-threaded encode_ordinary_batch), and the resulting
    tensor is saved to {cache_dir}/tokens-de-{num_articles}.pt.

    On subsequent calls only the tensor is loaded from disk — with 50k+
    articles that saves several minutes of tokenization per run start.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tokens-de-{num_articles}.pt")

    if os.path.exists(cache_path):
        print(f"Loading token cache: {cache_path}")
        tokens = torch.load(cache_path, weights_only=True)
        print(f"  Tokens: {len(tokens):,}")
        return tokens

    print(f"Tokenizing {num_articles} Wikipedia articles (no cache found)...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.de", split="train", streaming=False
    )
    ds = ds.select(range(min(num_articles, len(ds))))

    # Why NumPy chunks instead of a growing Python list:
    # `list.extend(tok_list)` builds a list of Python int objects — each
    # ~28 bytes. At 400M tokens that would be ~11 GB of RAM (and slow).
    # Instead we collect one compact np.int32 array per article (4 bytes/
    # token) and concatenate once at the end -> ~1.6 GB for 400M tokens.
    all_chunks = []
    running = 0
    chunk = 1000
    n_threads = os.cpu_count() or 4
    for i in range(0, len(ds), chunk):
        end = min(i + chunk, len(ds))
        texts = [a["text"] for a in ds.select(range(i, end))]
        # encode_ordinary_batch: multi-threaded, no special-token checking
        batched = encoding.encode_ordinary_batch(texts, num_threads=n_threads)
        for tok_list in batched:
            arr = np.asarray(tok_list, dtype=np.int32)
            all_chunks.append(arr)
            running += arr.size
        print(f"  {end:>6}/{len(ds)}: {running:,} tokens")

    tokens = torch.from_numpy(np.concatenate(all_chunks))
    mb = tokens.element_size() * tokens.numel() / 1024**2
    print(f"  Writing cache: {cache_path} ({mb:.1f} MB)")
    torch.save(tokens, cache_path)
    return tokens


class TextDataset(Dataset):
    """Next-token prediction:
        Input  : tokens [t0, t1, ..., t_{n-1}]
        Target : tokens [t1, t2, ..., t_n]
    """

    def __init__(self, tokens, seq_length=128):
        self.tokens = tokens
        self.seq_length = seq_length
        self.num_sequences = (len(tokens) - 1) // seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        # .long() instead of torch.tensor(slice): torch.tensor() on a tensor
        # copies unnecessarily (and PyTorch warns about it). .long() makes
        # exactly one copy; the two return values are views into it.
        seq = self.tokens[start:end].long()
        return seq[:-1], seq[1:]


class ShardDataset(Dataset):
    """Aligned windows over memmapped uint16 shards (from prepare_data.py).

    Why memmap: 10 billion tokens are ~20 GB of uint16 — too much for RAM.
    np.memmap opens the file without loading it; the OS pages in only what
    is actually read. Random access on NVMe is far faster than the GPU can
    consume the batches anyway.

    Split strategy: the last val_fraction windows of EACH shard are
    validation. That way the val split has the same source mix as the
    training data (shards are source-pure), and the leak at the cut edge
    is at most one document per shard.

    Windows are seq_length-aligned as in TextDataset; at shard boundaries
    at most one window per shard is lost — negligible with 100M-token
    shards.
    """

    def __init__(self, manifest_path, seq_length, split="train", val_fraction=0.02):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("tokenizer") != "gpt2":
            raise RuntimeError(
                f"Shards were built with tokenizer '{manifest.get('tokenizer')}', "
                "but training expects 'gpt2'."
            )
        base = os.path.dirname(manifest_path)
        self.seq_length = seq_length
        self._paths = []
        self._starts = []       # first window of this shard (within the split)
        cum = [0]               # cumulative window count across the shards
        for shard in manifest["shards"]:
            n_windows = (shard["num_tokens"] - 1) // seq_length
            n_val = int(n_windows * val_fraction)
            n_train = n_windows - n_val
            start, count = (0, n_train) if split == "train" else (n_train, n_val)
            if count <= 0:
                continue
            self._paths.append(os.path.join(base, shard["file"]))
            self._starts.append(start)
            cum.append(cum[-1] + count)
        self._cum = cum
        self._mmaps = {}  # shard idx -> np.memmap, lazy per process

    def __len__(self):
        return self._cum[-1]

    def __getstate__(self):
        # DataLoader workers receive the dataset via pickle. An np.memmap
        # would then get serialized as a FULL array (the entire shard!) —
        # so drop the open handles; each worker lazily reopens them.
        state = self.__dict__.copy()
        state["_mmaps"] = {}
        return state

    def _tokens(self, shard_idx):
        mm = self._mmaps.get(shard_idx)
        if mm is None:
            mm = np.memmap(self._paths[shard_idx], dtype=np.uint16, mode="r")
            self._mmaps[shard_idx] = mm
        return mm

    def __getitem__(self, idx):
        s = bisect.bisect_right(self._cum, idx) - 1
        window = self._starts[s] + (idx - self._cum[s])
        a = window * self.seq_length
        chunk = self._tokens(s)[a : a + self.seq_length + 1]
        # astype copies from the mmap into regular RAM — needed so the
        # tensor doesn't point into the file (and for int64, which
        # embedding lookups expect).
        seq = torch.from_numpy(chunk.astype(np.int64))
        return seq[:-1], seq[1:]


# ============================================================
# Model building blocks
# ============================================================

class TokenEmbedding(nn.Module):
    """Token lookup, without any further scaling.

    Previously: `embedding(x) * sqrt(d_model)`. That comes from the original
    transformer (Vaswani 2017), where the multiplication was meant to match
    the embedding magnitude to the additive sinusoidal positional encoding
    (values in [-1, 1]).

    With our modern architecture the scaling is actually HARMFUL:
      - RoPE replaces the additive PE -> the magnitude matching is unnecessary.
      - Weight tying: the same matrix embeds tokens AND performs the final
        projection. Scaled inputs inflate the magnitude of the residual
        stream, which after norm_f leads to very large initial logits and
        thus an absurdly high initial loss (484 instead of log(50257)).

    Hence: no more scaling. The modern LLM recipe (GPT-2, Llama).
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) — Llama / GPT-NeoX style.

    Instead of *adding* a positional encoding, a *rotation* is applied
    to the query and key vectors — depending on their position in the
    sequence.

    Key property: the dot product q_m * k_n after RoPE depends only on
    the *relative* position (m - n). So the model automatically learns
    position-relative attention, without us having to carry absolute
    positions around. Practical consequences:
      - Better extrapolation to sequences longer than trained on.
      - No embedding parameters needed for positions.
      - Standard in Llama, Mistral, Qwen, GPT-NeoX, PaLM-2, ...

    Mathematically, RoPE acts on each dimension pair (x_i, x_{i+d/2})
    as a 2D rotation by angle m * theta_i. The theta_i follow a
    geometric progression:
        theta_i = base^(-2i / d_head)   for i = 0..d_head/2-1
    -> small i: high frequency (local position),
       large i: low frequency (global position).
    """

    def __init__(self, d_head, max_seq_len=2048, base=10000.0):
        super().__init__()
        # Inverse frequencies: one value per dimension pair. Shape: [d_head/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))

        # angle(position m, pair i) = m * theta_i.  Shape: [T, d_head/2]
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)

        # persistent=False -> not in the state_dict (recomputable at any time)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, seq_len, offset=0):
        # offset: absolute position of the first query — during cached
        # generation the "sequence" handed to attention is just the newest
        # token, but RoPE must rotate it by its ABSOLUTE position.
        return (self.cos[offset:offset + seq_len],
                self.sin[offset:offset + seq_len])


def apply_rotary(x, cos, sin):
    """Apply the RoPE rotation to x (GPT-NeoX / Llama convention).

    x:        [B, H, T, d_head]
    cos, sin: [T, d_head/2]

    Convention: the first d_head/2 dimensions are rotated against the
    second d_head/2 dimensions. Mathematically equivalent to the original
    RoPE definition with adjacent pairs (0,1),(2,3),..., but cleaner in
    code because chunk()/cat() keep the memory contiguous.

        [new_x1]   [cos  -sin] [x1]
        [new_x2] = [sin   cos] [x2]
    """
    x1, x2 = x.chunk(2, dim=-1)              # each [B, H, T, d_head/2]
    cos = cos.unsqueeze(0).unsqueeze(0)      # [1, 1, T, d_head/2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    REFERENCE IMPLEMENTATION — the training path now uses
    F.scaled_dot_product_attention (fused kernel, Flash-Attention style).
    This explicit version shows what happens inside it, and can be used
    when you want to visualize the attention weights (the fused kernel
    never exposes them).

    It materializes the full [B, H, T, T] score matrix — fine at T=128,
    but the memory bottleneck for long sequences.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # -inf instead of -1e9: -1e9 would overflow in fp16/bf16.
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=2048):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # RoPE acts per head, i.e. on d_k-dimensional vectors.
        self.rope = RotaryEmbedding(self.d_k, max_seq_len=max_seq_len)

        # Dropout on the attention WEIGHTS (GPT-2: "attn_pdrop") —
        # passed directly to the fused SDPA kernel. The dropout on the
        # block OUTPUT ("resid_pdrop") lives in the DecoderBlock;
        # previously it was mistakenly in both places (double, ~0.19).
        self.attn_dropout = dropout

    def split_heads(self, x):
        # [B, T, D] -> [B, H, T, d_k]
        B, T, _ = x.size()
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # [B, H, T, d_k] -> [B, T, D]
        B, _, T, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, x):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Apply RoPE to Q and K (NOT to V — V carries content, not
        # position; only the attention scores need position).
        cos, sin = self.rope(Q.size(2))
        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        # Fused SDPA kernel (Flash-Attention style): never materializes
        # the [B, H, T, T] score matrix -> less memory, faster.
        # is_causal=True replaces our manual triangular mask.
        # dropout_p must be set to 0 manually in eval() — the functional
        # API doesn't know the module's train/eval state.
        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = self.combine_heads(out)
        return self.W_o(out)

    def forward_with_cache(self, x, kv_cache=None, pos_offset=0):
        """Inference-only twin of forward() with a KV cache.

        The observation that makes fast autoregressive inference possible:
        while generating, the K and V vectors of all PREVIOUS tokens never
        change — only the newest token contributes a new Q, K and V. So we
        compute projections only for the new token(s), append K and V to
        the cache, and let the new query attend over the whole cached
        history. Cost per generated token: O(T) instead of O(T^2).

        kv_cache:   (K, V) from previous calls, each [B, H, T_past, d_k];
                    None on the first (prefill) call.
        pos_offset: absolute position of x's first token in the sequence.

        Supports exactly the two shapes generate() produces: prefill
        (kv_cache=None, many query positions, causal mask needed) and
        decode (one query position, which may see ALL cached keys — for a
        single trailing query, causality needs no mask at all).

        Returns (out, (K, V)) with the updated cache.
        """
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        cos, sin = self.rope(Q.size(2), offset=pos_offset)
        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=Q.size(2) > 1,   # prefill: mask; decode (T=1): none
            dropout_p=0.0,             # cache path is inference-only
        )
        return self.W_o(self.combine_heads(out)), (K, V)


class FeedForward(nn.Module):
    """Classic two-layer FFN — activation is now GELU instead of ReLU.

    GELU is the variant used in GPT-2 / BERT:
        GELU(x) = x * Phi(x)         (Phi = standard normal CDF)
    Advantages over ReLU:
      - Smooth and differentiable everywhere (no "hard kink" even for x<0).
      - Produces a small, non-zero output for slightly negative inputs
        -> fewer "dead" neurons, better gradient flow.
      - Empirically consistently a tiny bit better in transformers.

    This class stays in as a reference baseline — but the DecoderBlock
    now uses SwiGLU (see below).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    """SwiGLU FFN — Llama / Mistral / PaLM style.

    Classic FFN:  out = W2( act(W1 x) )                       2 matrices
    SwiGLU:       out = W_down( SiLU(W_gate x) (*) W_up x )   3 matrices

    The crucial difference: a second parallel path (W_up) and an
    element-wise multiplication (*). The SiLU path acts as a "gate":
    it decides per position and per feature how much of the value
    path (W_up) is let through.

      SiLU(x) = x * sigmoid(x)   (also called "Swish")

    Why this helps: for every token, the model can dynamically
    choose which features are active in the FFN. Empirically clearly
    better than GELU at the same parameter count, and standard in
    all modern LLMs (Llama 2/3, Mistral, Mixtral, PaLM, ...).

    Parameter cost: 3 matrices instead of 2. Llama therefore reduces
    d_ff to about (2/3) * 4 * d_model to keep the parameter count
    constant. We leave d_ff at 2048 for now (-> about +50 % FFN params
    compared to the GELU FFN); if needed we can lower D_FF to 1408.

    No dropout of its own anymore: the residual dropout lives centrally
    in the DecoderBlock — previously we dropped here AND there (double).
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        # bias=False is the Llama convention; with weight tying & RMSNorm
        # the biases don't buy any model capacity anyway.
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(gated)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019).

    A simplification of LayerNorm:
      LayerNorm: y = gamma * (x - mean(x)) / std(x) + beta    (4 stats, 2 params)
      RMSNorm:   y = gamma * x / RMS(x)                       (1 stat,  1 param)

    Where RMS(x) = sqrt(mean(x^2)) is computed over the last dimension.

    What is dropped:
      - Mean subtraction ("re-centering"). Empirically dispensable in transformers.
      - Bias (beta). Can be absorbed by subsequent linear layers.

    What remains:
      - Magnitude normalization. Keeps forward/backward stable.
      - Learned scaling (gamma).

    Benefits: ~10-20 % fewer operations, equally good (often marginally
    better) training dynamics. Standard in Llama, Mistral, Qwen, Gemma, Falcon.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # mean of squares over the last dimension; rsqrt = 1/sqrt
        # (a single hardware instruction, faster than a separate division)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(ms + self.eps)
        return self.weight * x_normalized


class DecoderBlock(nn.Module):
    """Transformer decoder block with pre-layer norm.

    Post-LN (original):  x = LN( x + SubLayer(x) )
    Pre-LN  (GPT-2+):    x = x + SubLayer( LN(x) )

    Pre-LN keeps the residual path un-normalized, which makes deep
    transformers stably trainable without LR warmup.

    Dropout placement (GPT-2 scheme): exactly ONE residual dropout per
    sublayer, right before the addition onto the residual stream. The
    sublayers themselves no longer drop their output — previously we
    dropped twice (effectively ~0.19 instead of 0.1).
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, max_seq_len=4096):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout,
                                            max_seq_len=max_seq_len)
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

    def forward_with_cache(self, x, kv_cache=None, pos_offset=0):
        """Inference-only twin of forward() — no dropout, threads the
        KV cache through the attention sublayer."""
        attn_out, new_kv = self.attention.forward_with_cache(
            self.norm1(x), kv_cache, pos_offset)
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x, new_kv


def create_causal_mask(seq_len, device):
    """Lower-triangular mask: position i sees only positions <= i.

    REFERENCE — only relevant for the explicit scaled_dot_product_attention
    above. The training path uses is_causal=True in the fused kernel, so
    the mask never has to exist as a tensor.
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


class GPTDecoder(nn.Module):
    """GPT-style decoder-only transformer (pre-LN + weight tying)."""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 dropout=0.1, max_seq_len=4096):
        super().__init__()
        self.num_layers = num_layers

        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])

        # Pre-LN needs a final norm before the LM head, otherwise the
        # residual stream can grow without bound across the layers.
        self.norm_f = RMSNorm(d_model)

        # bias=False because we share the weights with the embedding matrix.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: the same [vocab_size, d_model] matrix serves both
        # for embedding (token ID -> vector) and for prediction (vector ->
        # logits over the vocab). Saves ~25M parameters and usually
        # improves sample quality visibly.
        self.lm_head.weight = self.embedding.embedding.weight

        self.dropout = nn.Dropout(dropout)

        # GPT-2 init recipe — see _init_weights.
        # Order matters: tie first, then init, so that both references
        # to the same parameter instance see the same initial sample.
        self.apply(self._init_weights)

        # Scaled init for the "output" projections of each residual branch:
        # with pre-LN, the activations would otherwise grow linearly per
        # block, because every residual gets added on. We divide the
        # initial magnitude of these projections by sqrt(2 * num_layers) —
        # the standard trick from the GPT-2 paper. Applies to W_o
        # (attention output) and w_down (SwiGLU output).
        proj_std = 0.02 / math.sqrt(2 * num_layers)
        for block in self.decoder_blocks:
            nn.init.normal_(block.attention.W_o.weight, mean=0.0, std=proj_std)
            nn.init.normal_(block.feed_forward.w_down.weight, mean=0.0, std=proj_std)

    @staticmethod
    def _init_weights(module):
        """GPT-2 init scheme: all weighted layers ~ N(0, 0.02^2), biases = 0.

        Why 0.02? An empirical value from the GPT-2 paper, well calibrated
        for LayerNorm-based transformers. It ensures that:
          - Initial activations stay in a reasonable range.
          - Initial logits are small (~N(0, sqrt(d_model)*0.02) = ~0.45),
            i.e. softmax nearly uniform -> initial loss ≈ log(vocab_size).
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0,
                 top_k=None, top_p=None, repetition_penalty=1.0,
                 eos_token_id=None, use_cache=True):
        """Autoregressive sampling with modern decoding strategies.

        temperature: scales logits before softmax. <1 sharper, >1 flatter.
        top_k:       keep only the k most likely tokens.
        top_p:       nucleus sampling — keep tokens whose cumulative
                     probability is <= p.
        repetition_penalty:
                     >1.0 lowers the probability of tokens that are already
                     in the context (against "Stadt im Stadt im Stadt").
        eos_token_id: stops generation as soon as this token is sampled
                     (batch=1). A base model practically never produces it —
                     only SFT teaches the model to signal "done". That is
                     why the parameter is optional.
        use_cache:   reuse the K/V vectors of previous tokens (KV cache)
                     instead of re-running the full forward pass over the
                     whole sequence for every new token. Same math, same
                     output distribution — per-step cost drops from O(T^2)
                     to O(T). False = the naive path, kept for comparison.

        Order: rep_penalty -> /temperature -> top_k -> top_p -> sample.
        """
        # Remember the training mode and restore it at the end — otherwise
        # a generate() call would have the side effect of permanently
        # disabling dropout if the caller keeps training.
        was_training = self.training
        self.eval()
        # The RoPE tables are the hard ceiling for the sequence length —
        # beyond them there are no position angles left.
        rope_max = self.decoder_blocks[0].attention.rope.cos.size(0)
        caches = [None] * self.num_layers
        for step in range(max_new_tokens):
            if not use_cache:
                # Naive path: full forward pass over the ENTIRE sequence,
                # throwing away everything but the last position's logits.
                logits = self.forward(input_ids)[:, -1, :]
            elif step == 0:
                # Prefill: one full pass over the prompt fills the cache.
                x = self.dropout(self.embedding(input_ids))  # no-op in eval
                for i, block in enumerate(self.decoder_blocks):
                    x, caches[i] = block.forward_with_cache(x, None, 0)
                logits = self.lm_head(self.norm_f(x[:, -1:, :]))[:, -1, :]
            else:
                # Decode: only the newest token runs through the model;
                # attention reads everything older from the cache.
                pos = input_ids.size(1) - 1
                if pos >= rope_max:
                    break
                x = self.embedding(input_ids[:, -1:])
                for i, block in enumerate(self.decoder_blocks):
                    x, caches[i] = block.forward_with_cache(x, caches[i], pos)
                logits = self.lm_head(self.norm_f(x))[:, -1, :]

            # 1. Repetition penalty: dampen the logits of tokens already seen.
            # HuggingFace convention: positive logits / penalty, negative * penalty
            # (both push the logit toward minus infinity).
            if repetition_penalty != 1.0:
                for b in range(input_ids.size(0)):
                    seen = torch.unique(input_ids[b])
                    score = logits[b, seen]
                    logits[b, seen] = torch.where(
                        score < 0, score * repetition_penalty, score / repetition_penalty
                    )

            # 2. Temperature
            logits = logits / temperature

            # 3. Top-k: set everything except the top-k logits to -inf.
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                threshold = topk_vals[..., -1, None]  # smallest top-k value
                logits = torch.where(logits < threshold,
                                     torch.full_like(logits, float("-inf")),
                                     logits)

            # 4. Top-p (nucleus): sort descending, keep the minimal set of
            # tokens whose cumulative prob is >= top_p.
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                # Mark tokens whose cumulative prob exceeds top_p.
                remove_sorted = cumulative > top_p
                # Shift right — still keep the first token that crosses the line.
                remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
                remove_sorted[..., 0] = False
                # Map the mask back onto the original indices.
                remove = torch.zeros_like(logits, dtype=torch.bool)
                remove.scatter_(-1, sorted_idx, remove_sorted)
                logits = logits.masked_fill(remove, float("-inf"))

            # 5. Softmax + multinomial sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if (eos_token_id is not None and input_ids.size(0) == 1
                    and next_token.item() == eos_token_id):
                break
        if was_training:
            self.train()
        return input_ids


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(model, optimizer, epoch, global_step, loss, filepath, config):
    # torch.compile wraps the model in an OptimizedModule — whose
    # state_dict keys get an '_orig_mod.' prefix. We always save the
    # uncompiled original so the checkpoint format doesn't depend on
    # whether training ran with or without compile.
    raw_model = getattr(model, "_orig_mod", model)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
        "arch_version": ARCH_VERSION,
    }, filepath)
    print(f"  Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer, device):
    ckpt = torch.load(filepath, map_location=device)

    # Detect incompatibility hard, instead of carrying on with a randomly
    # initialized norm_f / incorrectly tied lm_head weights.
    ckpt_arch = ckpt.get("arch_version", "legacy_post_ln")
    if ckpt_arch != ARCH_VERSION:
        raise RuntimeError(
            f"Checkpoint '{filepath}' has arch_version='{ckpt_arch}', "
            f"the current architecture is '{ARCH_VERSION}'. "
            "Delete it, rename it, or migrate it by hand."
        )

    # Older checkpoints were saved from the compiled model
    # ('_orig_mod.' prefix in the keys) — normalize when loading.
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {
        k.removeprefix("_orig_mod."): v
        for k, v in ckpt["model_state_dict"].items()
    }
    raw_model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError) as e:
            # E.g. a checkpoint from plain Adam (1 param group) while the
            # current optimizer is AdamW with decay/no-decay groups. The
            # model weights are loaded; only the Adam moments start fresh.
            print(f"  Optimizer state incompatible — starting with a fresh "
                  f"optimizer ({type(e).__name__}: {e})")
    # Exported weight files (export_weights.py) carry config + weights but
    # no training metadata — tolerate that, a warm start needs none of it.
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    loss = ckpt.get("loss", ckpt.get("train_loss", float("nan")))
    print(f"  Checkpoint loaded: epoch {epoch}, step {global_step}, loss {loss:.4f}")
    return epoch, global_step, loss


def find_latest_checkpoint(directory):
    """Finds the most recent checkpoint (epoch or rolling latest).

    Considers:
      - checkpoint_epoch_N.pt (written at the end of an epoch)
      - checkpoint_latest.pt  (rolling mid-epoch, always overwritten)

    Selected by mtime — the checkpoint written last wins, regardless
    of type. None if nothing is found.
    """
    paths = glob.glob(os.path.join(directory, "checkpoint_epoch_*.pt"))
    latest_path = os.path.join(directory, "checkpoint_latest.pt")
    if os.path.exists(latest_path):
        paths.append(latest_path)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def checkpoint_dims_match(filepath):
    """True if the checkpoint has the same model dimensions as the
    current configuration.

    Why this is needed: the ARCH_VERSION check only catches ARCHITECTURE
    changes (e.g. LayerNorm -> RMSNorm), not SIZE changes within the same
    architecture. If you scale from 51M to 124M with an old checkpoint
    still lying around, load_state_dict would otherwise die with a cryptic
    'size mismatch'. Here we check up front and cleanly start fresh.
    """
    try:
        cfg = torch.load(filepath, map_location="cpu").get("config", {})
    except Exception:
        return False
    return all(cfg.get(k) == v for k, v in (
        ("d_model", D_MODEL), ("num_layers", NUM_LAYERS),
        ("num_heads", NUM_HEADS), ("d_ff", D_FF),
    ))


# ============================================================
# Training
# ============================================================

def configure_optimizer(model, lr, weight_decay, betas=(0.9, 0.95)):
    """AdamW with selective weight decay — the GPT-2/Llama recipe.

    Why AdamW instead of Adam: with Adam, L2 regularization interacts with
    the per-parameter scaling of the gradients and therefore doesn't act
    as a true decay (Loshchilov & Hutter 2019). AdamW decouples the decay
    from the gradient update — only then does weight_decay work as intended.

    Decay only on matrices (dim >= 2): linear and embedding weights.
    Do NOT decay RMSNorm gains and biases (dim 1) — they scale
    activations; a pull toward 0 would only disturb training there
    without regularizing. (Convention from GPT-2/nanoGPT.)

    betas=(0.9, 0.95): the lower beta2 (instead of 0.999) reacts faster
    to changes in gradient variance — standard in LLM training
    (GPT-3, Llama).

    named_parameters() deduplicates shared parameters, so the tied
    embedding/lm_head weight ends up exactly once in the list.
    """
    decay_params = []
    no_decay_params = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay_params if p.dim() >= 2 else no_decay_params).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr, betas=betas,
    )


def get_lr(step, warmup_steps, max_steps, base_lr, min_lr):
    """LR schedule: linear warmup followed by cosine decay.

    Warmup (step < warmup_steps): linearly from ~0 up to base_lr.
    Cosine (afterwards): from base_lr down to min_lr in a cosine shape
    over the remainder.

    Why warmup? Adam needs a few steps to build up reasonable moment
    estimates — before that, a full LR can cause loss explosions,
    especially with random initialization.

    Why cosine decay? Empirically better than step decay / linear decay,
    because the model spends a long time exploring with a high LR at the
    start and converges gently at the end. Standard in Llama, GPT-3, and
    almost all modern LLMs.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_epoch(model, dataloader, criterion, optimizer, device, vocab_size,
                global_step, max_steps, warmup_steps, base_lr, min_lr, grad_clip,
                grad_accum_steps=1, amp_dtype=None,
                save_step_interval=None, save_callback=None):
    """Trains one epoch with gradient accumulation.

    Gradient accumulation: we accumulate the gradients from `grad_accum_steps`
    micro-batches before taking ONE optimizer step. Effectively we thereby
    train with (batch_size * grad_accum_steps) sequences per update, without
    needing the memory for such a large batch all at once. Why this matters:
    larger batches yield less noisy gradients (good for larger models), but
    GPU memory caps the REAL batch size. Accumulation decouples the two.

    The `loss / grad_accum_steps` trick: PyTorch ADDS gradients across
    multiple backward() calls. So that the sum equals the MEAN over the
    large batch (not its sum), we scale each partial loss down before
    its backward().

    global_step counts OPTIMIZER steps (not micro-batches) — only then does
    the LR schedule, which thinks in optimizer steps, line up.

    save_step_interval: every N optimizer steps, save_callback(global_step, loss).
    amp_dtype:          e.g. torch.bfloat16 -> forward+loss under autocast.
    """
    model.train()
    total_loss = torch.zeros((), device=device)  # accumulate on the device
    n_microbatches = 0
    last_lr = 0.0
    amp_ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype else nullcontext())

    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        with amp_ctx:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

        # Scale down -> the summed grads == mean over the large batch
        (loss / grad_accum_steps).backward()

        # .detach() instead of .item(): .item() forces a CPU<->GPU sync. We
        # accumulate as a tensor on the device and only sync when logging.
        total_loss += loss.detach()
        n_microbatches += 1

        # Take an optimizer step only once a full accumulation window is done.
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Set the LR for this optimizer step
            lr = get_lr(global_step, warmup_steps, max_steps, base_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            last_lr = lr

            # Gradient clipping: caps the global L2 norm of all gradients at
            # grad_clip. Prevents an outlier batch from kicking the model
            # into a bad region ("loss spike").
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 20 == 0:
                print(f"  Step {global_step}/{max_steps}, "
                      f"loss={loss.item():.4f}, lr={lr:.2e}")

            # Rolling mid-epoch checkpoint
            if save_step_interval and save_callback and global_step % save_step_interval == 0:
                save_callback(global_step, loss.item())

    return (total_loss / max(1, n_microbatches)).item(), global_step, last_lr


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, vocab_size, amp_dtype=None):
    """Mean loss on the validation set.

    eval() mode disables dropout, and @torch.no_grad() saves memory
    and time because no gradients are built.
    """
    model.eval()
    total_loss = torch.zeros((), device=device)
    n_batches = 0
    amp_ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype else nullcontext())
    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        with amp_ctx:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        total_loss += loss.detach()
        n_batches += 1
    return (total_loss / max(1, n_batches)).item()


def generate_samples(model, encoding, device, prompts, max_new_tokens=20,
                     temperature=0.8, top_p=0.9, repetition_penalty=1.2):
    """Generates samples with conservative, low-repetition sampling."""
    for prompt in prompts:
        start = encoding.encode(prompt)
        input_ids = torch.tensor([start], device=device)
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        text = encoding.decode(out_ids[0].cpu().tolist())
        print(f"  '{text}'")


# ============================================================
# Main
# ============================================================

def main():
    device = get_device()
    amp_dtype = get_amp_dtype(device) if USE_AMP else None
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Autocast: {amp_dtype if amp_dtype else 'off (fp32)'}\n")

    # --- Data ---
    encoding = tiktoken.get_encoding("gpt2")
    vocab_size = encoding.n_vocab
    print(f"  Vocab: {vocab_size}")

    if SHARD_MANIFEST:
        # Shard mode (large runs): memmap over prepare_data.py shards.
        if not os.path.exists(SHARD_MANIFEST):
            raise FileNotFoundError(
                f"SHARD_MANIFEST='{SHARD_MANIFEST}' does not exist — "
                "run `python prepare_data.py` first."
            )
        train_ds = ShardDataset(SHARD_MANIFEST, SEQ_LENGTH, "train", VAL_FRACTION)
        val_ds = ShardDataset(SHARD_MANIFEST, SEQ_LENGTH, "val", VAL_FRACTION)
        n_tok = (len(train_ds) + len(val_ds)) * SEQ_LENGTH
        print(f"  Shard mode: {SHARD_MANIFEST} (~{n_tok / 1e9:.2f} billion tokens)")
        print(f"  Split: {len(train_ds):,} train / {len(val_ds):,} val windows\n")
    else:
        # Legacy mode (small runs): a single token tensor in RAM.
        tokens = get_or_build_tokens(NUM_ARTICLES, encoding)

        # Train / val split — the last VAL_FRACTION of the tokens become
        # validation. (Sequential split, NOT shuffled: Wikipedia articles
        # are concatenated sequentially in the stream; a random token
        # shuffle would leak tokens from the same article into both splits.)
        split_idx = int(len(tokens) * (1 - VAL_FRACTION))
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        print(f"  Split: {len(train_tokens):,} train / {len(val_tokens):,} val\n")
        train_ds = TextDataset(train_tokens, seq_length=SEQ_LENGTH)
        val_ds = TextDataset(val_tokens, seq_length=SEQ_LENGTH)

    # pin_memory (CUDA only): batches land in page-locked RAM, so the
    # host->GPU copy then runs DMA-asynchronously. num_workers>0 offloads
    # the reading/converting into separate processes so the GPU never
    # waits for data — important in shard mode, because there every batch
    # is read from disk.
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            drop_last=False, num_workers=NUM_WORKERS, pin_memory=pin)
    print(f"DataLoader: {len(train_loader)} train batches, {len(val_loader)} val batches, "
          f"{NUM_WORKERS} workers\n")

    # --- Model ---
    model = GPTDecoder(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        # Buffer at least as large as the training length — otherwise RoPE
        # would index out of range on the first batch.
        max_seq_len=max(ROPE_MAX_SEQ, SEQ_LENGTH),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # torch.compile: fuses ops, eliminates sync points, can bring a
    # 30-100 % speedup on MPS. The first forward pass is slow (compilation),
    # faster afterwards. Fail-safe: fall back to eager on a crash.
    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            print("torch.compile: active (first batch runs slower due to JIT)")
        except Exception as e:
            print(f"torch.compile failed, eager mode: {e}")
    print()

    # --- Optimizer + Loss ---
    criterion = nn.CrossEntropyLoss()
    optimizer = configure_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)

    # --- Optional: Resume ---
    start_epoch = 0
    global_step = 0
    resume_path = RESUME_FROM
    if resume_path is None and AUTO_RESUME:
        resume_path = find_latest_checkpoint(CHECKPOINT_DIR)
        if resume_path:
            print(f"Auto-resume: '{resume_path}'")

    if resume_path and os.path.exists(resume_path) and not checkpoint_dims_match(resume_path):
        print(f"  Checkpoint '{resume_path}' has different model dimensions than "
              f"the current configuration ({D_MODEL}d / {NUM_LAYERS}L) — "
              f"ignoring it, starting fresh.")
        resume_path = None

    if resume_path and os.path.exists(resume_path):
        start_epoch, global_step, _ = load_checkpoint(resume_path, model, optimizer, device)
    elif INIT_FROM:
        # Warm start: weights only. checkpoint_dims_match would silently
        # fall back to random init on a size mismatch — for an explicit
        # warm start that must be a hard error instead.
        if not checkpoint_dims_match(INIT_FROM):
            raise RuntimeError(
                f"INIT_FROM='{INIT_FROM}' does not match the current model "
                f"dimensions ({D_MODEL}d / {NUM_LAYERS}L) — refusing to "
                "warm-start from a different-sized model."
            )
        load_checkpoint(INIT_FROM, model, None, device)
        print(f"Warm start from '{INIT_FROM}': weights loaded, "
              "optimizer/schedule/step counter start fresh.")
    elif resume_path:
        print(f"  (No checkpoint '{resume_path}' found — starting fresh.)")
    else:
        print("Starting fresh training (no checkpoint).")

    # --- Training loop ---
    # Also record the training hyperparameters, so that later you can
    # tell from a checkpoint which settings produced it.
    config = dict(
        vocab_size=vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
        d_ff=D_FF, num_layers=NUM_LAYERS, seq_length=SEQ_LENGTH,
        dropout=DROPOUT, batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE, min_lr=MIN_LR, weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP, num_articles=NUM_ARTICLES,
        amp_dtype=str(amp_dtype) if amp_dtype else None,
    )
    # German prompts — the model only speaks German.
    test_prompts = ["Die Geschichte", "Im Jahr", "Deutschland ist"]
    # max_steps counts OPTIMIZER steps: floor(#batches / grad_accum) per epoch.
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    max_steps = NUM_EPOCHS * steps_per_epoch
    eff_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    print(f"Effective batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = {eff_batch} "
          f"sequences ({eff_batch * SEQ_LENGTH:,} tokens/step)")
    print(f"Training: max_steps={max_steps}, steps/epoch={steps_per_epoch}, "
          f"warmup_steps={WARMUP_STEPS}, peak_lr={LEARNING_RATE:.2e}, "
          f"min_lr={MIN_LR:.2e}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Rolling mid-epoch checkpoint callback. Writes 'checkpoint_latest.pt'
        # with the CURRENT epoch — a resume restarts this epoch from the top.
        # Honest limitations of this simple scheme:
        #   - Everything that had already run within the epoch is re-done
        #     (in the worst case almost a full epoch, not just
        #     save_step_interval batches) — but the weights/steps are kept.
        #   - global_step keeps counting -> the cosine schedule shifts
        #     relative to the data.
        #   - The DataLoader shuffle has no fixed seed -> the resumed run
        #     sees a different batch order.
        # An exact resume would require saving sampler state + RNG state.
        def save_rolling(step, current_loss, epoch=epoch):
            save_checkpoint(
                model, optimizer, epoch, step, current_loss,
                "checkpoint_latest.pt", config,
            )

        train_loss, global_step, last_lr = train_epoch(
            model, train_loader, criterion, optimizer, device, vocab_size,
            global_step, max_steps, WARMUP_STEPS, LEARNING_RATE, MIN_LR, GRAD_CLIP,
            grad_accum_steps=GRAD_ACCUM_STEPS, amp_dtype=amp_dtype,
            save_step_interval=CHECKPOINT_EVERY_N_STEPS, save_callback=save_rolling,
        )
        val_loss = evaluate(model, val_loader, criterion, device, vocab_size,
                            amp_dtype=amp_dtype)

        # Perplexity = exp(cross-entropy). A more readable metric than loss:
        # ppl=N means "the model is on average uncertain between N tokens".
        train_ppl = math.exp(min(train_loss, 20))   # clip to avoid overflow
        val_ppl = math.exp(min(val_loss, 20))
        print(f"-> train: loss={train_loss:.4f}  ppl={train_ppl:.1f}")
        print(f"-> val:   loss={val_loss:.4f}  ppl={val_ppl:.1f}   "
              f"(last lr={last_lr:.2e})")

        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, global_step, train_loss,
                f"checkpoint_epoch_{epoch + 1}.pt", config,
            )

        print("  Text Generation Test:")
        generate_samples(model, encoding, device, test_prompts)
        print("-" * 60)


if __name__ == "__main__":
    main()
