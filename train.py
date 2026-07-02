"""GPT-Style Transformer fuer deutsche Wikipedia.

Decoder-only Architektur, trainiert mit Causal Language Modeling.
Laeuft auf MPS (Apple Silicon), CUDA und CPU.
"""

import glob
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
# Hyperparameter
# ============================================================
#
# ALLE zentralen Hyperparameter sind per Umgebungsvariable ueberschreibbar
# (siehe _env unten). So laeuft derselbe Code lokal klein (schneller Test
# auf MPS) und auf einer Vast.ai-GPU gross (124M), OHNE die Datei zu
# editieren — Remote-Runs bleiben damit reproduzierbar und diff-frei.
#
#   Beispiel:  D_MODEL=768 NUM_LAYERS=12 python train.py

def _env(name, default, cast=str):
    """Liest ENV-Var `name`, castet sie, faellt sonst auf `default` zurueck."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    return cast(raw)


# Modell — Default = GPT-2-small-Klasse (~124M Parameter).
D_MODEL = _env("D_MODEL", 768, int)
NUM_HEADS = _env("NUM_HEADS", 12, int)
D_FF = _env("D_FF", 2048, int)         # SwiGLU (3 Matrizen): (2/3)*4*d_model ~ 2048
NUM_LAYERS = _env("NUM_LAYERS", 12, int)
DROPOUT = _env("DROPOUT", 0.1, float)  # 0.0 waere Llama-Style; 0.1 schuetzt bei
                                       # unserem noch daten-limitierten Korpus vor Overfit

# Training
SEQ_LENGTH = _env("SEQ_LENGTH", 512, int)   # 128 -> 512: laengerer Kontext, realistischer
BATCH_SIZE = _env("BATCH_SIZE", 16, int)    # Mikro-Batch (was auf eine GPU passt)
GRAD_ACCUM_STEPS = _env("GRAD_ACCUM_STEPS", 3, int)  # effektiv 16*3 = 48 Sequenzen/Update
LEARNING_RATE = _env("LEARNING_RATE", 6e-4, float)   # peak LR — GPT-2-small-Standard
MIN_LR = _env("MIN_LR", 6e-5, float)        # LR-Untergrenze nach komplettem Cosine-Decay
WEIGHT_DECAY = _env("WEIGHT_DECAY", 0.1, float)  # AdamW-Decay, nur auf Matrizen
WARMUP_STEPS = _env("WARMUP_STEPS", 500, int)    # linearer Warmup ueber N Optimizer-Steps
GRAD_CLIP = _env("GRAD_CLIP", 1.0, float)        # globale L2-Norm-Schranke fuer Gradienten
NUM_EPOCHS = _env("NUM_EPOCHS", 1, int)
USE_TORCH_COMPILE = _env("USE_TORCH_COMPILE", True, bool)  # auf CUDA grosser Speedup
USE_AMP = _env("USE_AMP", True, bool)       # bf16-Autocast wenn die Hardware es kann

# Daten
NUM_ARTICLES = _env("NUM_ARTICLES", 400000, int)
VAL_FRACTION = _env("VAL_FRACTION", 0.02, float)  # Anteil Tokens fuer Validation-Split
DATA_CACHE_DIR = _env("DATA_CACHE_DIR", "data_cache", str)  # Token-Cache fuer Re-Runs

# Checkpoints
CHECKPOINT_DIR = "."
CHECKPOINT_EVERY_N_EPOCHS = 1     # ganze Epochen sind jetzt lang -> jede speichern
CHECKPOINT_EVERY_N_STEPS = 2000   # zusaetzlich rolling 'checkpoint_latest.pt'
AUTO_RESUME = True                # neuesten Checkpoint automatisch laden
RESUME_FROM = None                # ueberschreibt AUTO_RESUME wenn gesetzt
# Architektur-Marker im Checkpoint, damit alte Files erkannt werden
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
    """bf16 fuer Autocast, wenn die Hardware es kann — sonst None (= fp32).

    Warum bf16 statt fp16: gleicher Exponenten-Bereich wie fp32, d.h. kein
    Gradient-Underflow und KEIN GradScaler noetig. Die Parameter bleiben
    fp32 (Autocast castet nur die Ops im Forward), Optimizer-Step und
    Gradient-Clipping laufen unveraendert in fp32.

    Auf MPS koennen aeltere Chips (M1) kein bf16 — statt Versions-Raterei
    machen wir eine kleine Probe-Rechnung und fallen sauber auf fp32 zurueck.
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
# Daten
# ============================================================

def get_or_build_tokens(num_articles, encoding, cache_dir=DATA_CACHE_DIR):
    """Cached Tokenisierung: laedt Tokens aus Cache, oder baut + speichert.

    Beim ersten Aufruf mit gegebenem num_articles: Wikipedia-Snapshot wird
    artikelweise gelesen, in Chunks von 1000 batched-tokenisiert (mit
    tiktoken's multi-threaded encode_ordinary_batch), und das resultierende
    Tensor in {cache_dir}/tokens-de-{num_articles}.pt gespeichert.

    Bei Folge-Aufrufen wird nur das Tensor von Disk geladen — spart bei
    50k+ Artikeln pro Run-Start mehrere Minuten Tokenisierung.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tokens-de-{num_articles}.pt")

    if os.path.exists(cache_path):
        print(f"Lade Token-Cache: {cache_path}")
        tokens = torch.load(cache_path, weights_only=True)
        print(f"  Tokens: {len(tokens):,}")
        return tokens

    print(f"Tokenisiere {num_articles} Wikipedia-Artikel (kein Cache vorhanden)...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.de", split="train", streaming=False
    )
    ds = ds.select(range(min(num_articles, len(ds))))

    # Warum NumPy-Chunks statt einer wachsenden Python-Liste:
    # `list.extend(tok_list)` baut eine Liste aus Python-int-Objekten — jedes
    # ~28 Bytes. Bei 400M Tokens waeren das ~11 GB RAM (und langsam). Wir
    # sammeln stattdessen pro Artikel ein kompaktes np.int32-Array (4 Bytes/
    # Token) und konkatenieren einmal am Ende -> ~1.6 GB fuer 400M Tokens.
    all_chunks = []
    running = 0
    chunk = 1000
    n_threads = os.cpu_count() or 4
    for i in range(0, len(ds), chunk):
        end = min(i + chunk, len(ds))
        texts = [a["text"] for a in ds.select(range(i, end))]
        # encode_ordinary_batch: multi-threaded, keine Special-Token-Pruefung
        batched = encoding.encode_ordinary_batch(texts, num_threads=n_threads)
        for tok_list in batched:
            arr = np.asarray(tok_list, dtype=np.int32)
            all_chunks.append(arr)
            running += arr.size
        print(f"  {end:>6}/{len(ds)}: {running:,} tokens")

    tokens = torch.from_numpy(np.concatenate(all_chunks))
    mb = tokens.element_size() * tokens.numel() / 1024**2
    print(f"  Cache schreiben: {cache_path} ({mb:.1f} MB)")
    torch.save(tokens, cache_path)
    return tokens


class TextDataset(Dataset):
    """Naechstes-Token-Vorhersage:
        Input  : Tokens [t0, t1, ..., t_{n-1}]
        Target : Tokens [t1, t2, ..., t_n]
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
        # .long() statt torch.tensor(slice): torch.tensor() auf einem Tensor
        # kopiert unnoetig (und PyTorch warnt davor). .long() macht genau
        # eine Kopie, die beiden Rueckgaben sind Views darauf.
        seq = self.tokens[start:end].long()
        return seq[:-1], seq[1:]


# ============================================================
# Modell-Bausteine
# ============================================================

class TokenEmbedding(nn.Module):
    """Token-Lookup, ohne weitere Skalierung.

    Frueher: `embedding(x) * sqrt(d_model)`. Das stammt aus dem Original-
    Transformer (Vaswani 2017), wo die Multiplikation die Embedding-Magnitude
    an die additive sinusoidale Positional Encoding (Werte in [-1, 1])
    anpassen sollte.

    Mit unserer modernen Architektur ist die Skalierung sogar SCHAEDLICH:
      - RoPE ersetzt die additive PE -> die Magnitude-Anpassung ist unnoetig.
      - Weight Tying: dieselbe Matrix bildet Tokens ein UND macht die End-
        Projektion. Skalierte Inputs blaehen die Magnitude des Residual-
        Streams auf, was nach norm_f zu sehr grossen initialen Logits fuehrt
        und damit zu einem absurd hohen initialen Loss (484 statt log(50257)).

    Deshalb: keine Skalierung mehr. Modernes LLM-Recipe (GPT-2, Llama).
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) — Llama / GPT-NeoX Style.

    Statt eine Positional Encoding *aufzuaddieren*, wird hier eine
    *Rotation* auf Query- und Key-Vektoren angewandt — abhaengig von
    ihrer Position in der Sequenz.

    Schluesseleigenschaft: das Skalarprodukt q_m * k_n nach RoPE haengt
    nur von der *relativen* Position (m - n) ab. Das Modell lernt also
    automatisch positionsrelative Attention, ohne dass wir absolute
    Positionen mitfuehren muessen. Praktische Folgen:
      - Bessere Extrapolation auf laengere Sequenzen als trainiert.
      - Keine Embedding-Parameter fuer Positionen noetig.
      - Standard in Llama, Mistral, Qwen, GPT-NeoX, PaLM-2, ...

    Mathematisch wirkt RoPE auf jedes Dimensions-Paar (x_i, x_{i+d/2})
    als 2D-Rotation um Winkel m * theta_i. Die theta_i folgen einer
    geometrischen Progression:
        theta_i = base^(-2i / d_head)   fuer i = 0..d_head/2-1
    -> kleine i: hohe Frequenz (lokale Position),
       grosse i: niedrige Frequenz (globale Position).
    """

    def __init__(self, d_head, max_seq_len=2048, base=10000.0):
        super().__init__()
        # Inverse Frequenzen: ein Wert pro Dimensions-Paar. Shape: [d_head/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))

        # Winkel(Position m, Paar i) = m * theta_i.  Shape: [T, d_head/2]
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)

        # persistent=False -> nicht in state_dict (jederzeit neu berechenbar)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, seq_len):
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rotary(x, cos, sin):
    """RoPE-Rotation auf x anwenden (GPT-NeoX / Llama Konvention).

    x:        [B, H, T, d_head]
    cos, sin: [T, d_head/2]

    Konvention: die ersten d_head/2 Dimensionen werden gegen die zweiten
    d_head/2 Dimensionen rotiert. Mathematisch aequivalent zur Original-
    RoPE-Definition mit benachbarten Paaren (0,1),(2,3),..., aber sauberer
    im Code, weil chunk()/cat() den Speicher kontinuierlich halten.

        [new_x1]   [cos  -sin] [x1]
        [new_x2] = [sin   cos] [x2]
    """
    x1, x2 = x.chunk(2, dim=-1)              # je [B, H, T, d_head/2]
    cos = cos.unsqueeze(0).unsqueeze(0)      # [1, 1, T, d_head/2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    REFERENZ-IMPLEMENTIERUNG — der Trainingspfad nutzt inzwischen
    F.scaled_dot_product_attention (fused Kernel, Flash-Attention-Stil).
    Diese explizite Version zeigt, was dort intern passiert, und kann
    genutzt werden, wenn man die Attention-Gewichte visualisieren will
    (der fused Kernel gibt sie nie heraus).

    Sie materialisiert die volle [B, H, T, T]-Score-Matrix — bei T=128
    okay, bei langen Sequenzen der Speicher-Flaschenhals.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # -inf statt -1e9: -1e9 wuerde in fp16/bf16 overflowen.
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=2048):
        super().__init__()
        assert d_model % num_heads == 0, "d_model muss durch num_heads teilbar sein"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # RoPE wirkt pro Head, also auf d_k-dimensionale Vektoren.
        self.rope = RotaryEmbedding(self.d_k, max_seq_len=max_seq_len)

        # Dropout auf den Attention-GEWICHTEN (GPT-2: "attn_pdrop") —
        # wird direkt an den fused SDPA-Kernel uebergeben. Das Dropout
        # auf dem Block-OUTPUT ("resid_pdrop") liegt im DecoderBlock;
        # frueher lag es faelschlich an beiden Stellen (doppelt ~0.19).
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

        # RoPE auf Q und K anwenden (NICHT auf V — V traegt Content,
        # nicht Position; nur die Attention-Scores brauchen Position).
        cos, sin = self.rope(Q.size(2))
        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        # Fused SDPA-Kernel (Flash-Attention-Stil): materialisiert die
        # [B, H, T, T]-Score-Matrix nie -> weniger Speicher, schneller.
        # is_causal=True ersetzt unsere manuelle Dreiecks-Maske.
        # dropout_p muss manuell auf 0 in eval() — der Funktions-API
        # kennt den train/eval-Zustand des Moduls nicht.
        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = self.combine_heads(out)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Klassisches Two-Layer FFN — Aktivierung jetzt GELU statt ReLU.

    GELU ist die in GPT-2 / BERT verwendete Variante:
        GELU(x) = x * Phi(x)         (Phi = Standardnormal-CDF)
    Vorteile gegenueber ReLU:
      - Glatt und ueberall differenzierbar (auch fuer x<0 kein "harter Knick").
      - Liefert fuer leicht negative Inputs einen kleinen, nicht-Null
        Output -> weniger "tote" Neuronen, bessere Gradienten-Fluesse.
      - Empirisch konstant minimal besser bei Transformern.

    Diese Klasse bleibt als Referenz-Baseline drin — der DecoderBlock
    nutzt jetzt aber SwiGLU (siehe unten).
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
    """SwiGLU FFN — Llama / Mistral / PaLM Style.

    Klassisches FFN:  out = W2( act(W1 x) )                   2 Matrizen
    SwiGLU:           out = W_down( SiLU(W_gate x) (*) W_up x )  3 Matrizen

    Der entscheidende Unterschied: ein zweiter parallel-Pfad (W_up) und
    eine elementweise Multiplikation (*). Der SiLU-Pfad fungiert als
    "Gate": er entscheidet pro Position und pro Feature, wieviel vom
    Value-Pfad (W_up) durchgelassen wird.

      SiLU(x) = x * sigmoid(x)   (auch "Swish" genannt)

    Warum das hilft: das Modell kann fuer jedes Token dynamisch
    waehlen, welche Features im FFN aktiv sind. Empirisch deutlich
    besser als GELU bei gleicher Parameter-Anzahl, und Standard in
    allen modernen LLMs (Llama 2/3, Mistral, Mixtral, PaLM, ...).

    Parameter-Kosten: 3 statt 2 Matrizen. Llama reduziert daher
    d_ff auf ca. (2/3) * 4 * d_model, um die Parameter-Anzahl zu
    halten. Wir lassen d_ff erstmal bei 2048 (-> ca. +50 % FFN-Params
    gegenueber GELU-FFN); bei Bedarf koennen wir D_FF auf 1408 senken.

    Kein eigenes Dropout mehr: das Residual-Dropout liegt zentral im
    DecoderBlock — frueher wurde hier UND dort gedroppt (doppelt).
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        # bias=False ist Llama-Konvention; bei Weight-Tying & RMSNorm
        # sparen die Biases ohnehin keine Modell-Kapazitaet.
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(gated)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019).

    Vereinfachung der LayerNorm:
      LayerNorm: y = gamma * (x - mean(x)) / std(x) + beta    (4 stats, 2 params)
      RMSNorm:   y = gamma * x / RMS(x)                       (1 stat,  1 param)

    Wo RMS(x) = sqrt(mean(x^2)) ueber die letzte Dimension berechnet wird.

    Was wegfaellt:
      - Mean-Subtraktion ("re-centering"). Empirisch in Transformern entbehrlich.
      - Bias (beta). Kann von nachfolgenden Linear-Layers absorbiert werden.

    Was bleibt:
      - Magnitude-Normalisierung. Sichert stabilen Forward/Backward.
      - Gelernte Skalierung (gamma).

    Vorteile: ~10-20 % weniger Operations, gleich gute (oft minimal bessere)
    Trainings-Dynamik. Standard in Llama, Mistral, Qwen, Gemma, Falcon.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # mean of squares ueber die letzte Dimension; rsqrt = 1/sqrt
        # (eine Hardware-Instruktion, schneller als getrennte Division)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = x * torch.rsqrt(ms + self.eps)
        return self.weight * x_normalized


class DecoderBlock(nn.Module):
    """Transformer Decoder Block mit Pre-LayerNorm.

    Post-LN (Original):  x = LN( x + SubLayer(x) )
    Pre-LN  (GPT-2+):    x = x + SubLayer( LN(x) )

    Pre-LN haelt den Residual-Pfad un-normalisiert, was tiefe Transformer
    ohne LR-Warmup stabil trainierbar macht.

    Dropout-Platzierung (GPT-2-Schema): genau EIN Residual-Dropout pro
    Sublayer, direkt vor der Addition auf den Residual-Stream. Die
    Sublayer selbst droppen ihren Output nicht mehr — vorher wurde
    doppelt gedroppt (effektiv ~0.19 statt 0.1).
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


def create_causal_mask(seq_len, device):
    """Lower-triangular Maske: Position i sieht nur Positionen <= i.

    REFERENZ — nur noch fuer die explizite scaled_dot_product_attention
    oben relevant. Der Trainingspfad nutzt is_causal=True im fused Kernel,
    die Maske muss also nie als Tensor existieren.
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


class GPTDecoder(nn.Module):
    """GPT-Style Decoder-only Transformer (Pre-LN + Weight Tying)."""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Pre-LN braucht eine finale Norm vor dem LM-Head, sonst kann
        # der Residual-Stream ueber die Layer hinweg unbeschraenkt wachsen.
        self.norm_f = RMSNorm(d_model)

        # bias=False, weil wir die Gewichte mit der Embedding-Matrix teilen.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying: dieselbe [vocab_size, d_model] Matrix dient sowohl
        # zum Einbetten (Token-ID -> Vektor) als auch zum Vorhersagen
        # (Vektor -> Logits ueber Vocab). Spart ~25M Parameter und
        # verbessert die Sample-Qualitaet meist sichtbar.
        self.lm_head.weight = self.embedding.embedding.weight

        self.dropout = nn.Dropout(dropout)

        # GPT-2 Init-Recipe — siehe _init_weights.
        # Reihenfolge wichtig: erst Tying, dann Init, damit beide Verweise
        # auf dieselbe Parameter-Instanz dasselbe initiale Sample sehen.
        self.apply(self._init_weights)

        # Skalierte Init fuer "Output"-Projektionen jeder Residual-Branch:
        # nach Pre-LN wachsen die Aktivierungen sonst pro Block linear an,
        # weil jedes Residual aufaddiert. Wir teilen die initiale Magnitude
        # dieser Projektionen durch sqrt(2 * num_layers) — Standard-Trick
        # aus dem GPT-2 Paper. Betrifft W_o (Attention-Output) und w_down
        # (SwiGLU-Output).
        proj_std = 0.02 / math.sqrt(2 * num_layers)
        for block in self.decoder_blocks:
            nn.init.normal_(block.attention.W_o.weight, mean=0.0, std=proj_std)
            nn.init.normal_(block.feed_forward.w_down.weight, mean=0.0, std=proj_std)

    @staticmethod
    def _init_weights(module):
        """GPT-2 Init-Schema: alle gewichteten Layers ~ N(0, 0.02^2), Biases = 0.

        Warum 0.02? Empirischer Wert aus dem GPT-2 Paper, gut kalibriert
        fuer LayerNorm-basierte Transformer. Sorgt dafuer, dass:
          - Initial-Aktivierungen bleiben in einem vernuenftigen Bereich.
          - Initial-Logits klein sind (~N(0, sqrt(d_model)*0.02) = ~0.45),
            d.h. Softmax fast uniform -> Initial-Loss ≈ log(vocab_size).
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
                 top_k=None, top_p=None, repetition_penalty=1.0):
        """Autoregressives Sampling mit modernen Decoding-Strategien.

        temperature: skaliert Logits vor Softmax. <1 schaerfer, >1 flacher.
        top_k:       behalte nur die k wahrscheinlichsten Tokens.
        top_p:       Nucleus-Sampling — behalte Tokens deren kumulative
                     Wahrscheinlichkeit <= p ist.
        repetition_penalty:
                     >1.0 reduziert die Wahrscheinlichkeit von Tokens, die
                     bereits im Kontext sind (gegen "Stadt im Stadt im Stadt").

        Reihenfolge: rep_penalty -> /temperature -> top_k -> top_p -> sample.
        """
        # Trainings-Modus merken und am Ende wiederherstellen — sonst hat
        # ein generate()-Aufruf den Seiteneffekt, Dropout dauerhaft
        # abzuschalten, falls der Aufrufer weitertrainiert.
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            logits = logits[:, -1, :]  # [B, V] — Logits fuer das naechste Token

            # 1. Repetition Penalty: Logits bereits gesehener Tokens daempfen.
            # HuggingFace-Konvention: positive Logits / penalty, negative * penalty
            # (beides treibt das Logit Richtung minus-unendlich).
            if repetition_penalty != 1.0:
                for b in range(input_ids.size(0)):
                    seen = torch.unique(input_ids[b])
                    score = logits[b, seen]
                    logits[b, seen] = torch.where(
                        score < 0, score * repetition_penalty, score / repetition_penalty
                    )

            # 2. Temperature
            logits = logits / temperature

            # 3. Top-k: setze alle ausser den top-k Logits auf -inf.
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                threshold = topk_vals[..., -1, None]  # kleinster Top-k Wert
                logits = torch.where(logits < threshold,
                                     torch.full_like(logits, float("-inf")),
                                     logits)

            # 4. Top-p (Nucleus): sortiere absteigend, behalte minimale
            # Tokenmenge deren kumulative Prob >= top_p ist.
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                # Tokens markieren, deren kumulative Prob top_p ueberschreitet.
                remove_sorted = cumulative > top_p
                # Shift nach rechts — den ersten Ueberschreiter noch behalten.
                remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
                remove_sorted[..., 0] = False
                # Maske zurueck auf Original-Indizes mappen.
                remove = torch.zeros_like(logits, dtype=torch.bool)
                remove.scatter_(-1, sorted_idx, remove_sorted)
                logits = logits.masked_fill(remove, float("-inf"))

            # 5. Softmax + Multinomial Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        if was_training:
            self.train()
        return input_ids


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(model, optimizer, epoch, global_step, loss, filepath, config):
    # torch.compile wrappt das Modell in ein OptimizedModule — dessen
    # state_dict-Keys bekommen ein '_orig_mod.'-Praefix. Wir speichern immer
    # das unkompilierte Original, damit das Checkpoint-Format nicht davon
    # abhaengt, ob mit oder ohne compile trainiert wurde.
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
    print(f"  Checkpoint gespeichert: {filepath}")


def load_checkpoint(filepath, model, optimizer, device):
    ckpt = torch.load(filepath, map_location=device)

    # Inkompatibilitaet hart erkennen, statt mit zufaellig initialisierter
    # norm_f / falsch getieden lm_head-Gewichten weiterzumachen.
    ckpt_arch = ckpt.get("arch_version", "legacy_post_ln")
    if ckpt_arch != ARCH_VERSION:
        raise RuntimeError(
            f"Checkpoint '{filepath}' hat arch_version='{ckpt_arch}', "
            f"aktuelle Architektur ist '{ARCH_VERSION}'. "
            "Loeschen, umbenennen, oder von Hand migrieren."
        )

    # Aeltere Checkpoints wurden aus dem kompilierten Modell gespeichert
    # ('_orig_mod.'-Praefix in den Keys) — beim Laden normalisieren.
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
            # Z.B. Checkpoint von plain Adam (1 Param-Group), aktueller
            # Optimizer ist AdamW mit decay/no-decay-Groups. Die Modell-
            # Gewichte sind geladen; nur die Adam-Momente starten frisch.
            print(f"  Optimizer-State inkompatibel — starte mit frischem "
                  f"Optimizer ({type(e).__name__}: {e})")
    epoch = ckpt["epoch"]
    global_step = ckpt.get("global_step", 0)
    print(f"  Checkpoint geladen: Epoche {epoch}, Step {global_step}, Loss {ckpt['loss']:.4f}")
    return epoch, global_step, ckpt["loss"]


def find_latest_checkpoint(directory):
    """Sucht den juengsten Checkpoint (epoch- oder rolling-latest).

    Beruecksichtigt:
      - checkpoint_epoch_N.pt (am Epoche-Ende geschrieben)
      - checkpoint_latest.pt  (rolling mid-epoch, immer ueberschrieben)

    Auswahl per mtime — der zuletzt geschriebene Checkpoint gewinnt,
    egal welcher Typ. None wenn nichts gefunden.
    """
    paths = glob.glob(os.path.join(directory, "checkpoint_epoch_*.pt"))
    latest_path = os.path.join(directory, "checkpoint_latest.pt")
    if os.path.exists(latest_path):
        paths.append(latest_path)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def checkpoint_dims_match(filepath):
    """True, wenn der Checkpoint dieselben Modell-Dimensionen hat wie die
    aktuelle Konfiguration.

    Warum noetig: der ARCH_VERSION-Check faengt nur ARCHITEKTUR-Aenderungen ab
    (z.B. LayerNorm -> RMSNorm), nicht GROESSEN-Aenderungen bei gleicher
    Architektur. Wenn man von 51M auf 124M skaliert und noch ein alter
    Checkpoint herumliegt, wuerde load_state_dict sonst mit einem kryptischen
    'size mismatch' sterben. Hier pruefen wir vorab und starten sauber frisch.
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
    """AdamW mit selektivem Weight Decay — GPT-2/Llama-Recipe.

    Warum AdamW statt Adam: bei Adam interagiert L2-Regularisierung mit
    der per-Parameter-Skalierung der Gradienten und wirkt dadurch nicht
    als echter Decay (Loshchilov & Hutter 2019). AdamW entkoppelt den
    Decay vom Gradienten-Update — nur so wirkt weight_decay wie gedacht.

    Decay nur auf Matrizen (dim >= 2): Linear- und Embedding-Gewichte.
    RMSNorm-Gains und Biases (dim 1) NICHT decayen — sie skalieren
    Aktivierungen, ein Zug Richtung 0 wuerde dort nur das Training
    stoeren, ohne zu regularisieren. (Konvention aus GPT-2/nanoGPT.)

    betas=(0.9, 0.95): das niedrigere beta2 (statt 0.999) reagiert
    schneller auf Gradienten-Varianz-Aenderungen — Standard bei
    LLM-Training (GPT-3, Llama).

    named_parameters() dedupliziert geteilte Parameter, das getiede
    embedding/lm_head-Gewicht landet also genau einmal in der Liste.
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
    """LR-Schedule: linearer Warmup gefolgt von Cosine Decay.

    Warmup (step < warmup_steps): von ~0 linear auf base_lr.
    Cosine (danach): von base_lr cosinus-formig auf min_lr ueber den Rest.

    Warum Warmup? Adam braucht ein paar Schritte um vernuenftige
    Moment-Schaetzer aufzubauen — vorher kann ein voller LR zu
    Loss-Explosionen fuehren, gerade bei zufaelliger Initialisierung.

    Warum Cosine Decay? Empirisch besser als step decay / linear decay,
    weil das Modell anfangs lange mit hohem LR exploriert und am Ende
    sanft konvergiert. Standard in Llama, GPT-3, fast allen modernen LLMs.
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
    """Trainiert eine Epoche mit Gradient Accumulation.

    Gradient Accumulation: wir sammeln die Gradienten aus `grad_accum_steps`
    Mikro-Batches auf, bevor wir EINEN Optimizer-Step machen. Effektiv
    trainieren wir damit mit (batch_size * grad_accum_steps) Sequenzen pro
    Update, ohne den Speicher fuer so einen grossen Batch auf einmal zu
    brauchen. Warum das zaehlt: groessere Batches liefern weniger verrauschte
    Gradienten (gut fuer groessere Modelle), aber der GPU-Speicher deckelt die
    ECHTE Batch-Groesse. Accumulation entkoppelt beides.

    Der `loss / grad_accum_steps`-Trick: PyTorch ADDIERT Gradienten ueber
    mehrere backward()-Aufrufe. Damit die Summe dem MITTEL ueber den grossen
    Batch entspricht (nicht seiner Summe), skalieren wir jeden Teil-Loss vor
    dem backward() herunter.

    global_step zaehlt OPTIMIZER-Steps (nicht Mikro-Batches) — nur so passt
    der LR-Schedule, der in Optimizer-Steps gedacht ist.

    save_step_interval: alle N Optimizer-Steps save_callback(global_step, loss).
    amp_dtype:          z.B. torch.bfloat16 -> Forward+Loss unter Autocast.
    """
    model.train()
    total_loss = torch.zeros((), device=device)  # Akkumulation auf dem Device
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

        # Runterskalieren -> aufsummierte Grads == Mittel ueber den grossen Batch
        (loss / grad_accum_steps).backward()

        # .detach() statt .item(): .item() erzwingt einen CPU<->GPU-Sync. Wir
        # akkumulieren als Tensor auf dem Device und syncen nur beim Logging.
        total_loss += loss.detach()
        n_microbatches += 1

        # Optimizer-Step erst, wenn ein volles Accumulation-Fenster voll ist.
        if (batch_idx + 1) % grad_accum_steps == 0:
            # LR fuer diesen Optimizer-Step setzen
            lr = get_lr(global_step, warmup_steps, max_steps, base_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            last_lr = lr

            # Gradient Clipping: kappt die globale L2-Norm aller Gradienten auf
            # grad_clip. Verhindert, dass ein Ausreisser-Batch das Modell in
            # eine schlechte Region kickt ("loss spike").
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 20 == 0:
                print(f"  Step {global_step}/{max_steps}, "
                      f"loss={loss.item():.4f}, lr={lr:.2e}")

            # Rolling Mid-Epoch Checkpoint
            if save_step_interval and save_callback and global_step % save_step_interval == 0:
                save_callback(global_step, loss.item())

    return (total_loss / max(1, n_microbatches)).item(), global_step, last_lr


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, vocab_size, amp_dtype=None):
    """Mittlerer Loss auf dem Validation-Set.

    eval()-Modus deaktiviert Dropout, und @torch.no_grad() spart Speicher
    und Zeit, weil keine Gradienten gebaut werden.
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
    """Generiert Samples mit konservativem, repetitions-armem Sampling."""
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
    print(f"Autocast: {amp_dtype if amp_dtype else 'aus (fp32)'}\n")

    # --- Daten (cached) ---
    encoding = tiktoken.get_encoding("gpt2")
    vocab_size = encoding.n_vocab
    tokens = get_or_build_tokens(NUM_ARTICLES, encoding)
    print(f"  Vocab: {vocab_size}")

    # Train / Val Split — letzte VAL_FRACTION der Tokens als Validation.
    # (Sequentieller Split, NICHT Shuffle: Wikipedia-Artikel sind im
    # Stream sequenziell konkateniert, ein zufaelliger Token-Shuffle wuerde
    # Tokens aus demselben Artikel in beide Splits leaken.)
    split_idx = int(len(tokens) * (1 - VAL_FRACTION))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    print(f"  Split: {len(train_tokens):,} train / {len(val_tokens):,} val\n")

    train_ds = TextDataset(train_tokens, seq_length=SEQ_LENGTH)
    val_ds = TextDataset(val_tokens, seq_length=SEQ_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"DataLoader: {len(train_loader)} train batches, {len(val_loader)} val batches\n")

    # --- Modell ---
    model = GPTDecoder(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modell: {n_params:,} Parameter")

    # torch.compile: fusioniert Ops, eliminiert Sync-Punkte, kann auf MPS
    # 30-100 % Speedup bringen. Erster Forward-Pass ist langsam (Kompilation),
    # danach schneller. Fail-safe: bei Crash auf eager fallback.
    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            print("torch.compile: aktiv (erster Batch laeuft langsamer wegen JIT)")
        except Exception as e:
            print(f"torch.compile fehlgeschlagen, eager mode: {e}")
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
            print(f"Auto-Resume: '{resume_path}'")

    if resume_path and os.path.exists(resume_path) and not checkpoint_dims_match(resume_path):
        print(f"  Checkpoint '{resume_path}' hat andere Modell-Dimensionen als "
              f"die aktuelle Konfiguration ({D_MODEL}d / {NUM_LAYERS}L) — "
              f"ignoriere ihn, starte frisch.")
        resume_path = None

    if resume_path and os.path.exists(resume_path):
        start_epoch, global_step, _ = load_checkpoint(resume_path, model, optimizer, device)
    elif resume_path:
        print(f"  (Kein Checkpoint '{resume_path}' gefunden — starte frisch.)")
    else:
        print("Starte frisches Training (kein Checkpoint).")

    # --- Training-Loop ---
    # Auch die Trainings-Hyperparameter mitschreiben, damit man spaeter
    # einem Checkpoint ansieht, mit welchen Settings er entstand.
    config = dict(
        vocab_size=vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
        d_ff=D_FF, num_layers=NUM_LAYERS, seq_length=SEQ_LENGTH,
        dropout=DROPOUT, batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE, min_lr=MIN_LR, weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS, grad_clip=GRAD_CLIP, num_articles=NUM_ARTICLES,
        amp_dtype=str(amp_dtype) if amp_dtype else None,
    )
    test_prompts = ["Die Geschichte", "Im Jahr", "Deutschland ist"]
    # max_steps zaehlt OPTIMIZER-Steps: pro Epoche floor(#Batches / grad_accum).
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    max_steps = NUM_EPOCHS * steps_per_epoch
    eff_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    print(f"Effektive Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum = {eff_batch} "
          f"Sequenzen ({eff_batch * SEQ_LENGTH:,} Tokens/Step)")
    print(f"Training: max_steps={max_steps}, steps/epoch={steps_per_epoch}, "
          f"warmup_steps={WARMUP_STEPS}, peak_lr={LEARNING_RATE:.2e}, "
          f"min_lr={MIN_LR:.2e}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoche {epoch + 1}/{NUM_EPOCHS}")

        # Rolling Mid-Epoch Checkpoint Callback. Schreibt 'checkpoint_latest.pt'
        # mit der AKTUELLEN Epoche — ein Resume beginnt diese Epoche von vorn.
        # Ehrliche Einschraenkungen dieses simplen Schemas:
        #   - Re-done wird alles, was in der Epoche schon gelaufen war
        #     (im schlimmsten Fall fast eine ganze Epoche, nicht nur
        #     save_step_interval Batches) — aber die Gewichte/Steps bleiben.
        #   - global_step laeuft weiter -> der Cosine-Schedule verschiebt
        #     sich relativ zu den Daten.
        #   - DataLoader-Shuffle hat keinen festen Seed -> der Resume-Lauf
        #     sieht eine andere Batch-Reihenfolge.
        # Fuer exaktes Resume muesste man Sampler-State + RNG-State sichern.
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

        # Perplexity = exp(cross-entropy). Lesbarere Metrik als Loss:
        # ppl=N bedeutet "Modell ist im Schnitt zwischen N Tokens unsicher".
        train_ppl = math.exp(min(train_loss, 20))   # clip gegen overflow
        val_ppl = math.exp(min(val_loss, 20))
        print(f"-> train: loss={train_loss:.4f}  ppl={train_ppl:.1f}")
        print(f"-> val:   loss={val_loss:.4f}  ppl={val_ppl:.1f}   "
              f"(letzter lr={last_lr:.2e})")

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
