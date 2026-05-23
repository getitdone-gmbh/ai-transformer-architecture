"""GPT-Style Transformer fuer deutsche Wikipedia.

Decoder-only Architektur, trainiert mit Causal Language Modeling.
Laeuft auf MPS (Apple Silicon), CUDA und CPU.
"""

import glob
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tiktoken
from datasets import load_dataset


# ============================================================
# Hyperparameter
# ============================================================

# Modell
D_MODEL = 512
NUM_HEADS = 8
D_FF = 2048
NUM_LAYERS = 6
DROPOUT = 0.1

# Training
SEQ_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 3e-4   # peak LR (Ende des Warmups, Start des Cosine-Decays)
MIN_LR = 3e-5          # LR-Untergrenze nach komplettem Cosine-Decay
WARMUP_STEPS = 100     # linearer Warmup ueber die ersten N Optimizer-Steps
GRAD_CLIP = 1.0        # globale L2-Norm-Schranke fuer Gradienten
NUM_EPOCHS = 10

# Daten
NUM_ARTICLES = 1000
VAL_FRACTION = 0.05    # Anteil Tokens fuer Validation-Split (Ende der Sequenz)

# Checkpoints
CHECKPOINT_DIR = "."
CHECKPOINT_EVERY_N_EPOCHS = 2
AUTO_RESUME = True          # neuesten checkpoint_epoch_*.pt automatisch laden
RESUME_FROM = None          # ueberschreibt AUTO_RESUME wenn gesetzt
# Architektur-Marker im Checkpoint, damit alte Files erkannt werden
ARCH_VERSION = "pre_ln_tied_swiglu_rope_init_v1"


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Daten
# ============================================================

def load_wikipedia_text(num_articles):
    """Laedt deutsche Wikipedia-Artikel und konkateniert ihren Text."""
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.de", split="train", streaming=False
    )
    ds = ds.select(range(min(num_articles, len(ds))))
    return " ".join(article["text"] for article in ds)


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
        seq = self.tokens[start:end]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids


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
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
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

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # [B, T, D] -> [B, H, T, d_k]
        B, T, _ = x.size()
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # [B, H, T, d_k] -> [B, T, D]
        B, _, T, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # RoPE auf Q und K anwenden (NICHT auf V — V traegt Content,
        # nicht Position; nur die Attention-Scores brauchen Position).
        cos, sin = self.rope(Q.size(2))
        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        out = self.combine_heads(out)
        out = self.W_o(out)
        out = self.dropout(out)
        return out, attn


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
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # bias=False ist Llama-Konvention; bei Weight-Tying & RMSNorm
        # sparen die Biases ohnehin keine Modell-Kapazitaet.
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        out = self.w_down(gated)
        out = self.dropout(out)
        return out


class DecoderBlock(nn.Module):
    """Transformer Decoder Block mit Pre-LayerNorm.

    Post-LN (Original):  x = LN( x + SubLayer(x) )
    Pre-LN  (GPT-2+):    x = x + SubLayer( LN(x) )

    Pre-LN haelt den Residual-Pfad un-normalisiert, was tiefe Transformer
    ohne LR-Warmup stabil trainierbar macht.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = SwiGLU(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


def create_causal_mask(seq_len, device):
    """Lower-triangular Maske: Position i sieht nur Positionen <= i."""
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

        # Pre-LN braucht eine finale LayerNorm vor dem LM-Head, sonst kann
        # der Residual-Stream ueber die Layer hinweg unbeschraenkt wachsen.
        self.norm_f = nn.LayerNorm(d_model)

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

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.dropout(x)
        for block in self.decoder_blocks:
            x = block(x, mask)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)
            mask = create_causal_mask(seq_len, input_ids.device)
            logits = self.forward(input_ids, mask)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(model, optimizer, epoch, global_step, loss, filepath, config):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
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

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    global_step = ckpt.get("global_step", 0)
    print(f"  Checkpoint geladen: Epoche {epoch}, Step {global_step}, Loss {ckpt['loss']:.4f}")
    return epoch, global_step, ckpt["loss"]


def find_latest_checkpoint(directory):
    """Sucht checkpoint_epoch_N.pt mit groesstem N. None wenn keiner da."""
    candidates = []
    for path in glob.glob(os.path.join(directory, "checkpoint_epoch_*.pt")):
        m = re.search(r"checkpoint_epoch_(\d+)\.pt$", path)
        if m:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


# ============================================================
# Training
# ============================================================

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
                global_step, max_steps, warmup_steps, base_lr, min_lr, grad_clip):
    model.train()
    total_loss = 0.0
    last_lr = 0.0

    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # LR fuer diesen Step setzen
        lr = get_lr(global_step, warmup_steps, max_steps, base_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        last_lr = lr

        mask = create_causal_mask(input_ids.size(1), device)
        logits = model(input_ids, mask)
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping: kappt die globale L2-Norm aller Gradienten
        # auf grad_clip. Verhindert, dass ein einzelner Ausreisser-Batch
        # das Modell in eine schlechte Region kickt ("loss spike").
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                  f"loss={loss.item():.4f}, lr={lr:.2e}")

    return total_loss / len(dataloader), global_step, last_lr


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, vocab_size):
    """Mittlerer Loss auf dem Validation-Set.

    eval()-Modus deaktiviert Dropout, und @torch.no_grad() spart Speicher
    und Zeit, weil keine Gradienten gebaut werden.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        mask = create_causal_mask(input_ids.size(1), device)
        logits = model(input_ids, mask)
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def generate_samples(model, encoding, device, prompts, max_new_tokens=20, temperature=0.8):
    for prompt in prompts:
        start = encoding.encode(prompt)
        input_ids = torch.tensor([start], device=device)
        out_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        text = encoding.decode(out_ids[0].cpu().tolist())
        print(f"  '{text}'")


# ============================================================
# Main
# ============================================================

def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}\n")

    # --- Daten ---
    print(f"Lade {NUM_ARTICLES} Wikipedia-Artikel...")
    text = load_wikipedia_text(NUM_ARTICLES)
    print(f"  Zeichen: {len(text):,}")

    encoding = tiktoken.get_encoding("gpt2")
    tokens = encoding.encode(text)
    vocab_size = encoding.n_vocab
    print(f"  Tokens: {len(tokens):,} (Vocab: {vocab_size})")

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
    print(f"Modell: {n_params:,} Parameter\n")

    # --- Optimizer + Loss ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Optional: Resume ---
    start_epoch = 0
    global_step = 0
    resume_path = RESUME_FROM
    if resume_path is None and AUTO_RESUME:
        resume_path = find_latest_checkpoint(CHECKPOINT_DIR)
        if resume_path:
            print(f"Auto-Resume: '{resume_path}'")

    if resume_path and os.path.exists(resume_path):
        start_epoch, global_step, _ = load_checkpoint(resume_path, model, optimizer, device)
    elif resume_path:
        print(f"  (Kein Checkpoint '{resume_path}' gefunden — starte frisch.)")
    else:
        print("Starte frisches Training (kein Checkpoint).")

    # --- Training-Loop ---
    config = dict(
        vocab_size=vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
        d_ff=D_FF, num_layers=NUM_LAYERS, seq_length=SEQ_LENGTH,
    )
    test_prompts = ["Die Geschichte", "Im Jahr", "Deutschland ist"]
    max_steps = NUM_EPOCHS * len(train_loader)
    print(f"Training: max_steps={max_steps}, warmup_steps={WARMUP_STEPS}, "
          f"peak_lr={LEARNING_RATE:.2e}, min_lr={MIN_LR:.2e}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoche {epoch + 1}/{NUM_EPOCHS}")
        train_loss, global_step, last_lr = train_epoch(
            model, train_loader, criterion, optimizer, device, vocab_size,
            global_step, max_steps, WARMUP_STEPS, LEARNING_RATE, MIN_LR, GRAD_CLIP,
        )
        val_loss = evaluate(model, val_loader, criterion, device, vocab_size)

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
