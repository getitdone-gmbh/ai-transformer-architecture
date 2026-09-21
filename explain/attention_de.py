"""Looking inside Multi-Head Attention — on our own 540M, on German text.

The embedding chapter answered "what does the model see". This one answers
"how does it connect what it sees". 24 layers x 20 heads = 480 attention
heads, and we can look at every single one of them.

WHAT THIS SHOWS, measured on weights_540m_hq_fp32.pt:

  1. The matrix itself. Every token asks every earlier token "how relevant
     are you to me". The result is a T x T table. The upper triangle is
     empty — not because the model is polite, but because the causal mask
     sets it to -inf before the softmax.

  2. Heads are NOT interchangeable. L11 H13 is a clean previous-token
     head (sub-diagonal ~0.9). Its neighbours in the same layer do
     completely different things. That is what "multi-head" buys.

  3. Most heads do nothing most of the time. Averaged over all 480 heads,
     52% of all attention mass lands on token 0 — the "attention sink",
     a parking position for heads that have nothing to contribute here.

  4. Some heads have nameable jobs. SCAN mode ranks all 480 heads by how
     much attention flows from word A to word B, so you can search for
     them yourself instead of believing this docstring.

  5. And a control that killed our first hypothesis. L15 H18 looked like
     a "separable verb" head ("Er machte die Tuer AUF" -> auf points back
     at machte, 0.93). Then we tested "auf" as a plain preposition: "Er
     stieg auf den Berg" -> 0.97. Same strength. It is not a prefix head.
     Run CONTROL=1 to see it. The lesson is the point: an attention map
     invites you to see meaning that is not there.

WHY THIS IS NOT A HEATMAP TOY: the numbers come out of the real
forward pass. The training path uses the fused SDPA kernel, which never
materializes the score matrix, so we recompute Q and K from the captured
layer input and apply RoPE and the causal mask exactly like train.py
does. Same weights, same math, just observable.

Usage:
    python explain/attention_de.py                      # interactive: sentence in, map out
    python explain/attention_de.py "Der Arzt sagte, dass er müde war."
    DEMO=1 python explain/attention_de.py               # guided tour (screen recording)
    VIDEO=1 DEMO=1 python explain/attention_de.py       # dito with pauses
    LAYER=11 HEAD=13 python explain/attention_de.py "..."   # one specific head
    HEADS=1 LAYER=11 python explain/attention_de.py "..."   # all 20 heads of a layer
    SCAN=1 Q=er K=Arzt python explain/attention_de.py "..." # which head links these words?
    CONTROL=1 python explain/attention_de.py            # the falsified hypothesis
    CHECKPOINT=sft_540m_v2.pt python explain/attention_de.py ...
"""

import math
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import re

import tiktoken
import torch

# model/ on sys.path — the repo layout lives in model/paths.py, and
# anchoring to __file__ is what lets this script run from any directory.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
import train  # noqa: E402
from paths import ckpt_path  # noqa: E402

CHECKPOINT = ckpt_path(os.environ.get("CHECKPOINT", "weights_540m_hq_fp32.pt"))
VIDEO = os.environ.get("VIDEO", "0") in ("1", "true")
DEMO = os.environ.get("DEMO", "0") in ("1", "true")
SCAN = os.environ.get("SCAN", "0") in ("1", "true")
HEADS = os.environ.get("HEADS", "0") in ("1", "true")
CONTROL = os.environ.get("CONTROL", "0") in ("1", "true")
PLAIN = os.environ.get("NO_COLOR", "0") in ("1", "true")
LAYER = int(os.environ.get("LAYER", "11"))
HEAD = int(os.environ.get("HEAD", "13"))


def _c(code):
    return "" if PLAIN else f"\033[{code}m"


RESET, BOLD, DIM = _c(0), _c(1), _c(2)
CYAN, YELLOW, GREEN, RED, MAG = _c(96), _c(93), _c(92), _c(91), _c(95)

# Heat ramp as 256-colour background indices, dark -> bright. Chosen so
# that the shape survives both a bright and a dark terminal, and so that
# "almost zero" is visibly different from "exactly zero".
RAMP = [235, 17, 18, 19, 20, 26, 31, 37, 43, 49, 84, 120, 190, 220, 214, 208]


def say(text, end="\n"):
    """Typewriter in VIDEO mode. ANSI codes are not 'typed'."""
    if VIDEO:
        i = 0
        while i < len(text):
            if text[i] == "\033":
                j = text.index("m", i) + 1
                sys.stdout.write(text[i:j])
                i = j
                continue
            sys.stdout.write(text[i])
            sys.stdout.flush()
            time.sleep(0.008)
            i += 1
        sys.stdout.write(end)
        sys.stdout.flush()
    else:
        print(text, end=end)


def pause(sec):
    if VIDEO:
        time.sleep(sec)


def head(title):
    say("")
    say(f"{CYAN}{'=' * 74}{RESET}")
    say(f"{CYAN}{BOLD}  {title}{RESET}")
    say(f"{CYAN}{'=' * 74}{RESET}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load():
    dev = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    say(f"{DIM}Lade {CHECKPOINT} auf {dev} ...{RESET}")
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = train.GPTDecoder(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"], dropout=0.0).to(dev)
    model.load_state_dict({k.removeprefix("_orig_mod."): v
                           for k, v in ck["model_state_dict"].items()})
    model.eval()
    # Capture each attention block's INPUT. Pre-LN means this is already
    # normed, i.e. exactly the tensor W_q/W_k see inside forward().
    cap = {}
    for i, blk in enumerate(model.decoder_blocks):
        blk.attention.register_forward_hook(
            lambda mod, inp, out, i=i: cap.__setitem__(i, inp[0].detach()))
    return model, cap, dev, cfg


@torch.no_grad()
def attention(model, cap, dev, ids):
    """[L, H, T, T] attention weights — recomputed, not approximated.

    train.py runs F.scaled_dot_product_attention, a fused kernel that
    never builds the [B,H,T,T] matrix. To see the weights we redo the
    two lines it hides: scores = QK^T/sqrt(d_k), then softmax over the
    causally masked scores. Same Q, same K, same RoPE, same weights.
    """
    T = len(ids)
    model(torch.tensor([ids], device=dev))
    mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
    out = []
    for i, blk in enumerate(model.decoder_blocks):
        a = blk.attention
        x = cap[i]
        Q = a.split_heads(a.W_q(x))
        K = a.split_heads(a.W_k(x))
        cos, sin = a.rope(T)
        Q = train.apply_rotary(Q, cos, sin)
        K = train.apply_rotary(K, cos, sin)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(a.d_k)
        scores = scores.masked_fill(mask, float("-inf"))
        out.append(torch.softmax(scores.float(), -1)[0].cpu())
    return torch.stack(out)


ENC = tiktoken.get_encoding("gpt2")


def pieces(ids):
    """Token bytes -> printable. '·' for spaces, hex for split UTF-8."""
    out = []
    for i in ids:
        raw = ENC.decode_single_token_bytes(i)
        try:
            out.append(raw.decode("utf-8").replace(" ", "·"))
        except UnicodeDecodeError:
            out.append("".join(f"\\x{b:02x}" for b in raw))
    return out


def word_span(text, ids, word, last=False):
    """Token indices covering `word` — needed because GPT-2 shreds German.

    'auf' is not one token, it is ' a' + 'uf'. Asking "where does the
    word auf attend" therefore means asking about a SPAN, not an index.
    """
    hits = list(re.finditer(
        rf"(?<![\wÀ-ɏ]){re.escape(word)}(?![\wÀ-ɏ])", text))
    if not hits:
        return []
    c1, c2 = (hits[-1] if last else hits[0]).span()
    spans, pos = [], 0
    for k, i in enumerate(ids):
        n = len(ENC.decode_single_token_bytes(i).decode("utf-8", "replace"))
        if pos < c2 and pos + n > c1:
            spans.append(k)
        pos += n
    return spans


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def cell(v):
    """One heat cell. sqrt-scaled: attention is spiky, linear hides shape."""
    if v < 0.005:
        return f"{DIM} ·{RESET}"
    idx = min(len(RAMP) - 1, int(math.sqrt(v) * (len(RAMP) - 1)))
    fg = 232 if idx >= len(RAMP) - 4 else 255
    n = 99 if v >= 0.995 else max(1, round(v * 100))   # keep every cell 2 chars wide
    return f"{_c(f'48;5;{RAMP[idx]};38;5;{fg}')}{n:>2}{RESET}"


def matrix(A, L, H, tk, note=""):
    """One head as a labelled heat map. Rows ask, columns answer."""
    M = A[L, H]
    T = len(tk)
    say("")
    say(f"  {BOLD}Layer {L}, Kopf {H}{RESET}{DIM}   {note}{RESET}")
    say(f"  {DIM}Zeile = wer fragt · Spalte = worauf geschaut wird · "
        f"Zahl = % der Aufmerksamkeit{RESET}")
    say("")
    say(f"  {' ' * 12}" + "".join(f"{i:>3}" for i in range(T)))
    for r in range(T):
        cells = "".join(cell(v) for v in M[r].tolist())
        say(f"  {tk[r][:10]:>10} {r:>2}{cells}")
    say("")
    legend = "  ".join(f"{DIM}{i}{RESET}={t}" for i, t in enumerate(tk))
    say(f"  {DIM}Spalten:{RESET} {legend}")


def head_strip(A, L, tk):
    """All 20 heads of one layer, small, side by side — the 'multi' part.

    One character per cell so a whole layer fits in a normal terminal:
    the SHAPE is the message here, not the numbers.
    """
    nH, T = A.size(1), len(tk)
    per_row = max(1, 88 // (T + 3))
    say("")
    say(f"  {BOLD}Alle {nH} Köpfe von Layer {L}{RESET}"
        f"{DIM} — gleicher Satz, gleiche Schicht, andere Muster{RESET}")
    say(f"  {DIM}Ein Zeichen = ein Feld der Matrix. Hell = viel "
        f"Aufmerksamkeit.{RESET}")
    for base in range(0, nH, per_row):
        group = list(range(base, min(base + per_row, nH)))
        say("")
        say("   " + "  ".join(f"{DIM}H{h:<{T - 1}}{RESET}" for h in group))
        for r in range(T):
            cells = []
            for h in group:
                row = "".join(
                    f"{DIM}·{RESET}" if A[L, h, r, c] < 0.02 else
                    f"{_c('38;5;' + str(RAMP[min(len(RAMP) - 1, int(math.sqrt(A[L, h, r, c]) * (len(RAMP) - 1)))]))}█{RESET}"
                    for c in range(T))
                cells.append(row)
            say("   " + "  ".join(cells))


def scan(A, ids, text, qword, kword, topn=6):
    """Rank all 480 heads by attention from word A onto word B."""
    qs = word_span(text, ids, qword, last=True)
    ks = word_span(text, ids, kword)
    if not qs or not ks:
        say(f"  {RED}'{qword}' oder '{kword}' nicht im Satz gefunden.{RESET}")
        return None
    q = qs[-1]                      # the word is only 'complete' at its last token
    sc = A[:, :, q, ks].sum(-1)     # onto the whole target word
    nH = A.size(1)
    v, idx = sc.flatten().topk(topn)
    tk = pieces(ids)
    say("")
    say(f"  {BOLD}Suche:{RESET} Wer schaut von {YELLOW}«{qword}»{RESET} "
        f"(Token {q}: {tk[q]}) auf {GREEN}«{kword}»{RESET} "
        f"(Token {ks[0]}-{ks[-1]})?")
    say(f"  {DIM}Durchsucht: {A.size(0)}x{nH} = {A.size(0)*nH} Köpfe{RESET}")
    say("")
    for s, k in zip(v.tolist(), idx.tolist()):
        bar = "█" * int(s * 44)
        col = GREEN if s > 0.6 else (YELLOW if s > 0.3 else DIM)
        say(f"    L{k // nH:>2} H{k % nH:<2}  {col}{s:.3f}{RESET}  {col}{bar}{RESET}")
    return int(idx[0]) // nH, int(idx[0]) % nH


# ---------------------------------------------------------------------------
# Demo chapters
# ---------------------------------------------------------------------------

SENT = "Der Arzt sagte, dass er müde war."


def demo(model, cap, dev, cfg):
    nL, nH = cfg["num_layers"], cfg["num_heads"]
    ids = ENC.encode_ordinary(SENT)
    A = attention(model, cap, dev, ids)
    tk = pieces(ids)

    head("MULTI-HEAD ATTENTION — 480 Köpfe, einer nach dem anderen")
    say(f"{DIM}  Modell: {CHECKPOINT}, {nL} Schichten x {nH} Köpfe "
        f"= {nL*nH} Köpfe{RESET}")
    say(f"{DIM}  Satz:   «{SENT}» -> {len(ids)} Tokens{RESET}")
    pause(2.5)

    head("1 · DIE MATRIX — und warum sie nur halb gefüllt ist")
    say(f"{DIM}  Jedes Token fragt jedes frühere Token: wie wichtig bist du "
        f"für mich?{RESET}")
    say(f"{DIM}  Das obere Dreieck ist leer, weil die Maske es VOR dem "
        f"Softmax auf -inf{RESET}")
    say(f"{DIM}  setzt. Ein Token kann nicht in die Zukunft schauen — sonst "
        f"wäre das{RESET}")
    say(f"{DIM}  Training betrogen: die Antwort stünde in der Eingabe.{RESET}")
    matrix(A, 11, 13, tk, "Previous-Token-Kopf")
    say("")
    # Read the sub-diagonal from the matrix itself, so the text can never
    # drift from the numbers shown above it.
    sub = ", ".join(f"{round(A[11, 13, r, r - 1].item() * 100)}" for r in range(1, 7))
    say(f"  {YELLOW}Schau auf die Nebendiagonale:{RESET} {sub} ...")
    say(f"  {DIM}Dieser Kopf tut fast nichts anderes, als das unmittelbar "
        f"vorherige{RESET}")
    say(f"  {DIM}Token anzusehen. Das hat ihm niemand gesagt.{RESET}")
    pause(4)

    head("2 · WARUM 'MULTI' — dieselbe Schicht, 20 verschiedene Muster")
    head_strip(A, 11, tk)
    say("")
    say(f"  {DIM}Alle 20 sehen denselben Satz und rechnen gleichzeitig. "
        f"Würden sie{RESET}")
    say(f"  {DIM}dasselbe tun, wäre 19-mal Rechenzeit verschenkt. Tun sie "
        f"nicht.{RESET}")
    pause(4)

    head("3 · DER PARKPLATZ — wohin die meiste Aufmerksamkeit wirklich geht")
    sink = A[:, :, :, 0].mean(-1)
    say("")
    say(f"  {DIM}Anteil der Aufmerksamkeit, der auf Token 0 landet "
        f"(«{tk[0]}»):{RESET}")
    say("")
    for L in range(0, nL, 3):
        m = sink[L].median().item()
        bar = "█" * int(m * 48)
        col = RED if m > 0.7 else (YELLOW if m > 0.4 else GREEN)
        say(f"    Layer {L:>2}   {col}{m:.2f}{RESET}  {col}{bar}{RESET}")
    say("")
    say(f"  {BOLD}Mittel über alle {nL*nH} Köpfe: "
        f"{sink.mean().item():.1%}{RESET}")
    say(f"  {DIM}Der Grund ist der Softmax: er MUSS auf 1 summieren. Ein "
        f"Kopf kann also{RESET}")
    say(f"  {DIM}nicht 'nichts' ausgeben — er muss sein Gewicht irgendwo "
        f"hinlegen. Token 0{RESET}")
    say(f"  {DIM}ist das einzige, das jede Anfrage sehen kann, und das "
        f"Modell hat gelernt,{RESET}")
    say(f"  {DIM}dessen Wert-Vektor harmlos zu halten. Damit wird es zum "
        f"Parkplatz.{RESET}")
    say(f"  {DIM}Man nennt das 'attention sink'. Es heißt: die meisten "
        f"Köpfe sind{RESET}")
    say(f"  {DIM}die meiste Zeit im Leerlauf — und das ist kein Fehler.{RESET}")
    pause(5)

    head("4 · KÖPFE MIT JOB — selbst suchen statt glauben")
    say(f"{DIM}  «{SENT}»{RESET}")
    say(f"{DIM}  Wer verbindet das Pronomen 'er' mit seinem Bezugswort "
        f"'Arzt'?{RESET}")
    best = scan(A, ids, SENT, "er", "Arzt")
    if best:
        matrix(A, best[0], best[1], tk, "der Kandidat aus der Suche")
    say("")
    say(f"  {DIM}Gegenprobe mit weiblichem Bezug — derselbe Kopf?{RESET}")
    s2 = "Die Ärztin sagte, dass sie müde war."
    i2 = ENC.encode_ordinary(s2)
    A2 = attention(model, cap, dev, i2)
    scan(A2, i2, s2, "sie", "Ärztin", topn=4)
    say("")
    say(f"  {DIM}Und der berühmteste Kopf-Typ überhaupt — Induction: "
        f"«... A B ... A» ->{RESET}")
    say(f"  {DIM}wer schaut vom zweiten A auf B? Das ist der Mechanismus "
        f"hinter{RESET}")
    say(f"  {DIM}In-Context-Learning: Muster fortsetzen, das man gerade "
        f"erst gesehen hat.{RESET}")
    s3 = "Anna ging nach Hause. Anna"
    i3 = ENC.encode_ordinary(s3)
    A3 = attention(model, cap, dev, i3)
    scan(A3, i3, s3, "Anna", "ging", topn=4)
    pause(4)

    head("5 · DIE KONTROLLE, DIE UNSERE HYPOTHESE ERLEDIGT HAT")
    control(model, cap, dev)
    pause(3)

    head("WAS MAN DARAUS MITNIMMT")
    say("")
    say(f"  {GREEN}1.{RESET} Attention ist eine Tabelle, kein Zauber: "
        f"jedes Token gewichtet")
    say(f"     alle früheren. Die Maske macht daraus ein Dreieck.")
    say(f"  {GREEN}2.{RESET} 'Multi' heißt Arbeitsteilung — 20 Köpfe pro "
        f"Schicht, verschiedene Muster.")
    say(f"  {GREEN}3.{RESET} Die Mehrheit parkt. {sink.mean().item():.0%} "
        f"der Aufmerksamkeit landet auf Token 0.")
    say(f"  {GREEN}4.{RESET} Einzelne Köpfe haben klare Jobs — Vorgänger, "
        f"Bezugswort, Wiederholung.")
    say(f"  {GREEN}5.{RESET} Und man täuscht sich leicht. Ohne Kontrollsatz "
        f"ist eine Attention-")
    say(f"     Karte eine Einladung, Bedeutung zu sehen, wo keine ist.")
    say("")
    say(f"  {DIM}Alles nachrechenbar: python explain/attention_de.py "
        f"\"dein Satz\"{RESET}")
    say("")


def control(model, cap, dev):
    """The falsified hypothesis — kept in the demo on purpose."""
    say("")
    say(f"  {DIM}Erster Befund: in «Er machte die Tür auf» schaut 'auf' "
        f"mit 0.93 auf{RESET}")
    say(f"  {DIM}'machte' zurück. Trennbares Verb, sauber erkannt — dachten "
        f"wir.{RESET}")
    say(f"  {DIM}Kontrolle: derselbe Kopf, aber 'auf' als bloße "
        f"Präposition.{RESET}")
    say("")
    cases = [
        ("Er machte gestern die Tür auf.", "auf", "machte", "Verbpräfix", True),
        ("Er stieg auf den Berg.", "auf", "stieg", "Präposition", False),
        ("Das Buch liegt auf dem Tisch.", "auf", "liegt", "Präposition", False),
        ("Sie wartete auf den Zug.", "auf", "wartete", "Präposition", False),
    ]
    for text, qw, kw, kind, is_prefix in cases:
        ids = ENC.encode_ordinary(text)
        A = attention(model, cap, dev, ids)
        qs = word_span(text, ids, qw, last=True)
        ks = word_span(text, ids, kw)
        v = A[15, 18, qs[-1], ks].sum().item()
        col = YELLOW if is_prefix else RED
        say(f"    {col}{kind:<12}{RESET} {v:.3f} {col}{'█' * int(v * 40)}"
            f"{RESET}  {DIM}«{text}»{RESET}")
    say("")
    say(f"  {BOLD}{RED}Gleich stark.{RESET} L15 H18 ist kein "
        f"Trennbare-Verben-Kopf. Er verbindet")
    say(f"  {DIM}ein Wort mit dem finiten Verb bzw. dem Kopf seines "
        f"Teilsatzes — die{RESET}")
    say(f"  {DIM}Wortart ist ihm egal. Die schöne Interpretation war "
        f"unsere, nicht seine.{RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

model, cap, dev, cfg = load()
args = [a for a in sys.argv[1:] if a.strip()]

if CONTROL:
    head("KONTROLLE — die widerlegte Hypothese")
    control(model, cap, dev)
    say("")
elif DEMO:
    demo(model, cap, dev, cfg)
elif args or SCAN or HEADS:
    text = args[0] if args else SENT
    ids = ENC.encode_ordinary(text)
    A = attention(model, cap, dev, ids)
    tk = pieces(ids)
    say("")
    say(f"{BOLD}{YELLOW}» {text}{RESET}  {DIM}({len(ids)} Tokens){RESET}")
    if SCAN:
        q, k = os.environ.get("Q"), os.environ.get("K")
        if not q or not k:
            say(f"{RED}SCAN=1 braucht Q=... und K=... "
                f"(z.B. Q=er K=Arzt){RESET}")
        else:
            best = scan(A, ids, text, q, k)
            if best:
                matrix(A, best[0], best[1], tk, "bester Kopf aus der Suche")
    elif HEADS:
        head_strip(A, LAYER, tk)
    else:
        matrix(A, LAYER, HEAD, tk)
    say("")
else:
    say("")
    say(f"{DIM}Satz eingeben, Enter. Leere Zeile beendet.{RESET}")
    say(f"{DIM}Kopf wählen über LAYER=/HEAD= (aktuell L{LAYER} H{HEAD}).{RESET}")
    while True:
        try:
            text = input(f"{BOLD}{CYAN}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            say("")
            break
        if not text.strip():
            break
        ids = ENC.encode_ordinary(text)
        A = attention(model, cap, dev, ids)
        matrix(A, LAYER, HEAD, pieces(ids))
say("")
