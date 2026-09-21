"""Was das Modell für ein Wort gelernt hat — als Nachbarschaft im Vektorraum.

Jedes Token besitzt eine Zeile in der Embedding-Matrix: 1280 Zahlen,
gelernt aus dem Trainingstext. Niemand hat dem Modell gesagt, was Berlin
ist. Trotzdem liegen Hamburg, Dresden und Frankfurt danach ganz in der
Nähe — weil sie in ähnlichen Sätzen vorkommen. Dieses Skript macht genau
das sichtbar: Token rein, Vektor und nächste Nachbarn raus.

Die Matrix ist WEIGHT-TIED (train.py: lm_head teilt die Gewichte mit dem
Embedding). Dieselbe Zeile ist also Eingangs-Bedeutung UND Ausgangs-
Vorhersage. "Nahe" heißt damit beides: ähnlicher Kontext beim Lesen und
verwechselbar beim Schreiben.

DER INTERESSANTE FALL — und die Brücke zum Tokenizer: viele deutsche
Alltagswörter haben ÜBERHAUPT KEINEN eigenen Vektor. Von 30 getesteten
Wörtern sind 20 mehrteilig. " Wasser" ist nicht ein Vektor, sondern
" W" + "asser". " Stadt" braucht drei. Das Modell besitzt keine
Vorstellung von "Wasser" — es besitzt zwei Fragmente, die es jedes Mal
neu zusammensetzen muss. Genau das kostet der englische Tokenizer
(siehe tokenize_de.py).

ÄHNLICHKEIT: rohe Cosinus-Ähnlichkeit. Zwei Vorbehalte, ehrlich gesagt:
  1. Der Raum ist anisotrop — alle Vektoren teilen eine gemeinsame
     Richtung (Norm des Mittelvektors 0,42 gegen Median-Tokennorm 0,83),
     was jede Ähnlichkeit nach oben zieht. CENTER=1 zieht den
     Mittelvektor ab. Das säubert seltene Tokens (" Paris" -> London,
     Berlin, Frankfurt), macht aber häufige schlechter, weil dann
     Einzelbuchstaben zu Universal-Nachbarn werden. Deshalb ist es
     eine Option und nicht der Standard.
  2. Seltene Tokens haben verrauschte Nachbarschaften. Der Frequenz-
     filter (MIN_COUNT) lässt nur Tokens als Nachbarn zu, die in
     echtem deutschen Text oft genug vorkommen — gezählt auf
     eval_data/fineweb.jsonl. Ohne ihn stehen Zahlen-Tokens in jeder
     Liste. Die Nachbarschaftsqualität verrät damit selbst, wie oft das
     Modell ein Token gesehen hat.

Usage:
    python explain/neighbors_de.py                    # interaktiv
    python explain/neighbors_de.py " Berlin" " Auto"   # direkt (Leerzeichen zählt!)
    DEMO=1 python explain/neighbors_de.py             # kuratiert, drei Kapitel
    TABLE=1 DEMO=1 python explain/neighbors_de.py     # kompakte Tabelle, eine
                                              # Zeile pro TOKEN
    TABLE=1 TABLE_NB=4 python explain/neighbors_de.py " Kind" " Wasser"
    VIDEO=1 DEMO=1 python explain/neighbors_de.py     # mit Pausen fürs Recording
    TOPK=10 python explain/neighbors_de.py            # mehr Nachbarn
    CENTER=1 python explain/neighbors_de.py           # Mittelvektor abziehen
    MIN_COUNT=0 python explain/neighbors_de.py        # Frequenzfilter aus
    CHECKPOINT=weights_540m_hq_fp16.pt python explain/neighbors_de.py

Hinweis: das führende Leerzeichen ist Teil des Tokens. " Berlin" und
"Berlin" sind verschiedene Tokens mit verschiedenen Vektoren — das
Skript probiert beides und nimmt, was existiert.
"""

import collections
import json
import os
import sys
import time

import tiktoken
import torch
import torch.nn.functional as F

# model/ on sys.path — the repo layout lives in model/paths.py, and
# anchoring to __file__ is what lets this script run from any directory.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
from paths import EVAL_DATA, ckpt_path, data_path  # noqa: E402

CHECKPOINT = ckpt_path(os.environ.get("CHECKPOINT", "weights_540m_fp16.pt"))
TOPK = int(os.environ.get("TOPK", "8"))
MIN_COUNT = int(os.environ.get("MIN_COUNT", "5"))
CENTER = os.environ.get("CENTER", "0") in ("1", "true")
TABLE = os.environ.get("TABLE", "0") in ("1", "true")
TABLE_NB = int(os.environ.get("TABLE_NB", "3"))   # Nachbarn pro Zeile
VIDEO = os.environ.get("VIDEO", "0") in ("1", "true")
DEMO = os.environ.get("DEMO", "0") in ("1", "true")
PLAIN = os.environ.get("NO_COLOR", "0") in ("1", "true")
FREQ_CACHE = data_path("neighbors_freq_cache.json")
DIMS_SHOWN = 48          # so viele Vektor-Dimensionen als Balken


def _c(code):
    return "" if PLAIN else f"\033[{code}m"


RESET, BOLD, DIM = _c(0), _c(1), _c(2)
CYAN, YELLOW, GREEN, RED, MAG = _c(96), _c(93), _c(92), _c(91), _c(95)


def say(text, end="\n"):
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
            time.sleep(0.01)
            i += 1
        sys.stdout.write(end)
        sys.stdout.flush()
    else:
        print(text, end=end)


def pause(sec):
    if VIDEO:
        time.sleep(sec)


enc = tiktoken.get_encoding("gpt2")

say(f"{DIM}Lade {CHECKPOINT} ...{RESET}")
sd = torch.load(CHECKPOINT, map_location="cpu",
                weights_only=False)["model_state_dict"]
E = sd["embedding.embedding.weight"].float()
tied = torch.equal(E, sd["lm_head.weight"].float())
del sd

# Frequenzen auf echtem deutschem Text zählen (einmal, dann gecacht).
# Das ist die Kandidatenliste für Nachbarn: was hier nicht vorkommt, hat
# das Modell im Deutschen kaum gesehen und liegt entsprechend zufällig.
if MIN_COUNT > 0:
    if os.path.exists(FREQ_CACHE):
        with open(FREQ_CACHE) as f:
            counts = {int(k): v for k, v in json.load(f).items()}
    else:
        say(f"{DIM}Zähle Token-Frequenzen auf eval_data/fineweb.jsonl "
            f"(einmalig) ...{RESET}")
        counts = collections.Counter()
        with open(os.path.join(EVAL_DATA, "fineweb.jsonl")) as f:
            for line in f:
                counts.update(enc.encode(json.loads(line)["text"]))
        counts = dict(counts)
        with open(FREQ_CACHE, "w") as f:
            json.dump(counts, f)
    candidates = torch.tensor(sorted(t for t, c in counts.items()
                                     if c >= MIN_COUNT))
else:
    counts = {}
    candidates = torch.arange(E.shape[0])

M = E - E[candidates].mean(0) if CENTER else E
NORM = F.normalize(M, dim=1)
NORM_CAND = NORM[candidates]

BAR = " ▁▂▃▄▅▆▇█"


def vector_strip(vec):
    """Die ersten DIMS_SHOWN Dimensionen als Balken: blau positiv, rot negativ.

    Kein Diagramm, nur ein Fingerabdruck — man soll sehen, DASS es 1280
    Zahlen ohne einzeln lesbare Bedeutung sind. Die Bedeutung steckt
    nicht in einer Dimension, sondern in der Richtung des Ganzen.
    """
    v = vec[:DIMS_SHOWN]
    # Auf die Streuung skalieren, nicht aufs Maximum: einzelne Dimensionen
    # sind massive Ausreisser (bekannt bei Transformern) und wuerden alles
    # andere platt auf null druecken. 2 Sigma, dann clippen.
    scale = 2.0 * vec.std().item() or 1.0
    out = []
    for x in v.tolist():
        frac = min(1.0, abs(x) / scale)
        level = BAR[int(frac * (len(BAR) - 1))]
        out.append(f"{_c(94) if x >= 0 else _c(91)}{level}")
    return "".join(out) + RESET


def neighbors(tid, k):
    sims = NORM_CAND @ NORM[tid]
    self_pos = (candidates == tid).nonzero()
    if len(self_pos):
        sims[self_pos] = -1e9
    top = sims.topk(min(k, len(sims)))
    return [(candidates[j].item(), s)
            for s, j in zip(top.values.tolist(), top.indices.tolist())]


def show_token(tid, label=None, indent=""):
    piece = enc.decode([tid])
    freq = counts.get(tid)
    freq_txt = f" · {freq}x in 1,3M Tokens" if freq else ""
    say(f"{indent}{BOLD}{YELLOW}{label or repr(piece)}{RESET}"
        f"{DIM}  ID {tid} · Norm {E[tid].norm():.2f}{freq_txt}{RESET}")
    say(f"{indent}  {DIM}Vektor ({E.shape[1]} Dim, erste {DIMS_SHOWN}):{RESET} "
        f"{vector_strip(M[tid])}")
    say(f"{indent}  {DIM}{'  '.join(f'{x:+.4f}' for x in E[tid][:6].tolist())}"
        f"  ...{RESET}")
    say(f"{indent}  {DIM}nächste Nachbarn:{RESET}")
    for nid, sim in neighbors(tid, TOPK):
        bar = "█" * max(1, int(sim * 20))
        say(f"{indent}    {GREEN}{sim:.3f}{RESET} {DIM}{bar}{RESET} "
            f"{BOLD}{enc.decode([nid])!r}{RESET}")


def show_word(word):
    """Ein Wort ansehen — mit oder ohne führendes Leerzeichen."""
    say("")
    say(f"{CYAN}{'─' * 68}{RESET}")
    ids = enc.encode(word)
    if len(ids) == 1:
        show_token(ids[0])
        pause(2.5)
        return

    # Der Lehrmoment: für dieses Wort existiert kein Vektor.
    pieces = [enc.decode([i]) for i in ids]
    say(f"{BOLD}{RED}{word!r} hat KEINEN eigenen Vektor{RESET}"
        f"{DIM} — {len(ids)} Tokens: {' + '.join(repr(p) for p in pieces)}"
        f"{RESET}")
    say(f"{DIM}Das Modell muss dieses Wort jedes Mal aus Fragmenten "
        f"zusammensetzen.{RESET}")
    say(f"{DIM}Und jedes Fragment für sich bedeutet etwas anderes:{RESET}")
    for i in ids:
        say("")
        show_token(i, indent="  ")
    pause(3)


def _vis_len(text):
    """Sichtbare Laenge — ANSI-Codes zaehlen nicht mit."""
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\033":
            i = text.index("m", i) + 1
            continue
        out += 1
        i += 1
    return out


def _pad(text, width):
    return text + " " * max(0, width - _vis_len(text))


# Spaltenbreiten: fix, damit mehrere Tabellen untereinander buendig sind.
COLS = (14, 12, 7, 6, 8)   # Wort, Token, ID, Norm, Frequenz


def table_header():
    head = (_pad("WORT", COLS[0]) + _pad("TOKEN", COLS[1])
            + _pad("ID", COLS[2]) + _pad("NORM", COLS[3])
            + _pad("FREQ", COLS[4]) + f"{TABLE_NB} NÄCHSTE NACHBARN")
    say(f"{DIM}{head}{RESET}")
    say(f"{DIM}{'─' * (sum(COLS) + 48)}{RESET}")


def table_rows(word):
    """Eine Zeile pro TOKEN — das ist die Einheit, die einen Vektor besitzt.

    Ein mehrteiliges Wort belegt entsprechend mehrere Zeilen; die
    Wort-Spalte steht dann nur in der ersten und traegt den Zaehler
    (1/2, 2/2), damit sichtbar bleibt: hier gibt es keinen Wortvektor,
    sondern zwei Fragmentvektoren.
    """
    ids = enc.encode(word)
    for n, tid in enumerate(ids):
        if n == 0:
            first = f"{BOLD}{YELLOW}{word.strip()}{RESET}"
            if len(ids) > 1:
                first += f"{RED}✗{RESET}"     # kein eigener Vektor
        else:
            first = ""
        marker = f"{DIM}{n + 1}/{len(ids)}{RESET} " if len(ids) > 1 else ""
        freq = counts.get(tid)
        nbs = "  ".join(
            f"{BOLD}{enc.decode([j])!r}{RESET} {DIM}{sim:.2f}{RESET}"
            for j, sim in neighbors(tid, TABLE_NB))
        say(_pad(first, COLS[0])
            + _pad(f"{marker}{CYAN}{enc.decode([tid])!r}{RESET}", COLS[1])
            + _pad(f"{DIM}{tid}{RESET}", COLS[2])
            + _pad(f"{E[tid].norm():.2f}", COLS[3])
            + _pad(f"{DIM}{str(freq) + 'x' if freq else '–'}{RESET}", COLS[4])
            + nbs)
    pause(0.6)


DEMO_GROUPS = [
    ("KAPITEL 1 — WAS DAS MODELL VON SELBST GELERNT HAT",
     "Niemand hat ihm Geografie beigebracht. Es hat nur Text gelesen.",
     [" Kind", " Berlin", " Auto", " Paris"]),
    ("KAPITEL 2 — WÖRTER OHNE VEKTOR",
     "20 von 30 deutschen Alltagswörtern haben keinen eigenen Vektor.",
     [" Wasser", " Stadt", " Mensch"]),
    ("KAPITEL 3 — WAS EIN FRAGMENT STATTDESSEN LERNT",
     "Hier gruppiert sich Schreibweise, nicht Bedeutung.",
     ["ung", "vers"]),
]

say("")
say(f"{CYAN}{'=' * 68}{RESET}")
say(f"{CYAN}{BOLD}  EMBEDDINGS — was 1280 Zahlen über ein Wort wissen{RESET}")
say(f"{CYAN}{'=' * 68}{RESET}")
say(f"{DIM}  {os.path.basename(CHECKPOINT)} · {E.shape[0]} Tokens × "
    f"{E.shape[1]} Dimensionen{RESET}")
say(f"{DIM}  weight-tied mit lm_head: {'ja' if tied else 'nein'} — dieselbe "
    f"Zeile ist Bedeutung und Vorhersage{RESET}")
if MIN_COUNT:
    say(f"{DIM}  Nachbarn nur aus {len(candidates)} Tokens mit ≥ {MIN_COUNT} "
        f"Vorkommen in echtem deutschen Text{RESET}")
if CENTER:
    say(f"{DIM}  Mittelvektor abgezogen (CENTER=1){RESET}")
pause(2)

args = [a for a in sys.argv[1:]]
if args and TABLE:
    say("")
    table_header()
    for w in args:
        table_rows(w)
elif args:
    for w in args:
        show_word(w)
elif DEMO and TABLE:
    for title, sub, words in DEMO_GROUPS:
        say("")
        say(f"{MAG}{BOLD}  {title}{RESET}")
        say(f"{DIM}  {sub}{RESET}")
        say("")
        table_header()
        for w in words:
            table_rows(w)
elif DEMO:
    for title, sub, words in DEMO_GROUPS:
        say("")
        say(f"{MAG}{BOLD}  {title}{RESET}")
        say(f"{DIM}  {sub}{RESET}")
        pause(1.5)
        for w in words:
            show_word(w)
    say("")
    say(f"{DIM}Die Nachbarschaft ist gelernt, nicht programmiert — und sie{RESET}")
    say(f"{DIM}existiert nur für Tokens, die der Tokenizer ganz gelassen hat.{RESET}")
    say("")
    # Zwei Dinge, die in der Ausgabe oben stehen und der Ehrlichkeit dienen:
    say(f"{DIM}Zwei Beobachtungen gegen die einfache Geschichte:{RESET}")
    say(f"{DIM}  · Die Frequenz erklärt die Qualität. ' Berlin' (189x) hat "
        f"saubere{RESET}")
    say(f"{DIM}    Nachbarn, ' Paris' (25x) nur die ersten drei, dann "
        f"Rauschen.{RESET}")
    say(f"{DIM}  · Nicht jedes Fragment ist sinnlos: ' Mens' liegt neben "
        f"' Person'{RESET}")
    say(f"{DIM}    und ' Patient' — es kommt fast nur in \"Mensch\" vor. "
        f"Eindeutige{RESET}")
    say(f"{DIM}    Fragmente tragen Bedeutung, mehrdeutige wie 'ung' nicht.{RESET}")
else:
    say("")
    say(f"{DIM}Wort eingeben (führendes Leerzeichen wird automatisch "
        f"probiert). Leere Zeile beendet.{RESET}")
    if TABLE:
        say("")
        table_header()
    while True:
        try:
            raw = input(f"{BOLD}{CYAN}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            say("")
            break
        if not raw.strip():
            break
        # " Berlin" ist ein anderes Token als "Berlin" — nimm die Variante,
        # die als EIN Token existiert, sonst das Wort wie eingegeben.
        word = raw if raw.startswith(" ") else " " + raw
        if len(enc.encode(word)) > 1 and len(enc.encode(raw)) == 1:
            word = raw
        if TABLE:
            table_rows(word)
        else:
            show_word(word)

say("")
