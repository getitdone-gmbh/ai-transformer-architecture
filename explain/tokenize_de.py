"""Was der Tokenizer mit deutschen Woertern macht — zum Anschauen.

Das Modell sieht nie Buchstaben. Es sieht Token-IDs. Welche Stuecke der
Tokenizer aus einem Wort schneidet, entscheidet damit ueber alles, was
danach kommt: Kontextlaenge, Rechenkosten, und wie schwer ein Wort
ueberhaupt zu lernen ist.

WICHTIG, und der Grund fuer dieses Skript: unser Tokenizer ist der
GPT-2-BPE (tiktoken, 50.257 Tokens). Der wurde GELERNT — aber auf
ueberwiegend englischem Web-Text, nicht von uns und nicht auf Deutsch.
Deutsch faellt bei ihm durch das Raster:

    the capital of France             ->  4 Tokens  (5.2 Zeichen/Token)
    Donaudampfschifffahrtsgesellschaft -> 14 Tokens  (2.4 Zeichen/Token)

Beides ist derselbe Tokenizer. Deutsch kostet ihn gut doppelt so viele
Tokens pro Zeichen. Praktisch heisst das: dasselbe Trainingsbudget
kauft auf Deutsch nur die halbe Textmenge, und ein deutsches Wort muss
das Modell aus 14 zusammenhanglosen Fragmenten rekonstruieren statt aus
einem Stueck. Umlaute sind der Extremfall — 'ae' ist in UTF-8 zwei Bytes
und wird zu einem eigenen Token, mitten im Wort.

Gegen einen mehrsprachigen Tokenizer gemessen (COMPARE=1, Qwen2.5 mit
152k Vokabular), auf zusammenhaengendem Fliesstext:

    Deutsch:  GPT-2 2.64 Zeichen/Token  vs  Qwen 3.58  -> 1.36x mehr Tokens
    Englisch: GPT-2 5.69 Zeichen/Token  vs  Qwen 5.69  -> gleichauf

Der Aufschlag entsteht also nicht durch die Modellgroesse, sondern durch
die Sprache, auf der das Vokabular gelernt wurde. An Extremkomposita
sieht man das uebrigens NICHT — dort liegen beide fast gleichauf
(14 vs 13 Tokens), weil auch Qwen sie zerlegen muss. Der Unterschied
sitzt im normalen Text.

Genau das ist auch der Grund, warum compare_bpb.py in Bits-pro-Byte
misst statt in Loss-pro-Token: sonst vergleicht man Tokenizer, nicht
Modelle.

Usage:
    python explain/tokenize_de.py                    # interaktiv: Wort rein, Zerlegung raus
    python explain/tokenize_de.py "Krankenkasse"     # direkt ein Wort/Satz
    DEMO=1 python explain/tokenize_de.py             # kuratierte Wortliste (Screen-Recording)
    VIDEO=1 DEMO=1 python explain/tokenize_de.py     # dito mit Pausen
    COMPARE=1 python explain/tokenize_de.py          # Qwen-Tokenizer (152k, mehrsprachig)
                                             # zum Vergleich daneben
    IDS=1 python explain/tokenize_de.py              # zusaetzlich die Token-IDs zeigen
"""

import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import tiktoken

# model/ on sys.path — the repo layout lives in model/paths.py, and
# anchoring to __file__ is what lets this script run from any directory.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
from paths import data_path  # noqa: E402

VIDEO = os.environ.get("VIDEO", "0") in ("1", "true")
DEMO = os.environ.get("DEMO", "0") in ("1", "true")
COMPARE = os.environ.get("COMPARE", "0") in ("1", "true")
SHOW_IDS = os.environ.get("IDS", "0") in ("1", "true")
PLAIN = os.environ.get("NO_COLOR", "0") in ("1", "true")


def _c(code):
    return "" if PLAIN else f"\033[{code}m"


RESET, BOLD, DIM = _c(0), _c(1), _c(2)
CYAN, YELLOW, GREEN, RED = _c(96), _c(93), _c(92), _c(91)

# Abwechselnde Hintergruende: die Token-Grenzen muessen auf einem
# Screen-Recording ohne Erklaerung sichtbar sein. Heller Text auf
# dunklem Grund funktioniert in hellen wie dunklen Terminals.
TOKEN_BG = [_c("48;5;24;97"), _c("48;5;22;97"),
            _c("48;5;53;97"), _c("48;5;58;97")]

# Kuratierte Demo — jede Zeile macht einen anderen Punkt.
DEMO_WORDS = [
    # 1. Der Aufhaenger: Englisch billig, Deutsch teuer.
    "the capital of France",
    "die Hauptstadt von Frankreich",
    # 2. Komposita — die deutsche Spezialitaet zerfaellt in Staub.
    "Donaudampfschifffahrtsgesellschaft",
    "Geschwindigkeitsbegrenzung",
    "Krankenversicherung",
    # 3. Umlaute und ß: eigene Tokens mitten im Wort.
    "Fußgängerüberweg",
    "Bäckerei",
    # 4. Gegenprobe: englische Woerter derselben Laenge bleiben ganz.
    "unbelievable",
    "understanding",
    # 5. Fliesstext statt Einzelwoerter — erst hier wird der Vergleich
    #    mit Qwen (COMPARE=1) ehrlich. Am Extremkompositum liegen beide
    #    Tokenizer fast gleichauf (14 vs 13); auf zusammenhaengendem
    #    Text zieht Qwens mehrsprachiges 152k-Vokabular davon: GEMESSEN
    #    1.36x mehr Tokens fuer GPT-2 auf Deutsch, 1.00x auf Englisch.
    "Die Bundesregierung hat gestern eine neue Verordnung zur "
    "Krankenversicherung beschlossen.",
    "The federal government yesterday passed a new regulation on "
    "health insurance.",
]


def say(text, end="\n"):
    """Typewriter im VIDEO-Modus. ANSI-Codes werden nicht 'getippt'."""
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
            time.sleep(0.012)
            i += 1
        sys.stdout.write(end)
        sys.stdout.flush()
    else:
        print(text, end=end)


def pause(sec):
    if VIDEO:
        time.sleep(sec)


def render(raw):
    """bytes eines Tokens -> anzeigbarer Text.

    Zwei Faelle, die man sehen koennen muss:
      - Leerzeichen sind Teil des Tokens (GPT-2 haengt sie vorne an) und
        wuerden sonst unsichtbar mit dem Hintergrund verschmelzen -> '.'
      - Ein Token kann mitten in einem UTF-8-Zeichen enden, ist also fuer
        sich genommen gar kein Zeichen -> als Hex-Byte zeigen. Das ist
        kein Anzeigefehler, das ist die Realitaet des byte-level BPE.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"\\x{b:02x}" for b in raw)
    return text.replace(" ", "·")


def split(enc_name, encode_fn, piece_fn, text):
    """Eine Zeile Zerlegung + eine Zeile Statistik."""
    ids = encode_fn(text)
    pieces = [render(piece_fn(i)) for i in ids]

    boxes = "".join(f"{TOKEN_BG[n % len(TOKEN_BG)]}{p}{RESET}"
                    for n, p in enumerate(pieces))
    say(f"  {DIM}{enc_name:<12}{RESET}{boxes}")

    if SHOW_IDS:
        say(f"  {' ' * 12}{DIM}{' '.join(str(i) for i in ids)}{RESET}")

    # Zeichen, nicht Bytes: "wie viel Wort steckt in einem Token" ist die
    # Frage, die der Leser im Kopf hat.
    per_token = len(text) / len(ids) if ids else 0.0
    # Ab ~4 Zeichen/Token arbeitet der Tokenizer normal (englischer
    # Referenzbereich), darunter zahlt man drauf.
    col = GREEN if per_token >= 4.0 else (YELLOW if per_token >= 3.0 else RED)
    say(f"  {' ' * 12}{col}{len(ids)} Tokens{RESET}{DIM} · "
        f"{len(text)} Zeichen · {per_token:.1f} Zeichen/Token{RESET}")
    return len(ids)


gpt2 = tiktoken.get_encoding("gpt2")
ENCODERS = [("GPT-2 50k", gpt2.encode, gpt2.decode_single_token_bytes)]

# Der selbst trainierte deutsche BPE, falls train_tokenizer.py gelaufen ist.
# Er ist der eigentliche Punkt: gleicher Algorithmus, gleiche
# Vokabulargröße, nur auf Deutsch gelernt.
DE_BPE = data_path(os.environ.get("DE_BPE", "de_bpe.json"))
if os.path.exists(DE_BPE):
    from tokenizers import Tokenizer as HFTokenizer
    de = HFTokenizer.from_file(DE_BPE)
    ENCODERS.append((
        f"DE-BPE {de.get_vocab_size() // 1000}k",
        lambda t: de.encode(t).ids,
        lambda i: de.decode([i]).encode("utf-8"),
    ))

if COMPARE:
    from transformers import AutoTokenizer
    qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    def _qwen_piece(i):
        # convert_ids_to_tokens liefert die BPE-Darstellung mit 'G'-Marker
        # fuer Leerzeichen; ueber die Bytes bleibt die Anzeige einheitlich
        # mit der GPT-2-Spalte.
        return qwen.decode([i]).encode("utf-8")

    ENCODERS.append(("Qwen 152k", lambda t: qwen(t).input_ids, _qwen_piece))


def show(text):
    say("")
    say(f"{BOLD}{YELLOW}» {text}{RESET}")
    counts = [split(name, enc, piece, text) for name, enc, piece in ENCODERS]
    # Immer gegen Spalte 1 (GPT-2) — das ist der Tokenizer, den unser
    # Modell wirklich benutzt, also die Referenz, gegen die alles zählt.
    for (name, _, _), n in zip(ENCODERS[1:], counts[1:]):
        d = counts[0] - n
        if d == 0:
            say(f"  {' ' * 12}{DIM}-> GPT-2 braucht gleich viele Tokens "
                f"wie {name}{RESET}")
            continue
        tok_word = "Token" if abs(d) == 1 else "Tokens"
        more = "mehr" if d > 0 else "weniger"
        say(f"  {' ' * 12}{DIM}-> GPT-2 braucht {abs(d)} {tok_word} {more} "
            f"als {name} ({counts[0] / n:.2f}x){RESET}")
    pause(2.2)


say("")
say(f"{CYAN}{'=' * 72}{RESET}")
say(f"{CYAN}{BOLD}  WOHIN WOERTER ZERFALLEN — der Tokenizer unseres 540M-Modells{RESET}")
say(f"{CYAN}{'=' * 72}{RESET}")
say(f"{DIM}  GPT-2 BPE, 50.257 Tokens. Gelernt auf englischem Web-Text.{RESET}")
if any(n.startswith("DE-BPE") for n, _, _ in ENCODERS):
    say(f"{DIM}  Darunter: gleicher Algorithmus, gleiche Vokabulargröße —{RESET}")
    say(f"{DIM}  aber auf DEUTSCH gelernt (train_tokenizer.py).{RESET}")
say(f"{DIM}  Farbwechsel = Token-Grenze. '·' = Leerzeichen im Token.{RESET}")
pause(2)

args = [a for a in sys.argv[1:] if a.strip()]

if args:
    for text in args:
        show(text)
elif DEMO:
    for word in DEMO_WORDS:
        show(word)
    say("")
    if len(ENCODERS) > 1:
        # Die eigentliche Pointe steht in den letzten zwei Zeilen der Demo:
        # der deutsche Tokenizer ist auf Englisch genau so schlecht wie
        # GPT-2 auf Deutsch. Kein Tokenizer ist "besser" — jeder ist auf
        # die Sprache optimiert, auf der er gelernt hat.
        say(f"{DIM}Der deutsche Tokenizer ist auf ENGLISCH genau so schlecht"
            f"{RESET}")
        say(f"{DIM}wie GPT-2 auf Deutsch. Keiner ist besser — jeder passt zu{RESET}")
        say(f"{DIM}der Sprache, auf der er gelernt hat. Deshalb ist der{RESET}")
        say(f"{DIM}Tokenizer eine Entscheidung, keine technische Qualität.{RESET}")
    else:
        say(f"{DIM}Derselbe Tokenizer, zwei Sprachen. Deutsch kostet rund die"
            f"{RESET}")
        say(f"{DIM}doppelte Anzahl Tokens pro Zeichen — also die halbe "
            f"Textmenge{RESET}")
        say(f"{DIM}pro Trainingsbudget und pro Kontextfenster.{RESET}")
else:
    say("")
    say(f"{DIM}Wort oder Satz eingeben, Enter. Leere Zeile beendet.{RESET}")
    while True:
        try:
            text = input(f"{BOLD}{CYAN}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            say("")
            break
        if not text.strip():
            break
        show(text)

say("")
