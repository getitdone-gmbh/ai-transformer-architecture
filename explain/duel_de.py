"""German duel: 540M (base) vs 540M-HQ (continued pretraining) vs Qwen2.5-0.5B.

Three-way on the base sections — the HQ column shows what +5B
quality-filtered tokens bought (facts!). Fair pairings:

  A) Completions/facts/python/english (base vs base vs base):
     weights_540m_fp32.pt | weights_540m_hq_fp32.pt | Qwen2.5-0.5B
  B) Questions (chat vs chat, two-way):
     sft_540m_v2.pt vs Qwen2.5-0.5B-Instruct — the HQ base has no SFT
     yet (that is the upcoming tool-calling chapter), so it sits out.

Decoding is pure greedy for everyone — deterministic, anyone can rerun
this and get byte-identical output. That is the point.

Usage:
    python explain/duel_de.py                 # everything, plain output
    VIDEO=1 python explain/duel_de.py         # typewriter + pauses (screen recording)
    SECTION=fakten python explain/duel_de.py  # one section only
    (sections: complete, fakten, python, english, chat)
"""

import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import tiktoken
import torch

# model/ on sys.path — the repo layout lives in model/paths.py, and
# anchoring to __file__ is what lets this script run from any directory.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
import train  # noqa: E402
from paths import ckpt_path  # noqa: E402

VIDEO = os.environ.get("VIDEO", "0") in ("1", "true")
SECTION = os.environ.get("SECTION", "all")
MAX_NEW = int(os.environ.get("MAX_NEW", "60"))
# Pure greedy, NO repetition penalty: it would penalize tokens that appear
# in the prompt — "Wer zuerst kommt," + penalty turns the correct
# continuation "mahlt zuerst" into "mahlt zu." (measured, not theory).

# SKIP_HQ=1 hides the HQ column — video 1 tells the story of run 1
# (540M vs Qwen); the HQ model is the star of video 2.
BASE_MODELS = [
    ("540M    ", ckpt_path("weights_540m_fp32.pt")),
]
if os.environ.get("SKIP_HQ", "0") not in ("1", "true"):
    BASE_MODELS.append(("540M HQ ", ckpt_path("weights_540m_hq_fp32.pt")))

COMPLETION_PROMPTS = [
    "Das größte Landtier der Erde ist der",
    "Bienen produzieren",
    "Die drei größten Städte Deutschlands sind",
    "Ein Sprichwort sagt: Wer zuerst kommt,",
    "Sehr geehrte Damen und Herren,",
    "Rezept für Apfelkuchen: Zuerst",
]

# Factual completions — two curated sets, both MEASURED on cloze_de.jsonl,
# selected for different stories:
#
# Three-way (video 2): the flip cases — the 5 items the HQ training turned
# from wrong to right, plus the 1 it LOST (Mauerfall: the HQ model now
# guesses 1961, the year the wall was BUILT — a smarter mistake, but a
# mistake; stays in for honesty).
#
# Two-way / SKIP_HQ (video 1): a REPRESENTATIVE sample instead — 3 measured
# hits + 3 measured misses of the old model (mirroring its true 57%).
# Using the flip set here would be selection bias: those items are wrong-
# by-construction for the old model and make it look artificially weak.
FAKTEN_FLIPS = [
    "Die Hauptstadt der Schweiz heißt",
    "Berlin ist die Hauptstadt von",
    "Der rote Planet in unserem Sonnensystem ist der",
    "Pflanzen erzeugen ihre Energie durch",
    "In Deutschland bezahlt man mit dem",
    "Die Berliner Mauer fiel im Jahr",
]
FAKTEN_REPRESENTATIVE = [
    "Der erste Bundeskanzler der Bundesrepublik Deutschland war Konrad",
    "Die Zauberflöte komponierte Wolfgang Amadeus",
    "Die Mona Lisa malte Leonardo da",
    "Paris liegt an der",
    "Das chemische Symbol für Gold ist",
    "Zehn mal zehn ist",
]
FAKTEN_PROMPTS = (FAKTEN_REPRESENTATIVE
                  if os.environ.get("SKIP_HQ", "0") in ("1", "true")
                  else FAKTEN_FLIPS)

QUESTIONS = [
    "Was ist die Hauptstadt von Frankreich?",
    "Was ist die Hauptstadt von Australien?",
    "Was produzieren Bienen?",
    "Wie viele Beine hat eine Spinne?",
    "Wer hat den Faust geschrieben?",
    "Nenne die drei größten Städte Deutschlands.",
    "Warum ist der Himmel blau?",
]

# Qwen's home game: 18T tokens included serious amounts of code —
# our 1B (+0.5B replay) Python tokens are a rounding error against that.
PYTHON_PROMPTS = [
    "def bubble_sort(arr):\n",
    "def is_even(n):\n    ",
    "# Gibt die Summe aller Zahlen in der Liste zurueck\ndef summe(liste):\n",
    'import os\n\nfor filename in os.listdir("."):\n    ',
]

# Our guaranteed away loss: both 540M variants saw almost no English.
ENGLISH_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "Water boils at a temperature of",
]

# CRITICAL: baked into the SFT weights — do not translate or alter.
PROMPT_TMPL = "### Frage:\n{frage}\n\n### Antwort:\n"

device = train.get_device()
enc = tiktoken.get_encoding("gpt2")

# ANSI colors (disable with NO_COLOR=1). Fixed color per model so the
# viewer can follow a column across the whole video.
PLAIN = os.environ.get("NO_COLOR", "0") in ("1", "true")


def _c(code):
    return "" if PLAIN else f"\033[{code}m"


RESET, BOLD, DIM = _c(0), _c(1), _c(2)
CYAN, YELLOW = _c(96), _c(93)
MODEL_COLORS = {"540M": _c(92), "540M HQ": _c(95), "QWEN 0.5B": _c(94),
                "540M SFT": _c(92), "QWEN INST": _c(94)}


def say(text, end="\n"):
    """Typewriter print in VIDEO mode, plain print otherwise.

    ANSI escape sequences are emitted instantly (not "typed") so colors
    switch cleanly instead of flickering through half-written codes.
    """
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


def load_ours(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = train.GPTDecoder(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"], dropout=0.0,
    ).to(device)
    m.load_state_dict({k.removeprefix("_orig_mod."): v
                       for k, v in ckpt["model_state_dict"].items()})
    m.eval()
    return m


def free(model):
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def gen_ours(model, text, stop_at_eot):
    ids = torch.tensor([enc.encode(text)], device=device)
    out = model.generate(
        ids, max_new_tokens=MAX_NEW, temperature=1.0, top_k=1,
        eos_token_id=enc.eot_token if stop_at_eot else None,
    )
    txt = enc.decode(out[0].tolist())[len(text):]
    return txt.split("<|endoftext|>")[0].strip()


def gen_qwen(model, tok, input_ids):
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][input_ids.shape[1]:],
                      skip_special_tokens=True).strip()


def block(label, text, color="", text_color=""):
    """Multi-line safe: code prompts/answers keep their line breaks."""
    lines = text.split("\n")
    say(f"{color}{label}{RESET}{text_color}{lines[0]}{RESET}")
    pad = " " * len(label)
    for line in lines[1:]:
        say(f"{pad}{text_color}{line}{RESET}")


def show(title, prompts, rows):
    """rows: list of (label, answers) — one row per model."""
    say("")
    say(f"{CYAN}{'=' * 72}{RESET}")
    say(f"{CYAN}{BOLD}  {title}{RESET}")
    say(f"{CYAN}{'=' * 72}{RESET}")
    pause(2)
    for i, p in enumerate(prompts):
        say("")
        block("» ", p, color=f"{BOLD}{YELLOW}",
              text_color=f"{BOLD}{YELLOW}")
        pause(0.8)
        for label, answers in rows:
            col = MODEL_COLORS.get(label.strip(), "")
            block(f"  {label}: ", answers[i], color=f"{BOLD}{col}")
            pause(0.8)
        say(f"{DIM}{'-' * 72}{RESET}")
        pause(2.5)


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

BASE_SECTIONS = [
    ("complete",
     "TEIL 1 — SATZ VERVOLLSTÄNDIGEN, DEUTSCH",
     COMPLETION_PROMPTS),
    ("fakten",
     ("TEIL 2 — FAKTEN (hier arbeiten die 5 Mrd. HQ-Tokens)"
      if len(BASE_MODELS) > 1 else
      "TEIL 2 — FAKTEN (die Grenze von 540M Parametern)"),
     FAKTEN_PROMPTS),
    ("python",
     "TEIL 3 — PYTHON (Heimspiel für Qwen)",
     PYTHON_PROMPTS),
    ("english",
     "TEIL 4 — ENGLISCH (Auswärtsspiel für beide 540M)",
     ENGLISH_PROMPTS),
]
wanted = [s for s in BASE_SECTIONS if SECTION in ("all", s[0])]

if wanted:
    results = {}   # results[model_label][section_key] = [answers]
    for label, path in BASE_MODELS:
        model = load_ours(path)
        results[label] = {key: [gen_ours(model, p, stop_at_eot=False)
                                for p in prompts]
                          for key, _, prompts in wanted}
        free(model)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.float32).to(device).eval()
    results["QWEN 0.5B"] = {}
    for key, _, prompts in wanted:
        results["QWEN 0.5B"][key] = [
            gen_qwen(qwen, tok, tok(p, return_tensors="pt").input_ids.to(device))
            for p in prompts]
    free(qwen)

    labels = [lbl for lbl, _ in BASE_MODELS] + ["QWEN 0.5B"]
    width = max(len(l) for l in labels)
    for key, title, prompts in wanted:
        rows = [(lbl.ljust(width), results[lbl][key]) for lbl in labels]
        show(title, prompts, rows)

if SECTION in ("all", "chat"):
    sft = load_ours(ckpt_path("sft_540m_v2.pt"))
    b_ours = [gen_ours(sft, PROMPT_TMPL.format(frage=q), stop_at_eot=True)
              for q in QUESTIONS]
    free(sft)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).to(device).eval()
    b_qwen = []
    for q in QUESTIONS:
        templated = tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        b_qwen.append(gen_qwen(qwen, tok, templated["input_ids"].to(device)))
    free(qwen)
    show("TEIL 5 — FRAGEN (Chat: 540M-SFT vs Qwen-Instruct — "
         "HQ-SFT folgt im Tool-Calling-Kapitel)",
         QUESTIONS,
         [("540M SFT ", b_ours), ("QWEN INST", b_qwen)])

say("")
say("Alle Modelle: pures greedy decoding — deterministisch,")
say("reproduzierbar mit: python explain/duel_de.py")
