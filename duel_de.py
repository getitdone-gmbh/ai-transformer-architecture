"""German duel: our 540M vs Qwen2.5-0.5B — side-by-side, reproducible.

Two fair pairings:
  A) Completion (base vs base):   weights_540m_fp32.pt  vs Qwen2.5-0.5B
  B) Questions (chat vs chat):    sft_540m_v2.pt        vs Qwen2.5-0.5B-Instruct
     (each model gets ITS OWN chat template — ours the trained
      "### Frage:/### Antwort:" markers, Qwen its chat_template)

Decoding is greedy (deterministic) with repetition_penalty 1.2 for both —
anyone can rerun this and get the same output. That is the point.

Usage:
    python duel_de.py                # run everything, plain output
    VIDEO=1 python duel_de.py        # typewriter effect + pauses (screen recording)
    SECTION=chat python duel_de.py   # only questions (or: complete)
"""

import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import tiktoken
import torch

import train

VIDEO = os.environ.get("VIDEO", "0") in ("1", "true")
SECTION = os.environ.get("SECTION", "all")
MAX_NEW = int(os.environ.get("MAX_NEW", "60"))
# Pure greedy, NO repetition penalty: it would penalize tokens that appear
# in the prompt — "Wer zuerst kommt," + penalty turns the correct
# continuation "mahlt zuerst" into "mahlt zu." (measured, not theory).
REP_PENALTY = 1.0

COMPLETION_PROMPTS = [
    "Das größte Landtier der Erde ist der",
    "Bienen produzieren",
    "Die drei größten Städte Deutschlands sind",
    "Ein Sprichwort sagt: Wer zuerst kommt,",
    "Sehr geehrte Damen und Herren,",
    "Rezept für Apfelkuchen: Zuerst",
]

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
# our 1B Python tokens are a rounding error against that.
PYTHON_PROMPTS = [
    "def bubble_sort(arr):\n",
    "def is_even(n):\n    ",
    "# Gibt die Summe aller Zahlen in der Liste zurueck\ndef summe(liste):\n",
    'import os\n\nfor filename in os.listdir("."):\n    ',
]

# Our guaranteed away loss: the 540M has seen almost no English.
ENGLISH_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "Water boils at a temperature of",
]

# CRITICAL: baked into the SFT weights — do not translate or alter.
PROMPT_TMPL = "### Frage:\n{frage}\n\n### Antwort:\n"

device = train.get_device()
enc = tiktoken.get_encoding("gpt2")


def say(text, end="\n"):
    """Typewriter print in VIDEO mode, plain print otherwise."""
    if VIDEO:
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(0.012)
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
        repetition_penalty=REP_PENALTY,
        eos_token_id=enc.eot_token if stop_at_eot else None,
    )
    txt = enc.decode(out[0].tolist())[len(text):]
    return txt.split("<|endoftext|>")[0].strip()


def gen_qwen(model, tok, input_ids):
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=MAX_NEW, do_sample=False,
            repetition_penalty=REP_PENALTY, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][input_ids.shape[1]:],
                      skip_special_tokens=True).strip()


def block(label, text):
    """Multi-line safe: code prompts/answers keep their line breaks."""
    lines = text.split("\n")
    say(f"{label}{lines[0]}")
    for line in lines[1:]:
        say(f"             {line}")


def show(title, prompts, ours_answers, qwen_answers):
    say("")
    say("=" * 72)
    say(f"  {title}")
    say("=" * 72)
    pause(2)
    for p, o, q in zip(prompts, ours_answers, qwen_answers):
        say("")
        block("» ", p)
        pause(0.8)
        block("  UNSER 540M: ", o)
        pause(0.8)
        block("  QWEN 0.5B : ", q)
        say("-" * 72)
        pause(2.5)


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# All base-vs-base sections share one model load per side.
BASE_SECTIONS = [
    ("complete",
     "TEIL 1 — SATZ VERVOLLSTÄNDIGEN, DEUTSCH (Basis gegen Basis)",
     COMPLETION_PROMPTS),
    ("python",
     "TEIL 2 — PYTHON (Basis gegen Basis — Heimspiel für Qwen)",
     PYTHON_PROMPTS),
    ("english",
     "TEIL 3 — ENGLISCH (Basis gegen Basis — Auswärtsspiel für uns)",
     ENGLISH_PROMPTS),
]
wanted = [s for s in BASE_SECTIONS if SECTION in ("all", s[0])]

if wanted:
    ours = load_ours("weights_540m_fp32.pt")
    ours_res = {key: [gen_ours(ours, p, stop_at_eot=False) for p in prompts]
                for key, _, prompts in wanted}
    free(ours)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", dtype=torch.float32).to(device).eval()
    qwen_res = {}
    for key, _, prompts in wanted:
        outs = []
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            outs.append(gen_qwen(qwen, tok, ids))
        qwen_res[key] = outs
    free(qwen)

    for key, title, prompts in wanted:
        show(title, prompts, ours_res[key], qwen_res[key])

if SECTION in ("all", "chat"):
    sft = load_ours("sft_540m_v2.pt")
    b_ours = [gen_ours(sft, PROMPT_TMPL.format(frage=q), stop_at_eot=True)
              for q in QUESTIONS]
    free(sft)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).to(device).eval()
    b_qwen = []
    for q in QUESTIONS:
        msgs = [{"role": "user", "content": q}]
        templated = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_dict=True,
            return_tensors="pt")
        b_qwen.append(gen_qwen(qwen, tok, templated["input_ids"].to(device)))
    free(qwen)
    show("TEIL 4 — FRAGEN BEANTWORTEN (Chat-Modell gegen Chat-Modell)",
         QUESTIONS, b_ours, b_qwen)

say("")
say("Beide Seiten: pures greedy decoding — deterministisch,")
say("reproduzierbar mit: python duel_de.py")
