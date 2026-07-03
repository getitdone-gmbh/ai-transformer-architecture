"""Öffentliche Demo: FastAPI-Service um das selbst trainierte Modell.

Boot-Ablauf:
  1. Gewichte von WEIGHTS_URL laden (einmalig, ~300 MB / ~1,1 GB), cachen.
  2. Modell aus der im File mitgereisten config bauen (größenagnostisch —
     124M heute, 540M morgen: nur WEIGHTS_URL wechseln).
  3. fp16-Gewichte -> fp32 im RAM (CPU rechnet fp32 deutlich schneller
     und stabiler als fp16).

Ein threading.Lock serialisiert die Generierung: ein CPU-Container schafft
genau eine Inferenz auf einmal — parallele Anfragen warten, statt sich
gegenseitig die Kerne zu klauen.
"""

import os
import threading
import time
import urllib.request

import tiktoken
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model import GPTDecoder

WEIGHTS_URL = os.environ.get(
    "WEIGHTS_URL",
    "https://github.com/b3rtram/transformer-test/releases/download/"
    "weights-124m/weights_124m_fp16.pt",
)
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/tmp/model_weights.pt")
MAX_PROMPT_TOKENS = 256

torch.set_num_threads(os.cpu_count() or 4)

app = FastAPI(title="transformer-test demo")
_lock = threading.Lock()


def _load():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Lade Gewichte: {WEIGHTS_URL}")
        tmp = WEIGHTS_PATH + ".part"
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        os.replace(tmp, WEIGHTS_PATH)
    blob = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = GPTDecoder(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"],
    )
    state = {k: v.float() if v.is_floating_point() else v
             for k, v in blob["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Modell bereit: {n:,} Parameter ({n / 1e6:.0f}M)")
    return model, cfg, n


ENC = tiktoken.get_encoding("gpt2")
MODEL, CFG, N_PARAMS = _load()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=48, ge=1, le=80)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/info")
def info():
    return {
        "params": N_PARAMS,
        "params_mio": round(N_PARAMS / 1e6),
        "d_model": CFG["d_model"],
        "num_layers": CFG["num_layers"],
        "num_heads": CFG["num_heads"],
        "seq_length": CFG.get("seq_length"),
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
def generate(req: GenerateRequest):
    ids = ENC.encode(req.prompt)
    if len(ids) > MAX_PROMPT_TOKENS:
        raise HTTPException(400, f"Prompt zu lang (max {MAX_PROMPT_TOKENS} Tokens)")

    input_ids = torch.tensor([ids])
    t0 = time.time()
    # Lock: FastAPI führt sync-Endpoints im Threadpool aus — ohne Lock
    # würden parallele Anfragen gleichzeitig rechnen und sich ausbremsen.
    with _lock:
        out = MODEL.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=0.9,
            repetition_penalty=1.2,
        )
    seconds = time.time() - t0
    full = ENC.decode(out[0].tolist())
    return {
        "prompt": req.prompt,
        "continuation": full[len(req.prompt):],
        "seconds": round(seconds, 2),
        "tokens_per_s": round((out.size(1) - len(ids)) / max(seconds, 1e-9), 1),
    }
