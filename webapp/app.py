"""Public demo: FastAPI service wrapping the self-trained model.

Boot sequence:
  1. Download weights from WEIGHTS_URL (once, ~300 MB / ~1.1 GB), cache them.
  2. Build the model from the config shipped inside the file (size-agnostic —
     124M today, 540M tomorrow: just swap WEIGHTS_URL).
  3. fp16 weights -> fp32 in RAM (CPUs run fp32 noticeably faster and more
     stably than fp16).

A threading.Lock serializes generation: one CPU container can handle exactly
one inference at a time — parallel requests wait instead of stealing each
other's cores.
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
    "https://github.com/getitdone-gmbh/ai-transformer-architecture/releases/download/"
    "weights-124m/weights_124m_fp16.pt",
)
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/tmp/model_weights.pt")
MAX_PROMPT_TOKENS = 256

torch.set_num_threads(os.cpu_count() or 4)

app = FastAPI(title="ai-transformer-architecture demo")
_lock = threading.Lock()


def _load():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Downloading weights: {WEIGHTS_URL}")
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
    print(f"Model ready: {n:,} parameters ({n / 1e6:.0f}M)")
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
        raise HTTPException(400, f"Prompt too long (max {MAX_PROMPT_TOKENS} tokens)")

    input_ids = torch.tensor([ids])
    t0 = time.time()
    # Lock: FastAPI runs sync endpoints in a thread pool — without the lock,
    # parallel requests would compute simultaneously and slow each other down.
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
