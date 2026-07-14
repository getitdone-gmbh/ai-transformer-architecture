"""Mini inference server for the chat terminal (terminal/).

Why a server instead of loading the model per request: the checkpoint is
~1.5 GB and loading takes seconds. The server loads ONCE and keeps the model
in memory — the terminal UI can restart as often as it likes and only talks
to it over HTTP. Deliberately Python stdlib only (http.server), no new deps.

Start:
    CHAT_TEMPLATE=1 python chat_server.py          # uses sft_540m_v2.pt
    CHECKPOINT=other_file.pt python chat_server.py

Endpoints:
    GET  /info      -> model metadata (parameters, device, config)
    POST /generate  -> {"prompt": "...", "max_new_tokens": 80, ...}
                    -> {"text": "...", "seconds": 1.2, "tokens_per_s": 55.0}

The model dimensions come from the checkpoint itself (config dict) —
so the same server works unchanged for the 124M model and later for
the 540M one.
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import tiktoken
import torch

import train

CHECKPOINT = os.environ.get("CHECKPOINT", "sft_540m_v2.pt")
PORT = int(os.environ.get("PORT", "8123"))

# Sampling defaults — the UI can override every value per request.
DEFAULTS = dict(max_new_tokens=80, temperature=0.8, top_p=0.9,
                repetition_penalty=1.2)

# Chat mode (CHAT_TEMPLATE=1): for SFT checkpoints. Wraps the input in the
# training template and stops at <|endoftext|>. Leave it off for the base
# model — it knows neither the template nor when to stop.
CHAT_TEMPLATE = os.environ.get("CHAT_TEMPLATE", "0").lower() in ("1", "true")
# CRITICAL: this template ("### Frage" / "### Antwort", German for
# question/answer) is baked into the trained SFT model weights — the model
# was fine-tuned on exactly these marker strings. Do NOT translate or alter
# it, or the model will stop following the chat format.
PROMPT_TMPL = "### Frage:\n{frage}\n\n### Antwort:\n"


def load_model():
    device = train.get_device()
    print(f"Loading {CHECKPOINT} on {device}...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = train.GPTDecoder(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"], dropout=0.0,  # inference: no dropout
    ).to(device)
    sd = {k.removeprefix("_orig_mod."): v
          for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Ready: {n_params:,} parameters, val loss at training time: "
          f"{ckpt.get('loss', float('nan')):.3f}")
    return model, cfg, device, n_params


ENC = tiktoken.get_encoding("gpt2")
MODEL, CFG, DEVICE, N_PARAMS = load_model()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # no request-log spam in the terminal

    def do_GET(self):
        if self.path != "/info":
            return self._send(404, {"error": "unknown path"})
        self._send(200, {
            "checkpoint": CHECKPOINT,
            "params": N_PARAMS,
            "device": str(DEVICE),
            "d_model": CFG["d_model"],
            "num_layers": CFG["num_layers"],
            "defaults": DEFAULTS,
        })

    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, {"error": "unknown path"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            prompt = req["prompt"]
            opts = {**DEFAULTS, **{k: req[k] for k in DEFAULTS if k in req}}
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": f"malformed request: {e}"})

        model_input = (PROMPT_TMPL.format(frage=prompt)
                       if CHAT_TEMPLATE else prompt)
        ids = torch.tensor([ENC.encode(model_input)], device=DEVICE)
        t0 = time.time()
        out = MODEL.generate(
            ids,
            max_new_tokens=opts["max_new_tokens"],
            temperature=opts["temperature"],
            top_p=opts["top_p"],
            repetition_penalty=opts["repetition_penalty"],
            eos_token_id=ENC.eot_token if CHAT_TEMPLATE else None,
        )
        seconds = time.time() - t0
        new_tokens = out.size(1) - ids.size(1)
        text = ENC.decode(out[0].cpu().tolist())
        if CHAT_TEMPLATE:
            # Return only the answer, without the template and the EOT token.
            text = (text[len(model_input):]
                    .replace("<|endoftext|>", "").strip())
        self._send(200, {
            "text": text,
            "seconds": round(seconds, 2),
            "tokens_per_s": round(new_tokens / max(seconds, 1e-9), 1),
        })


if __name__ == "__main__":
    print(f"Chat server at http://127.0.0.1:{PORT}  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
