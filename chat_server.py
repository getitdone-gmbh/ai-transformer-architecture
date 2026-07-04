"""Mini-Inferenz-Server fuer das Chat-Terminal (terminal/).

Warum ein Server statt Modell-Laden pro Anfrage: der Checkpoint ist ~1,5 GB
und das Laden dauert Sekunden. Der Server laedt EINMAL und haelt das Modell
im Speicher — die Terminal-UI kann beliebig oft neu starten und fragt nur
per HTTP an. Bewusst nur Python-Stdlib (http.server), keine neuen Deps.

Start:
    python chat_server.py                          # nimmt checkpoint_epoch_1.pt
    CHECKPOINT=andere_datei.pt python chat_server.py

Endpunkte:
    GET  /info      -> Modell-Metadaten (Parameter, Device, Config)
    POST /generate  -> {"prompt": "...", "max_new_tokens": 80, ...}
                    -> {"text": "...", "seconds": 1.2, "tokens_per_s": 55.0}

Die Modell-Dimensionen kommen aus dem Checkpoint selbst (config-Dict) —
derselbe Server funktioniert also unveraendert fuer das 124M- und spaeter
das 540M-Modell.
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import tiktoken
import torch

import train

CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoint_epoch_1.pt")
PORT = int(os.environ.get("PORT", "8123"))

# Sampling-Defaults — die UI kann jeden Wert pro Anfrage ueberschreiben.
DEFAULTS = dict(max_new_tokens=80, temperature=0.8, top_p=0.9,
                repetition_penalty=1.2)

# Chat-Modus (CHAT_TEMPLATE=1): fuer SFT-Checkpoints. Wickelt die Eingabe
# in das Trainings-Template und stoppt am <|endoftext|>. Beim Basismodell
# aus lassen — es kennt weder das Template noch das Aufhoeren.
CHAT_TEMPLATE = os.environ.get("CHAT_TEMPLATE", "0").lower() in ("1", "true")
PROMPT_TMPL = "### Frage:\n{frage}\n\n### Antwort:\n"


def load_model():
    device = train.get_device()
    print(f"Lade {CHECKPOINT} auf {device}...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = train.GPTDecoder(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"], dropout=0.0,  # Inferenz: kein Dropout
    ).to(device)
    sd = {k.removeprefix("_orig_mod."): v
          for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Bereit: {n_params:,} Parameter, val-Loss beim Training: "
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
        pass  # kein Request-Log-Spam im Terminal

    def do_GET(self):
        if self.path != "/info":
            return self._send(404, {"error": "unbekannter Pfad"})
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
            return self._send(404, {"error": "unbekannter Pfad"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            prompt = req["prompt"]
            opts = {**DEFAULTS, **{k: req[k] for k in DEFAULTS if k in req}}
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": f"kaputte Anfrage: {e}"})

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
            # Nur die Antwort zurueckgeben, ohne Template und EOT.
            text = (text[len(model_input):]
                    .replace("<|endoftext|>", "").strip())
        self._send(200, {
            "text": text,
            "seconds": round(seconds, 2),
            "tokens_per_s": round(new_tokens / max(seconds, 1e-9), 1),
        })


if __name__ == "__main__":
    print(f"Chat-Server auf http://127.0.0.1:{PORT}  (Ctrl-C beendet)")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
