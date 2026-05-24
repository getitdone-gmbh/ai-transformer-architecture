"""Sidecar zum Trainings-Run: extrahiert Embedding-Snapshots.

Pollt checkpoint_latest.pt auf Aenderungen (mtime), und bei jeder neuen
Version wird die Embedding-Matrix fuer eine fest gewaehlte Menge der
TOP_N haeufigsten Tokens extrahiert und nach snapshots/step_NNNN.pt
gespeichert.

Damit umgeht man, dass das eigentliche Checkpoint-File staendig
ueberschrieben wird — wir bewahren pro Step eine schlanke Kopie.

Aufruf parallel zum laufenden Training:
    uv run python snapshot_embeddings.py
"""

import os
import time

import torch


CHECKPOINT_PATH = "checkpoint_latest.pt"
SNAPSHOT_DIR = "snapshots"
TOKEN_CACHE = "data_cache/tokens-de-50000.pt"
TOP_N = 800           # Wieviele haeufigste Tokens wir verfolgen
POLL_INTERVAL = 30    # Sekunden zwischen mtime-Checks


def find_top_tokens(token_cache_path, top_n):
    """Findet die top_n haeufigsten Token-IDs in der getokenisierten Wiki."""
    print(f"Lade Token-Cache fuer Frequenz-Analyse: {token_cache_path}")
    tokens = torch.load(token_cache_path, weights_only=True)
    print(f"  Tokens insgesamt: {len(tokens):,}")
    unique_ids, counts = torch.unique(tokens, return_counts=True)
    top_idx = counts.argsort(descending=True)[:top_n]
    top_ids = unique_ids[top_idx]
    return top_ids.long().sort().values  # sortiert fuer Stabilitaet


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    top_token_path = os.path.join(SNAPSHOT_DIR, "token_ids.pt")
    if os.path.exists(top_token_path):
        top_ids = torch.load(top_token_path, weights_only=True)
        print(f"Token-IDs aus Cache geladen: {len(top_ids)} Tokens")
    else:
        top_ids = find_top_tokens(TOKEN_CACHE, TOP_N)
        torch.save(top_ids, top_token_path)
        print(f"Top-{TOP_N} Token-IDs gespeichert: {top_token_path}")

    print(f"\nWatching {CHECKPOINT_PATH} (poll alle {POLL_INTERVAL}s)...")
    print("Ctrl-C zum Stoppen.\n")

    last_mtime = 0.0
    last_step = -1

    while True:
        try:
            if os.path.exists(CHECKPOINT_PATH):
                mt = os.path.getmtime(CHECKPOINT_PATH)
                if mt > last_mtime:
                    try:
                        ckpt = torch.load(
                            CHECKPOINT_PATH, map_location="cpu", weights_only=False
                        )
                    except Exception as e:
                        # Race-Condition: Datei wird gerade geschrieben
                        print(f"  Retry naechste Runde (load failed: {type(e).__name__})")
                        time.sleep(5)
                        continue

                    step = ckpt.get("global_step", 0)
                    if step != last_step:
                        emb = ckpt["model_state_dict"]["embedding.embedding.weight"]
                        selected = emb[top_ids].clone()  # [TOP_N, d_model]
                        out_path = os.path.join(SNAPSHOT_DIR, f"step_{step:08d}.pt")
                        torch.save({
                            "step": step,
                            "epoch": ckpt.get("epoch", 0),
                            "loss": ckpt.get("loss", float("nan")),
                            "embeddings": selected,
                        }, out_path)
                        size_kb = os.path.getsize(out_path) / 1024
                        print(f"  step={step:>7}  epoch={ckpt.get('epoch', '?')}  "
                              f"loss={ckpt.get('loss', float('nan')):.4f}  "
                              f"-> {out_path} ({size_kb:.0f} KB)")
                        last_step = step
                    last_mtime = mt
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\nGestoppt.")
            break


if __name__ == "__main__":
    main()
