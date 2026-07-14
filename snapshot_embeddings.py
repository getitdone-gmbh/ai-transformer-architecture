"""Sidecar to the training run: extracts embedding snapshots.

Polls checkpoint_latest.pt for changes (mtime), and for every new version
extracts the embedding matrix for a fixed set of the TOP_N most frequent
tokens and saves it to snapshots/step_NNNN.pt.

This works around the fact that the actual checkpoint file is constantly
overwritten — we preserve a lean copy per step.

Run alongside the training in progress:
    uv run python snapshot_embeddings.py
"""

import os
import time

import torch


CHECKPOINT_PATH = "checkpoint_latest.pt"
SNAPSHOT_DIR = "snapshots"
TOKEN_CACHE = "data_cache/tokens-de-50000.pt"
TOP_N = 800           # how many of the most frequent tokens we track
POLL_INTERVAL = 30    # seconds between mtime checks


def find_top_tokens(token_cache_path, top_n):
    """Finds the top_n most frequent token IDs in the tokenized wiki."""
    print(f"Loading token cache for frequency analysis: {token_cache_path}")
    tokens = torch.load(token_cache_path, weights_only=True)
    print(f"  Total tokens: {len(tokens):,}")
    unique_ids, counts = torch.unique(tokens, return_counts=True)
    top_idx = counts.argsort(descending=True)[:top_n]
    top_ids = unique_ids[top_idx]
    return top_ids.long().sort().values  # sorted for stability


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    top_token_path = os.path.join(SNAPSHOT_DIR, "token_ids.pt")
    if os.path.exists(top_token_path):
        top_ids = torch.load(top_token_path, weights_only=True)
        print(f"Token IDs loaded from cache: {len(top_ids)} tokens")
    else:
        top_ids = find_top_tokens(TOKEN_CACHE, TOP_N)
        torch.save(top_ids, top_token_path)
        print(f"Top-{TOP_N} token IDs saved: {top_token_path}")

    print(f"\nWatching {CHECKPOINT_PATH} (polling every {POLL_INTERVAL}s)...")
    print("Ctrl-C to stop.\n")

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
                        # Race condition: the file is being written right now
                        print(f"  Retrying next round (load failed: {type(e).__name__})")
                        time.sleep(5)
                        continue

                    step = ckpt.get("global_step", 0)
                    if step != last_step:
                        sd = ckpt["model_state_dict"]
                        # torch.compile prefixes all keys with '_orig_mod.'.
                        # Support both variants.
                        key = "embedding.embedding.weight"
                        if key not in sd:
                            key = "_orig_mod.embedding.embedding.weight"
                        emb = sd[key]
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
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
