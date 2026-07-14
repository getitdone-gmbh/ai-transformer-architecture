"""Training checkpoint -> lean inference weights (fp16, weights-only).

A training checkpoint carries baggage that inference never needs:
  - Adam states (2 moments per parameter = 2/3 of the file size!)
  - fp32 precision (needed for training stability, not for inference)

This export throws both out: only model_state_dict, cast to fp16.
  124M: ~1.5 GB -> ~250 MB     540M: ~6.5 GB -> ~1.1 GB

The config travels along in the file so the loading code does not have
to guess the architecture (same idea as in chat_server.py).

    python export_weights.py checkpoint_epoch_1.pt weights_124m_fp16.pt
"""

import sys

import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    state = {
        # Normalize the '_orig_mod.' prefix (torch.compile), as everywhere else
        k.removeprefix("_orig_mod."): v.half() if v.is_floating_point() else v
        for k, v in ckpt["model_state_dict"].items()
    }
    out = {
        "config": ckpt["config"],
        "arch_version": ckpt.get("arch_version"),
        "model_state_dict": state,
        "train_loss": ckpt.get("loss"),
        "global_step": ckpt.get("global_step"),
    }
    torch.save(out, dst)

    import os
    src_mb = os.path.getsize(src) / 1024**2
    dst_mb = os.path.getsize(dst) / 1024**2
    print(f"{src} ({src_mb:.0f} MB) -> {dst} ({dst_mb:.0f} MB)")


if __name__ == "__main__":
    main()
