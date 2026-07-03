"""Trainings-Checkpoint -> schlanke Inferenz-Gewichte (fp16, weights-only).

Ein Trainings-Checkpoint traegt Ballast, den Inferenz nie braucht:
  - Adam-Zustaende (2 Momente pro Parameter = 2/3 der Dateigroesse!)
  - fp32-Praezision (fuer Trainings-Stabilitaet noetig, fuer Inferenz nicht)

Dieser Export wirft beides raus: nur model_state_dict, gecastet auf fp16.
  124M: ~1,5 GB -> ~250 MB     540M: ~6,5 GB -> ~1,1 GB

Die config wandert mit in die Datei, damit der Lade-Code die Architektur
nicht raten muss (gleiche Idee wie in chat_server.py).

    python export_weights.py checkpoint_epoch_1.pt weights_124m_fp16.pt
"""

import sys

import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    state = {
        # '_orig_mod.'-Praefix (torch.compile) normalisieren wie ueberall
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
