"""Where things live — resolved against the repo, not the working directory.

Every script here can be started from anywhere:

    python explain/duel_de.py          # from the repo root
    cd explain && python duel_de.py    # from its own folder

Both do the same thing, because the paths below are anchored to THIS file's
location instead of os.getcwd(). Scripts outside model/ reach this module
through the two-line preamble at the top of each one, which puts model/ on
sys.path (that same line is what makes `import train` work).

The layout:

    model/        the library — train.py, prepare_data.py, sft.py
    explain/      scripts that measure the finished model
    eval/         bpb + cloze benchmarks
    tools/        weight export, profiling, embedding snapshots
    serve/        chat server, terminal UI, web demo
    runs/         the documented training presets
    checkpoints/  *.pt — gitignored, these are gigabytes
    data/         tokenizer and small frozen inputs, checked in
    eval_data/    frozen bpb eval set — gitignored, third-party text
    shards/       prepare_data.py output — gitignored, ~20 GB
    logs/         training logs — gitignored
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINTS = os.path.join(ROOT, "checkpoints")
DATA = os.path.join(ROOT, "data")
EVAL_DATA = os.path.join(ROOT, "eval_data")
SHARDS = os.path.join(ROOT, "shards")
DATA_CACHE = os.path.join(ROOT, "data_cache")
LOGS = os.path.join(ROOT, "logs")


def ckpt_path(name):
    """A checkpoint file. A bare name resolves into checkpoints/.

    A name containing a path separator is returned unchanged, so
    CHECKPOINT=./experiment.pt and CHECKPOINT=/mnt/big/run.pt still point
    where the caller meant — only the bare filenames in the docstrings
    (CHECKPOINT=sft_540m_v2.pt) get the checkpoints/ prefix.
    """
    return name if os.sep in name else os.path.join(CHECKPOINTS, name)


def data_path(name):
    """A small data file (tokenizer, frozen inputs). Same rule as ckpt_path()."""
    return name if os.sep in name else os.path.join(DATA, name)
