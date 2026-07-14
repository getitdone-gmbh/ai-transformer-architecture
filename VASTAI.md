# Vast.ai Runbook — Training the Base Model

Goal: train a base model on German data so we can then use it as a
sandbox for compression (pruning, quantization, …). The walkthrough
below uses the 124M run (GPT-2-small class, `run_124m.sh`) — the larger
presets (`run_350m.sh`, `run_500m.sh`) work identically, but they need
`prepare_data.py` (shard pipeline) beforehand and a stronger GPU;
details are in the comments of each run script.

## 1. Choose an instance

Rent a GPU instance on <https://vast.ai>:

- **GPU:** an RTX 4090 (24 GB) is plenty. Alternatively RTX 3090 / A100 40GB.
- **Disk:** at least **50 GB** — the German Wikipedia downloads as a dump
  (~6–7 GB), plus token cache (~1–2 GB) and checkpoints (~1.4 GB each).
- **Image / Template:** a PyTorch template with CUDA (e.g. "PyTorch (cuDNN)"
  or `pytorch/pytorch`). All we need on top is `tiktoken` + `datasets`.
- **On-Demand** is fine; Interruptible is cheaper, but the run can get
  killed (our checkpoint system covers that — `checkpoint_latest.pt`).

Rough cost: RTX 4090 ~$0.3–0.5/h → a 400k-article run over 1 epoch
takes ~3–6 h → **about $2–4**.

## 2. Connect

Vast.ai shows an SSH line, e.g.:

```bash
ssh -p <PORT> root@<HOST>
```

(Port + host are in the Vast.ai dashboard under "Connect".)

## 3. Repo + setup

On the instance:

```bash
git clone <YOUR_REPO_URL> transformer-test   # or upload via scp
cd transformer-test
bash vastai_setup.sh                          # installs tiktoken, datasets, numpy
```

No git remote? Then upload from your local machine (only the files needed):

```bash
# from your local machine:
scp -P <PORT> train.py run_124m.sh vastai_setup.sh root@<HOST>:~/transformer-test/
```

## 4. Start training

```bash
bash run_124m.sh
```

This runs in the foreground and writes live to `train_124m.log`. For a
run that survives an SSH disconnect:

```bash
nohup bash run_124m.sh > train_124m.log 2>&1 &
tail -f train_124m.log     # watch; Ctrl-C only kills the tail, not the run
```

Expected output at the start: device `cuda`, `Parameters: 123,588,096`,
`Effective batch: 16 x 3 = 48 sequences`. The first batch is slow because
of `torch.compile` (JIT) — after that it picks up speed.

### Knobs (if needed)

- **`CUDA out of memory`:** in `run_124m.sh` set `BATCH_SIZE=8 GRAD_ACCUM_STEPS=6`
  (the effective batch stays at 48).
- **Stronger model / more data:** raise `NUM_ARTICLES` (up to ~2.8M = the entire
  German Wikipedia) and/or `NUM_EPOCHS=2`. Note: 400k articles ≈ 200M tokens —
  by the Chinchilla rule (~2.5 billion tokens optimal) that is still
  *undertrained* for 124M. For a pure compression sandbox it is enough; for a
  genuinely strong model feed it more data (costs linearly more time/money).

## 5. Monitor

```bash
tail -f train_124m.log
nvidia-smi           # GPU utilization / VRAM
```

Watch for: falling `loss`, decreasing `val ppl` at the end of each epoch, and
GPU utilization (`nvidia-smi`) close to 100% (otherwise the DataLoader is the
bottleneck).

## 6. Retrieve the checkpoint

After the run, the instance holds `checkpoint_epoch_1.pt` (end of epoch) and
`checkpoint_latest.pt` (rolling). **The instance is ephemeral — save the
checkpoint before you destroy it:**

```bash
# from your local machine:
scp -P <PORT> root@<HOST>:~/transformer-test/checkpoint_epoch_1.pt ./checkpoint_124m.pt
```

Then **destroy** the Vast.ai instance (don't just stop it), otherwise it
keeps accruing cost.

## 7. Back to local

With `checkpoint_124m.pt` we can then run the actually interesting
experiments on the Mac: activation logging → structured pruning →
quantization, measuring perplexity + speed before/after each step.
