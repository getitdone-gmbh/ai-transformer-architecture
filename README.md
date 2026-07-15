# ai-transformer-architecture — a GPT trained from scratch

A decoder-only transformer (GPT-style) implemented and trained **from scratch in PyTorch** — no HuggingFace `transformers`, no pretrained weights. The largest run is a **540M-parameter German base model trained on ~10B tokens** (FineWeb2 + Wikipedia + Python code) on a rented H100, followed by supervised fine-tuning into a German question-answering model on an Apple M4 Max.

This is a learning project: the code is deliberately explicit, and the comments explain *why* each decision was made, not just what the code does. Note that the model itself speaks German — it was trained on German data, so all example prompts are (and must be) German.

## The journey

| Stage | Artifact | What happened |
|-------|----------|---------------|
| 1. Notebook | `transformer.ipynb` | First working model (~44M params) on 1000 Wikipedia articles — attention, causal masking, training loop, all hand-built |
| 2. Proper training script | `train.py` + `run_124m.sh` | GPT-2-small-class model (124M) on Vast.ai, ~$2–4 per run |
| 3. Scaling up | `prepare_data.py` + `run_350m.sh` / `run_500m.sh` | Streaming data pipeline (10B tokens as uint16 shards), 540M main run on an H100 (~1.5 days, ~$70) |
| 4. Fine-tuning | `sft.py` + `run_sft.sh` | SFT on German instruction data with loss masking — turns the text continuator into an answer generator |
| 5. Serving | `chat_server.py`, `terminal/`, `webapp/` | Local inference server + Ink terminal UI, plus a public CPU demo (FastAPI) |

## Architecture

Modern GPT recipe, built up piece by piece in `train.py`:

- Decoder-only transformer, **pre-LayerNorm** with **RMSNorm**
- **Rotary position embeddings (RoPE)** with an oversized buffer (4096) for context extrapolation experiments
- **SwiGLU** feed-forward (Llama-style, 3 matrices)
- Tied input/output embeddings
- GPT-2 tokenizer via `tiktoken` (vocab 50,257 — tokens fit in uint16)
- bf16 autocast, `torch.compile`, gradient accumulation, cosine LR schedule with warmup, AdamW with selective weight decay
- Runs on CUDA, Apple Silicon (MPS), and CPU

All hyperparameters are overridable via environment variables, so the same file runs a small local test on a Mac and a multi-day GPU run without edits. The `run_*.sh` scripts are the documented presets:

| Preset | Params | d_model / layers / heads | Context | Tokens |
|--------|--------|--------------------------|---------|--------|
| `run_124m.sh` | 124M | 768 / 12 / 12 | 512 | ~200M (Wikipedia-DE) |
| `run_350m.sh` | 355M | 1024 / 24 / 16 | 1024 | ~10B |
| `run_500m.sh` | 540M | 1280 / 24 / 20 | 2048 | ~10B (FineWeb2-DE + Wikipedia-DE + Python) |

## Pipeline

```
prepare_data.py   stream sources → tokenize → uint16 shards (~20 GB for 10B tokens)
train.py          pretraining; reads shards via np.memmap, never loads all data into RAM
export_weights.py training checkpoint → lean fp16 inference weights (drops Adam state, ~⅓ size)
sft.py            supervised fine-tuning with loss masking (loss only on the answer tokens)
chat_server.py    stdlib-only HTTP inference server, loads the model once
terminal/         Ink (React for the terminal) chat UI talking to chat_server.py
webapp/           public FastAPI demo, CPU inference, downloads weights from a GitHub release
```

For the full step-by-step guide to renting a GPU on Vast.ai and running a training job, see [VASTAI.md](VASTAI.md).

## Running it

```bash
# Install (Python ≥ 3.13, uv recommended)
uv sync

# Small local training run (MPS/CPU-friendly defaults)
python train.py

# Full preset on a CUDA machine — see VASTAI.md
python prepare_data.py        # once, builds shards/
bash run_500m.sh

# Inference against a checkpoint
CHECKPOINT=sft_540m_v2.pt CHAT_TEMPLATE=1 python chat_server.py
cd terminal && npm install && npm start
```

Model weights are not in the repo (`.gitignore` excludes `*.pt`); the web demo pulls fp16 weights from a GitHub release.

## What the model can and cannot do

The 540M base model produces fluent, grammatical German and coherent continuations; after SFT it answers simple questions in a fixed template. It was trained on ~10B tokens — orders of magnitude less than comparable-size open models — so it has thin world knowledge and no real reasoning. That is expected and part of the point: the model doubles as a sandbox for compression experiments (pruning, quantization) where a fully-understood, self-trained baseline is worth more than a stronger black box.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — RMSNorm/SwiGLU/RoPE recipe
- [nanoGPT](https://github.com/karpathy/nanoGPT) — inspiration for the shard-based data pipeline

## License

[MIT](LICENSE)
