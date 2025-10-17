# Transformer Language Model - PyTorch Implementation

A complete GPT-style Transformer model implemented in PyTorch with Apple Silicon (MPS) support.

## 🎯 Overview

This project implements a Decoder-Only Transformer (similar to GPT) from scratch in PyTorch. The model is trained on German Wikipedia articles and can generate coherent text.

### Features

- ✅ **Multi-Head Attention** with Scaled Dot-Product Attention
- ✅ **Positional Encoding** for sequence information
- ✅ **Causal Masking** for autoregressive text generation
- ✅ **Layer Normalization** and Residual Connections
- ✅ **Apple Silicon GPU Support** (MPS Backend)
- ✅ **Checkpoint System** for saving/loading models
- ✅ **Text Generation** with Temperature Sampling

## 🏗️ Model Architecture

```
GPT Decoder (Decoder-only Transformer)
├── Token Embedding (Vocab: 50257)
├── Positional Encoding
├── 6x Decoder Blocks
│   ├── Masked Multi-Head Attention (8 Heads)
│   ├── Layer Normalization
│   ├── Feed-Forward Network (512 → 2048 → 512)
│   └── Layer Normalization
└── Language Model Head (512 → 50257)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| num_heads | 8 |
| d_ff | 2048 |
| num_layers | 6 |
| vocab_size | 50257 (GPT-2 Tokenizer) |
| seq_length | 128 |
| batch_size | 8 |
| learning_rate | 3e-4 |

**Total Parameters:** ~44 Million

## 📋 Requirements

- Python >= 3.13
- Apple Silicon Mac (for MPS) or CPU

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd transfomer-test
```

2. **Install dependencies:**
```bash
pip install -e .
```

Or manually:
```bash
pip install torch numpy tiktoken datasets wikipedia
```

## 💻 Usage

### Training

Training is done in the Jupyter Notebook `transformer.ipynb`:

```bash
jupyter notebook transformer.ipynb
```

**Training Pipeline:**

1. **Load Data:** German Wikipedia articles via Hugging Face Datasets
2. **Tokenization:** Using GPT-2 Tokenizer (tiktoken)
3. **Model Training:** Cross-Entropy Loss with Adam Optimizer
4. **Checkpoints:** Automatic saving every 2 epochs

### Text Generation

```python
# Load checkpoint
checkpoint = torch.load('checkpoint_epoch_3.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate text
start_text = "Die Geschichte"
start_tokens = encoding.encode(start_text)
input_ids = torch.tensor([start_tokens], device=device)

generated_ids = model.generate(
    input_ids, 
    max_new_tokens=50, 
    temperature=0.8
)

generated_text = encoding.decode(generated_ids[0].cpu().tolist())
print(generated_text)
```

### Standalone Script

```bash
python main.py
```

## 📁 Project Structure

```
transfomer-test/
├── transformer.ipynb      # Main notebook with training & generation
├── main.py               # Standalone Python script
├── pyproject.toml        # Project configuration
├── checkpoint_epoch_3.pt # Saved model
└── README.md            # This file
```

## 🧠 Implementation Details

### Attention Mechanism

```python
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
```

- **Q (Query):** What are we looking for?
- **K (Key):** What do we have?
- **V (Value):** What do we return?

### Causal Masking

The model uses a lower-triangular mask so tokens can only attend to previous positions:

```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

### Positional Encoding

Sinusoidal functions for position information:

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## 📊 Training

**Dataset:** 1000 German Wikipedia articles (~1M Tokens)

**Training Progress:**
- Epoch 1: Loss ~8.5
- Epoch 5: Loss ~5.2
- Epoch 10: Loss ~3.8

**Hardware:** Apple M-Series with MPS Backend

## 🎨 Text Generation Examples

```
Prompt: "Die Geschichte"
Output: "Die Geschichte der deutschen Literatur beginnt im Mittelalter..."

Prompt: "Im Jahr"
Output: "Im Jahr 1945 endete der Zweite Weltkrieg in Europa..."

Prompt: "Deutschland ist"
Output: "Deutschland ist ein föderaler Staat in Mitteleuropa..."
```

## 🔧 Configuration

Adjust hyperparameters in `transformer.ipynb`:

```python
SEQ_LENGTH = 128      # Sequence length
BATCH_SIZE = 8        # Batch size
NUM_EPOCHS = 10       # Number of epochs
NUM_LAYERS = 6        # Transformer layers
NUM_HEADS = 8         # Attention heads
D_MODEL = 512         # Embedding dimension
D_FF = 2048          # Feed-forward dimension
```

## 📚 References

- [Attention is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Original Transformer Paper
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 Paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

## 🛠️ Technology Stack

- **PyTorch** - Deep Learning Framework
- **tiktoken** - GPT-2/3/4 Tokenizer
- **Hugging Face Datasets** - Wikipedia Data
- **NumPy** - Numerical Operations

## 📝 TODO

- [ ] Gradient Accumulation for larger batch sizes
- [ ] Learning Rate Scheduling
- [ ] Beam Search for better generation
- [ ] Top-k/Top-p Sampling
- [ ] Model Evaluation (Perplexity)
- [ ] Multi-GPU Training
- [ ] Mixed Precision Training (FP16)

## 🐛 Known Issues

- Training on CPU is very slow (use MPS or CUDA)
- Large sequences (>256) require significant VRAM
- Text generation can become repetitive (adjust temperature)

## 📄 License

MIT License - Free to use for learning and research purposes

## 👤 Author

Created as a learning project to implement Transformer architectures

---

⭐ **Star this project** if it helped you understand Transformers!
