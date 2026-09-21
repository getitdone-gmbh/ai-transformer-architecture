"""Lean inference-only copy of the model architecture from train.py.

Deliberately duplicated instead of imported: train.py drags in datasets/numpy
and training code — the demo container only needs torch + tiktoken.
The classes are 1:1 compatible with the state_dict from export_weights.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, seq_len):
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rotary(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.d_k, max_seq_len=max_seq_len)

    def forward(self, x):
        B, T, _ = x.size()

        def split(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        cos, sin = self.rope(T)
        Q, K = apply_rotary(Q, cos, sin), apply_rotary(K, cos, sin)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(ms + self.eps))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len=4096):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, max_seq_len=max_seq_len)
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_len=4096):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        for block in self.decoder_blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0,
                 top_k=None, top_p=None, repetition_penalty=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)[:, -1, :]

            if repetition_penalty != 1.0:
                for b in range(input_ids.size(0)):
                    seen = torch.unique(input_ids[b])
                    score = logits[b, seen]
                    logits[b, seen] = torch.where(
                        score < 0, score * repetition_penalty,
                        score / repetition_penalty)

            logits = logits / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                logits = torch.where(logits < topk_vals[..., -1, None],
                                     torch.full_like(logits, float("-inf")), logits)

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_sorted = cumulative > top_p
                remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
                remove_sorted[..., 0] = False
                remove = torch.zeros_like(logits, dtype=torch.bool)
                remove.scatter_(-1, sorted_idx, remove_sorted)
                logits = logits.masked_fill(remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
