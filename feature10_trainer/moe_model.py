"""Epic 10.1 — a tiny Mixture-of-Experts transformer (PyTorch, CPU, deterministic).

Small by design: token + position embeddings, a couple of blocks each with manual
multi-head causal attention (masked by the packer's block-diagonal same-segment
mask, so attention never crosses a document boundary or attends to padding) and a
top-k MoE feed-forward, then a tied-width head to vocab logits. Everything is
float32 with no dropout, so a fixed seed gives byte-stable results on one machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ModelConfig", "MoETransformer"]


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_experts: int = 4
    top_k: int = 2
    d_ff: int = 256
    seq_len: int = 256


class _Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class _MoE(nn.Module):
    """Top-k gated mixture of small MLP experts."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts)
        self.experts = nn.ModuleList([_Expert(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts)])
        self.top_k = cfg.top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)                                  # [B, S, E]
        weight, idx = torch.topk(gate, self.top_k, dim=-1)   # [B, S, k]
        weight = F.softmax(weight, dim=-1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            sel = idx[..., k]                                # [B, S] chosen expert
            w = weight[..., k].unsqueeze(-1)                 # [B, S, 1]
            for e, expert in enumerate(self.experts):
                mask = sel == e
                if mask.any():
                    out[mask] = out[mask] + w[mask] * expert(x[mask])
        return out


class _Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # [B, h, S, dh]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # [B, h, S, S]
        m = allowed[:, None, :, :]                           # [B, 1, S, S] bool
        att = att.masked_fill(~m, float("-inf"))
        att = torch.nan_to_num(F.softmax(att, dim=-1))       # padding query rows -> 0
        out = (att @ v).transpose(1, 2).reshape(B, S, D)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = _Attention(cfg)
        self.moe = _MoE(cfg)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), allowed)
        x = x + self.moe(self.ln2(x))
        return x


class MoETransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self, tokens: torch.Tensor, position_ids: torch.Tensor, allowed: torch.Tensor
    ) -> torch.Tensor:
        x = self.tok_emb(tokens) + self.pos_emb(position_ids)
        for block in self.blocks:
            x = block(x, allowed)
        return self.head(self.ln_f(x))                       # [B, S, vocab]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
