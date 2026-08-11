"""Epics 10.2-10.5 — deterministic training step, learning ledger, ΔS.

The trainer consumes the reproducible batch stream, one deterministic step per
batch. Per-token cross-entropy IS the F1 surprisal (nats); its mean per batch,
tied to the batch id (and thus to the source docs), goes into an append-only
learning ledger (10.3, 10.5). After training, F2 ΔS = S(y-) - S(y+) is measured
per contrastive pair on the continuation span only (10.4).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from feature8_packer.packer import attention_mask, pack_contrastive_pair
from feature10_trainer.moe_model import ModelConfig, MoETransformer

__all__ = ["Trainer", "LearningLedger", "batch_tensors", "compute_loss", "contrastive_delta_s"]


def _seed_int(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:4], "big")


def batch_tensors(stream: Any, batch: Any):
    """Build (tokens, position_ids, allowed_mask, loss_mask) tensors for a batch."""
    seqs = [stream.sequences[j] for j in batch.seq_indices]
    tokens = torch.tensor([s.tokens for s in seqs], dtype=torch.long)
    pos = torch.tensor([s.position_ids for s in seqs], dtype=torch.long)
    loss_mask = torch.tensor([s.loss_mask for s in seqs], dtype=torch.long)
    allowed = torch.tensor(np.stack([attention_mask(s.segment_ids) for s in seqs]))
    return tokens, pos, allowed, loss_mask


def compute_loss(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor):
    """Masked next-token cross-entropy. Returns (mean_loss, per_position_surprisal)."""
    B, S, V = logits.shape
    logit = logits[:, :-1].reshape(-1, V)
    target = tokens[:, 1:].reshape(-1)
    m = loss_mask[:, :-1].reshape(-1).to(logits.dtype)
    ce = F.cross_entropy(logit, target, reduction="none")   # surprisal (nats) per position
    n = m.sum().clamp(min=1.0)
    return (ce * m).sum() / n, ce


class Trainer:
    """A tiny deterministic MoE trainer."""

    def __init__(self, cfg: ModelConfig, *, seed: str = "v5-trainer-2026", lr: float = 3e-4) -> None:
        torch.manual_seed(_seed_int(seed))
        torch.set_num_threads(1)          # deterministic CPU reductions -> stable losses
        self.model = MoETransformer(cfg)
        self.model.train()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, tokens, position_ids, allowed, loss_mask) -> float:
        logits = self.model(tokens, position_ids, allowed)
        loss, _ = compute_loss(logits, tokens, loss_mask)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())


class LearningLedger:
    """Append-only per-step record linking loss (F1 surprisal) to its batch/source."""

    def __init__(self, path: str, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self._fh = self.path.open("a" if append else "w", encoding="utf-8", newline="\n")

    def record(self, *, step: int, batch: Any, loss: float, n_loss_tokens: int) -> None:
        row = {
            "step": step,
            "batch_index": batch.index,
            "batch_id": batch.batch_id,
            "content_hash": batch.content_hash,
            "loss_nats": round(loss, 6),
            "mean_surprisal": round(loss, 6),
            "n_loss_tokens": n_loss_tokens,
        }
        self.rows.append(row)
        self._fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _continuation_surprisal(model, tokenizer, prefix: str, cont: str, seq_len: int) -> float:
    """Summed surprisal (nats) of `cont` given `prefix`, on the continuation span."""
    bos, eos = tokenizer.vocab["<bos>"], tokenizer.vocab["<eos>"]
    prefix_ids = [bos] + tokenizer.encode(prefix)
    cont_ids = tokenizer.encode(cont) + [eos]
    seq = pack_contrastive_pair(prefix_ids, cont_ids, seq_len=seq_len, pad_id=tokenizer.vocab["<pad>"])
    tokens = torch.tensor([seq.tokens], dtype=torch.long)
    pos = torch.tensor([seq.position_ids], dtype=torch.long)
    allowed = torch.tensor(attention_mask(seq.segment_ids))[None, :, :]
    loss_mask = torch.tensor([seq.loss_mask], dtype=torch.long)
    with torch.no_grad():
        logits = model(tokens, pos, allowed)
        B, S, V = logits.shape
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1), reduction="none")
        m = loss_mask[:, :-1].reshape(-1).to(logits.dtype)
        return float((ce * m).sum())


def contrastive_delta_s(model, tokenizer, pairs: Sequence[Any], *, seq_len: int) -> list[dict[str, Any]]:
    """F2 signal: ΔS = S(y-) - S(y+) per contrastive pair (continuation surprisal)."""
    model.eval()
    rows: list[dict[str, Any]] = []
    for p in pairs:
        s_plus = _continuation_surprisal(model, tokenizer, p.prefix, p.y_plus, seq_len)
        s_minus = _continuation_surprisal(model, tokenizer, p.prefix, p.y_minus, seq_len)
        rows.append({
            "pair_id": p.id, "topic": p.topic,
            "S_plus": round(s_plus, 4), "S_minus": round(s_minus, 4),
            "delta_s": round(s_minus - s_plus, 4),
        })
    return rows
