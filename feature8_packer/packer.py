"""Feature 8 — the packer (epics 8.1-8.6).

Packs documents into fixed-length sequences and produces everything a trainer
needs to treat packed documents as if they were trained separately:

  * sequence packing to seq_len with doc boundaries (8.1): whole docs are packed
    until seq_len; the remainder is padded. A doc longer than seq_len is
    truncated (rare here -- cleaned docs are short).
  * loss mask (8.2): a position contributes to loss only if its NEXT token is a
    real token in the SAME document (padding and cross-doc/last positions = 0).
  * attention mask (8.3): causal AND same-segment, so a token can never attend
    across a document boundary.
  * position ids (8.4): reset to 0 at the start of every document.
  * contrastive-pair packing (8.5): prefix + continuation in one sequence, with
    the loss mask on the CONTINUATION span only, so F1/F2 surprisal (ΔS) is
    measured on y, not the shared prefix.
  * packed-batch report (8.6): sequences, real vs padding tokens, packing
    efficiency, loss positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

__all__ = [
    "PAD_SEGMENT", "PackedSequence", "pack_documents", "attention_mask",
    "pack_contrastive_pair", "packed_batch_report",
]

PAD_SEGMENT = -1  # segment id for padding positions


@dataclass(frozen=True)
class PackedSequence:
    tokens: list[int]
    position_ids: list[int]
    segment_ids: list[int]   # doc index within the sequence; PAD_SEGMENT for padding
    loss_mask: list[int]     # 1 if this position's next-token target is real & same-doc
    doc_ids: list[str]
    lane: str | None = None  # lane of the sequence (its first document)
    cont_span: tuple[int, int] | None = None  # (start, end) for contrastive sequences

    def n_real_tokens(self) -> int:
        return sum(1 for s in self.segment_ids if s != PAD_SEGMENT)

    def n_loss_positions(self) -> int:
        return sum(self.loss_mask)


def _loss_mask(segment_ids: list[int]) -> list[int]:
    """1 where the next token is real and belongs to the same document."""
    n = len(segment_ids)
    mask = [0] * n
    for i in range(n - 1):
        if segment_ids[i] != PAD_SEGMENT and segment_ids[i + 1] == segment_ids[i]:
            mask[i] = 1
    return mask


def pack_documents(
    docs: Iterable[dict[str, Any]],
    *,
    seq_len: int,
    pad_id: int = 0,
) -> list[PackedSequence]:
    """Pack {doc_id, tokens} records into padded, doc-boundaried sequences."""
    seqs: list[PackedSequence] = []
    tok: list[int] = []
    pos: list[int] = []
    seg: list[int] = []
    ids: list[str] = []
    seg_no = 0
    lane: str | None = None

    def flush() -> None:
        nonlocal tok, pos, seg, ids, seg_no, lane
        if not tok:
            return
        pad = seq_len - len(tok)
        full_tok = tok + [pad_id] * pad
        full_pos = pos + [0] * pad
        full_seg = seg + [PAD_SEGMENT] * pad
        seqs.append(PackedSequence(
            full_tok, full_pos, full_seg, _loss_mask(full_seg), list(ids), lane=lane))
        tok, pos, seg, ids, seg_no, lane = [], [], [], [], 0, None

    for doc in docs:
        t = list(doc["tokens"])[:seq_len]
        if not t:
            continue
        if tok and len(tok) + len(t) > seq_len:
            flush()
        if lane is None:
            lane = doc.get("lane")           # lane of the sequence's first doc
        tok.extend(t)
        pos.extend(range(len(t)))            # position ids reset per doc
        seg.extend([seg_no] * len(t))
        ids.append(doc["doc_id"])
        seg_no += 1
    flush()
    return seqs


def attention_mask(segment_ids: list[int]) -> np.ndarray:
    """Boolean [L, L]: position i may attend to j iff j<=i, same segment, not pad."""
    seg = np.array(segment_ids)
    n = seg.size
    causal = np.tril(np.ones((n, n), dtype=bool))
    same = seg[:, None] == seg[None, :]
    real = seg[:, None] != PAD_SEGMENT
    return causal & same & real


def pack_contrastive_pair(
    prefix_tokens: list[int],
    cont_tokens: list[int],
    *,
    seq_len: int,
    pad_id: int = 0,
) -> PackedSequence:
    """Pack prefix+continuation; loss mask on the CONTINUATION span only."""
    prefix = list(prefix_tokens)
    cont = list(cont_tokens)
    body = (prefix + cont)[:seq_len]
    p_len = min(len(prefix), seq_len)
    pad = seq_len - len(body)
    tokens = body + [pad_id] * pad
    position_ids = list(range(len(body))) + [0] * pad
    segment_ids = [0] * len(body) + [PAD_SEGMENT] * pad

    # score the continuation: positions predicting a continuation token
    cont_start = max(p_len - 1, 0)
    cont_end = len(body) - 1  # last position has no successor to predict
    loss_mask = [0] * seq_len
    for i in range(cont_start, cont_end):
        loss_mask[i] = 1
    return PackedSequence(
        tokens, position_ids, segment_ids, loss_mask, doc_ids=["<contrastive>"],
        cont_span=(cont_start, cont_end),
    )


def packed_batch_report(sequences: list[PackedSequence], *, seq_len: int) -> dict[str, Any]:
    """Summarize packing: efficiency (real tokens / capacity) and loss coverage."""
    n = len(sequences)
    capacity = n * seq_len
    real = sum(s.n_real_tokens() for s in sequences)
    loss = sum(s.n_loss_positions() for s in sequences)
    return {
        "kind": "packing_report",
        "seq_len": seq_len,
        "n_sequences": n,
        "capacity_tokens": capacity,
        "real_tokens": real,
        "padding_tokens": capacity - real,
        "loss_positions": loss,
        "packing_efficiency": (real / capacity) if capacity else 0.0,
    }
