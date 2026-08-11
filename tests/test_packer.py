"""Feature 8 tests — packing, masks, position ids, contrastive policy. Offline."""

from __future__ import annotations

import numpy as np

from feature8_packer.packer import (
    PAD_SEGMENT, attention_mask, pack_contrastive_pair, pack_documents,
    packed_batch_report,
)

DOCS = [
    {"doc_id": "a", "tokens": [1, 10, 11, 2]},   # <bos>..<eos>
    {"doc_id": "b", "tokens": [1, 20, 21, 22, 2]},
    {"doc_id": "c", "tokens": [1, 30, 2]},
]


def test_position_ids_reset_per_document() -> None:
    seq = pack_documents(DOCS, seq_len=16, pad_id=0)[0]
    # doc a (len 4) -> 0,1,2,3 ; doc b (len 5) -> 0,1,2,3,4 ; doc c (len 3) -> 0,1,2
    assert seq.position_ids[:12] == [0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2]


def test_segments_and_padding() -> None:
    seq = pack_documents(DOCS, seq_len=16, pad_id=0)[0]
    assert seq.segment_ids[:12] == [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    assert set(seq.segment_ids[12:]) == {PAD_SEGMENT}       # rest is padding
    assert all(seq.loss_mask[i] == 0 for i in range(12, 16))  # no loss on padding


def test_loss_mask_stops_at_document_boundaries() -> None:
    seq = pack_documents(DOCS, seq_len=16, pad_id=0)[0]
    # last token of doc a is index 3; predicting index 4 would cross into doc b -> no loss
    assert seq.loss_mask[3] == 0
    assert seq.loss_mask[0] == 1 and seq.loss_mask[1] == 1  # within doc a


def test_attention_blocks_cross_document() -> None:
    seq = pack_documents(DOCS, seq_len=16, pad_id=0)[0]
    m = attention_mask(seq.segment_ids)
    # token 4 (doc b start) must NOT attend to token 0 (doc a)
    assert not m[4, 0]
    # it may attend to itself and is causal (not to the future)
    assert m[4, 4] and not m[4, 5]
    # padding attends to nothing
    assert not m[13].any()


def test_whole_docs_do_not_split_when_a_boundary_is_hit() -> None:
    # seq_len 8 forces doc c into a second sequence rather than splitting doc b
    seqs = pack_documents(DOCS, seq_len=8, pad_id=0)
    assert len(seqs) == 2
    assert seqs[0].doc_ids == ["a"] or seqs[0].doc_ids == ["a", "b"]
    assert "c" in seqs[-1].doc_ids


def test_contrastive_loss_is_on_the_continuation_only() -> None:
    prefix = [1, 5, 6, 7]        # shared framing
    cont = [8, 9, 2]            # the continuation y
    seq = pack_contrastive_pair(prefix, cont, seq_len=16, pad_id=0)
    lo, hi = seq.cont_span
    # every loss position lies within the continuation-scoring span...
    assert all(seq.loss_mask[i] == 1 for i in range(lo, hi))
    # ...and nothing before the prefix boundary is scored
    assert all(seq.loss_mask[i] == 0 for i in range(0, lo))


def test_report_efficiency_and_determinism() -> None:
    seqs = pack_documents(DOCS, seq_len=16, pad_id=0)
    rep = packed_batch_report(seqs, seq_len=16)
    assert rep["real_tokens"] == 4 + 5 + 3
    assert 0 < rep["packing_efficiency"] <= 1
    assert pack_documents(DOCS, seq_len=16)[0].tokens == pack_documents(DOCS, seq_len=16)[0].tokens
