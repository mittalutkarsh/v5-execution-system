"""Feature 10 tests — MoE model, deterministic step, learning ledger, ΔS. Offline."""

from __future__ import annotations

import torch

from feature8_packer.packer import pack_documents
from feature9_batches.batch_stream import BatchStream
from feature10_trainer.moe_model import ModelConfig, MoETransformer
from feature10_trainer.trainer import (
    LearningLedger, Trainer, batch_tensors, compute_loss, contrastive_delta_s,
)

VOCAB, SEQ = 64, 16


def _stream():
    docs = [{"doc_id": f"d{i}", "lane": "web",
             "tokens": [1] + [(i * 7 + j * 3) % VOCAB for j in range(SEQ - 2)] + [2]}
            for i in range(16)]
    seqs = pack_documents(docs, seq_len=SEQ, pad_id=0)
    return BatchStream(seqs, seed="t", batch_size=4)


def _cfg():
    return ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                       n_experts=4, top_k=2, d_ff=64, seq_len=SEQ)


def test_model_forward_shape_and_size() -> None:
    m = MoETransformer(_cfg())
    stream = _stream()
    tokens, pos, allowed, _ = batch_tensors(stream, stream.batch(0))
    logits = m(tokens, pos, allowed)
    assert logits.shape == (4, SEQ, VOCAB)
    assert m.n_params() > 0


def test_training_is_deterministic() -> None:
    stream = _stream()
    losses = []
    for _ in range(2):
        tr = Trainer(_cfg(), seed="fixed", lr=1e-3)
        run = [tr.train_step(*batch_tensors(stream, stream.batch(i))) for i in range(5)]
        losses.append(run)
    assert losses[0] == losses[1]                      # byte-identical loss trajectory


def test_loss_decreases_on_a_repeated_batch() -> None:
    stream = _stream()
    tr = Trainer(_cfg(), seed="fixed", lr=5e-3)
    tokens, pos, allowed, lm = batch_tensors(stream, stream.batch(0))
    first = tr.train_step(tokens, pos, allowed, lm)
    for _ in range(30):
        tr.train_step(tokens, pos, allowed, lm)
    last = tr.train_step(tokens, pos, allowed, lm)
    assert last < first                                # the model is learning


def test_loss_ignores_padding_positions() -> None:
    cfg = _cfg()
    m = MoETransformer(cfg)
    # a batch with heavy padding still yields a finite loss (padding masked out)
    docs = [{"doc_id": "a", "lane": "web", "tokens": [1, 5, 6, 2]}]
    from feature8_packer.packer import pack_documents as pk
    seqs = pk(docs, seq_len=SEQ, pad_id=0)
    stream = BatchStream(seqs, seed="t", batch_size=1)
    tokens, pos, allowed, lm = batch_tensors(stream, stream.batch(0))
    loss, _ = compute_loss(m(tokens, pos, allowed), tokens, lm)
    assert torch.isfinite(loss)


def test_learning_ledger_links_loss_to_batch(tmp_path) -> None:
    stream = _stream()
    tr = Trainer(_cfg(), seed="fixed", lr=1e-3)
    ledger = LearningLedger(str(tmp_path / "learning.jsonl"))
    for i in range(4):
        batch = stream.batch(i)
        tokens, pos, allowed, lm = batch_tensors(stream, batch)
        loss = tr.train_step(tokens, pos, allowed, lm)
        ledger.record(step=i, batch=batch, loss=loss, n_loss_tokens=int(lm.sum()))
    ledger.close()
    assert [r["batch_id"] for r in ledger.rows] == [stream.batch(i).batch_id for i in range(4)]
    assert all("mean_surprisal" in r for r in ledger.rows)


class _Pair:
    def __init__(self, i, topic, prefix, yp, ym):
        self.id, self.topic, self.prefix, self.y_plus, self.y_minus = i, topic, prefix, yp, ym


def test_delta_s_is_computed_per_pair() -> None:
    from feature3_tokenizer.bpe_train import train_bpe
    from feature3_tokenizer.bpe_tokenizer import Tokenizer
    vocab, merges = train_bpe(["the reserve bank of india sets monetary policy in mumbai"] * 20,
                              vocab_size=320, special_tokens=("<pad>", "<bos>", "<eos>", "<doc>"))
    tok = Tokenizer(vocab, merges, ("<pad>", "<bos>", "<eos>", "<doc>"))
    cfg = ModelConfig(vocab_size=len(tok.vocab), d_model=32, n_layers=1, n_heads=2,
                      n_experts=2, top_k=1, d_ff=64, seq_len=32)
    model = MoETransformer(cfg)
    pairs = [_Pair("p1", "rbi", "monetary policy is set by",
                   "the reserve bank of india", "the federal reserve")]
    rows = contrastive_delta_s(model, tok, pairs, seq_len=32)
    assert rows[0]["pair_id"] == "p1"
    assert rows[0]["delta_s"] == round(rows[0]["S_minus"] - rows[0]["S_plus"], 4)
