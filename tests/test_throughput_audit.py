"""Features 15-16 tests — throughput + audit/evidence. Offline."""

from __future__ import annotations

import json
from pathlib import Path

from feature8_packer.packer import pack_documents, packed_batch_report
from feature9_batches.batch_stream import BatchStream
from feature10_trainer.moe_model import ModelConfig
from feature10_trainer.trainer import Trainer
from feature15_throughput.throughput import build_performance, measure_throughput, packing_utilization
from feature16_audit.audit import EXPECTED_PASS_EVENTS, REQUIREMENTS

VOCAB, SEQ = 48, 12


def _stream():
    docs = [{"doc_id": f"d{i}", "lane": "web",
             "tokens": [1] + [(i + j) % VOCAB for j in range(SEQ - 2)] + [2]} for i in range(12)]
    return BatchStream(pack_documents(docs, seq_len=SEQ, pad_id=0), seed="s", batch_size=4)


def _make():
    return Trainer(ModelConfig(vocab_size=VOCAB, d_model=24, n_layers=1, n_heads=2,
                               n_experts=2, top_k=1, d_ff=48, seq_len=SEQ), seed="f", lr=1e-3)


def test_packing_utilization_is_a_fraction() -> None:
    seqs = pack_documents([{"doc_id": "a", "lane": "web", "tokens": [1, 2, 3, 2]}], seq_len=SEQ)
    rep = packed_batch_report(seqs, seq_len=SEQ)
    assert 0 < packing_utilization(rep) <= 1


def test_measure_throughput_reports_positive_rate() -> None:
    perf = measure_throughput(_make, _stream(), n_steps=3)
    assert perf["steps"] == 3 and perf["loss_bearing_tokens"] > 0
    assert perf["loss_tokens_per_sec"] >= 0


def test_build_performance_structure() -> None:
    seqs = pack_documents([{"doc_id": "a", "lane": "web", "tokens": [1, 2, 3, 2]}], seq_len=SEQ)
    rep = packed_batch_report(seqs, seq_len=SEQ)
    perf = build_performance(rep, measure_throughput(_make, _stream(), n_steps=2))
    assert perf["kind"] == "performance" and "throughput" in perf


def test_expected_events_include_the_required_names() -> None:
    for required in ("tokenizer_hash_verified", "eval_shard_blocked", "checkpoint_saved",
                     "resume_next_batch_matched", "replay_hash_matched", "manifests_validated"):
        assert required in EXPECTED_PASS_EVENTS


def test_requirements_table_covers_the_rubric() -> None:
    labels = {r[0] for r in REQUIREMENTS}
    assert labels == {"Tokenizer integrity", "Evaluation firewall", "Packing correctness",
                      "Mixture compliance", "OPUS audit trail", "Crash recovery", "Replay",
                      "Learning trace", "Throughput"}
