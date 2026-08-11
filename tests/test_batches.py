"""Feature 9 tests — deterministic batch stream + consumption ledger. Offline."""

from __future__ import annotations

from feature8_packer.packer import pack_documents
from feature9_batches.batch_stream import BatchStream, ConsumptionLedger, verify_reconstruction
from feature9_batches.rng import derive_seed


def _pool():
    # seq_len 4 == doc length, so each doc is its own sequence and keeps its lane
    docs = [{"doc_id": f"d{i}", "lane": ("indic" if i % 2 else "web"),
             "tokens": [1, 100 + i, 200 + i, 2]} for i in range(24)]
    return pack_documents(docs, seq_len=4, pad_id=0)


def test_derive_seed_is_stable_and_index_specific() -> None:
    assert derive_seed("s", 5) == derive_seed("s", 5)
    assert derive_seed("s", 5) != derive_seed("s", 6)


def test_same_seed_same_batches() -> None:
    seqs = _pool()
    a = BatchStream(seqs, seed="x", batch_size=4)
    b = BatchStream(seqs, seed="x", batch_size=4)
    assert [a.batch(i).content_hash for i in range(10)] == [b.batch(i).content_hash for i in range(10)]


def test_different_seed_different_batches() -> None:
    seqs = _pool()
    a = BatchStream(seqs, seed="x", batch_size=4)
    b = BatchStream(seqs, seed="y", batch_size=4)
    assert a.batch(0).content_hash != b.batch(0).content_hash


def test_batch_is_reconstructible_out_of_order() -> None:
    # THE invariant: batch 7 can be rebuilt directly, without touching 0..6
    seqs = _pool()
    s = BatchStream(seqs, seed="x", batch_size=4)
    direct = s.batch(7)
    again = BatchStream(seqs, seed="x", batch_size=4).batch(7)
    assert direct.content_hash == again.content_hash
    assert direct.seq_indices == again.seq_indices


def test_seed_plus_offset_reconstructs_any_batch(tmp_path) -> None:
    seqs = _pool()
    stream = BatchStream(seqs, seed="x", batch_size=4)
    ledger = ConsumptionLedger(str(tmp_path / "consumption.jsonl"))
    for i in range(12):
        ledger.record(stream.batch(i))
    ledger.close()
    # a fresh stream (only the seed) reconstructs every recorded batch + hash
    fresh = BatchStream(seqs, seed="x", batch_size=4)
    assert verify_reconstruction(fresh, ledger.rows) is True


def test_reconstruction_fails_on_a_tampered_hash(tmp_path) -> None:
    seqs = _pool()
    stream = BatchStream(seqs, seed="x", batch_size=4)
    rows = [stream.batch(i).as_ledger_row() for i in range(4)]
    rows[2]["content_hash"] = "0" * 64
    assert verify_reconstruction(stream, rows) is False


def test_mixture_weights_bias_lane_draws() -> None:
    seqs = _pool()
    # weight indic heavily; over many draws most sequences should be indic
    stream = BatchStream(seqs, seed="x", batch_size=8, lane_weights={"indic": 0.95, "web": 0.05})
    indic_idx = {j for j, s in enumerate(seqs) if s.lane == "indic"}
    picks = [j for i in range(20) for j in stream.batch(i).seq_indices]
    share = sum(1 for j in picks if j in indic_idx) / len(picks)
    assert share > 0.75


def test_ledger_offset_tracks_consumption(tmp_path) -> None:
    seqs = _pool()
    stream = BatchStream(seqs, seed="x", batch_size=4)
    ledger = ConsumptionLedger(str(tmp_path / "c.jsonl"))
    assert ledger.offset == 0
    for i in range(5):
        ledger.record(stream.batch(i))
    assert ledger.offset == 5
    ledger.close()
