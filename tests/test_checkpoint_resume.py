"""Features 11-14 tests — checkpoint, resume, replay, fork. Offline."""

from __future__ import annotations

from feature8_packer.packer import pack_documents
from feature9_batches.batch_stream import BatchStream
from feature10_trainer.moe_model import ModelConfig
from feature10_trainer.trainer import Trainer
from feature11_checkpoint.checkpoint import load_checkpoint, model_hash, save_checkpoint, verify_checkpoint
from feature12_resume.resume import crash_and_resume, train_range
from feature13_replay.replay import replay_interval
from feature14_fork.fork import fork_run

VOCAB, SEQ = 48, 12


def _stream(seed="s"):
    docs = [{"doc_id": f"d{i}", "lane": "web",
             "tokens": [1] + [(i * 5 + j) % VOCAB for j in range(SEQ - 2)] + [2]}
            for i in range(16)]
    return BatchStream(pack_documents(docs, seq_len=SEQ, pad_id=0), seed=seed, batch_size=4)


def _cfg():
    return ModelConfig(vocab_size=VOCAB, d_model=24, n_layers=1, n_heads=2,
                       n_experts=2, top_k=1, d_ff=48, seq_len=SEQ)


def _make():
    return Trainer(_cfg(), seed="fixed", lr=1e-3)


# -- Feature 11: checkpoint ------------------------------------------------

def test_checkpoint_save_and_verify(tmp_path) -> None:
    tr = _make()
    train_range(tr, _stream(), 0, 3)
    manifest = save_checkpoint(tr, step=3, ledger_offset=3, seed="s", out_dir=str(tmp_path))
    assert manifest["ledger_offset"] == 3
    assert verify_checkpoint(str(tmp_path)) is True


def test_restore_reproduces_the_model(tmp_path) -> None:
    tr = _make()
    train_range(tr, _stream(), 0, 3)
    save_checkpoint(tr, step=3, ledger_offset=3, seed="s", out_dir=str(tmp_path))
    restored, _ = load_checkpoint(str(tmp_path))
    assert model_hash(restored.model) == model_hash(tr.model)


# -- Feature 12: crash + resume --------------------------------------------

def test_resume_matches_a_clean_run(tmp_path) -> None:
    result = crash_and_resume(_make, _stream(), total=8, crash_at=4,
                              checkpoint_dir=str(tmp_path), seed="s")
    assert result["resume_offset"] == 4
    assert result["no_skip_or_repeat"] is True
    assert result["loss_trajectory_matched"] is True   # model+opt+rng restored exactly


# -- Feature 13: replay -----------------------------------------------------

def test_replay_matches_the_ledger() -> None:
    stream = _stream()
    ledger = [stream.batch(i).as_ledger_row() for i in range(10)]
    result = replay_interval(stream, ledger, 2, 7)
    assert result["matched"] is True and result["checked"] == 5


def test_replay_detects_a_divergent_seed() -> None:
    original = _stream(seed="a")
    ledger = [original.batch(i).as_ledger_row() for i in range(10)]
    other = _stream(seed="b")                          # different seed -> won't match
    assert replay_interval(other, ledger, 0, 5)["matched"] is False


# -- Feature 14: fork -------------------------------------------------------

def test_fork_records_lineage_and_diverges(tmp_path) -> None:
    parent = _stream(seed="parent")
    tr = _make()
    train_range(tr, parent, 0, 4)
    save_checkpoint(tr, step=4, ledger_offset=4, seed="parent", out_dir=str(tmp_path))
    lineage = fork_run(str(tmp_path), parent, branch_id="exp-1", fork_seed="branch", steps=3)
    assert lineage["branch_id"] == "exp-1"
    assert lineage["parent"]["ledger_offset"] == 4
    assert lineage["diverged"] is True                 # new seed -> different batches
    assert len(lineage["branch_batch_ids"]) == 3
