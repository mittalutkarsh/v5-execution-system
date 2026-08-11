"""Epic 3.2 tests — the deterministic BPE trainer. Fully offline."""

from __future__ import annotations

from byte_level import BYTE_TO_UNICODE
from bpe_train import train_bpe

CORPUS = [
    "the cat sat on the mat",
    "the cat ate the rat",
    "a cat and a rat",
    "बिल्ली और चूहा",          # Hindi: cat and mouse
    "the the the the the",
] * 20


def test_base_vocab_covers_specials_then_all_256_bytes() -> None:
    vocab, _ = train_bpe(CORPUS, vocab_size=300, special_tokens=("<pad>", "<eos>"))
    assert vocab["<pad>"] == 0 and vocab["<eos>"] == 1
    for b in range(256):
        assert BYTE_TO_UNICODE[b] in vocab
    # ids are contiguous 0..N-1
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_target_vocab_size_is_an_upper_bound_with_consistent_accounting() -> None:
    vocab, merges = train_bpe(CORPUS, vocab_size=300, special_tokens=("<pad>",))
    assert len(vocab) <= 300                      # never exceeds the target
    assert len(vocab) == (1 + 256) + len(merges)  # specials + 256 bytes + merges


def test_training_is_deterministic() -> None:
    a = train_bpe(CORPUS, vocab_size=320, special_tokens=("<pad>",))
    b = train_bpe(list(CORPUS), vocab_size=320, special_tokens=("<pad>",))
    assert a[0] == b[0] and a[1] == b[1]


def test_merges_reference_existing_tokens_in_order() -> None:
    vocab, merges = train_bpe(CORPUS, vocab_size=320)
    for a, b in merges:
        assert a in vocab and b in vocab and (a + b) in vocab


def test_a_frequent_word_is_learned_as_a_token() -> None:
    # "the" is the most common word; BPE builds it up (t + he -> the)
    vocab, merges = train_bpe(CORPUS, vocab_size=280)
    joined = ["".join(m) for m in merges]
    assert "the" in joined


def test_runs_out_of_pairs_gracefully() -> None:
    # ask for far more merges than the tiny corpus can supply
    vocab, merges = train_bpe(["ab ab ab"], vocab_size=12000)
    assert len(vocab) < 12000            # stopped when no pairs remained
    assert sorted(vocab.values()) == list(range(len(vocab)))
