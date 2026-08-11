"""Epics 3.3-3.6 tests — serialize/load, encode, decode, integrity. Offline."""

from __future__ import annotations

import pytest

from feature3_tokenizer.bpe_tokenizer import Tokenizer
from feature3_tokenizer.bpe_train import train_bpe

SPECIALS = ("<pad>", "<bos>", "<eos>", "<doc>")
CORPUS = [
    "the monsoon reaches kerala in june",
    "the cat sat on the mat and the rat ran",
    "हिन्दी भारत की भाषा है",
    "বাংলা ভাষা দক্ষিণ এশিয়ার ভাষা",
    "def add(a, b): return a + b",
] * 20

ROUNDTRIP = [
    "The monsoon reaches Kerala.",
    "हिन्दी भाषा 😀 ₹100",
    "বাংলা ভাষা — naïve café",
    "def add(a, b):\n    return a + b\n",
]


@pytest.fixture
def tok():
    vocab, merges = train_bpe(CORPUS, vocab_size=500, special_tokens=SPECIALS)
    return Tokenizer(vocab, merges, SPECIALS).check_integrity()


def test_encode_is_deterministic_and_ids_are_in_vocab(tok) -> None:
    ids = tok.encode("the monsoon reaches kerala")
    assert ids == tok.encode("the monsoon reaches kerala")
    assert all(0 <= i < len(tok.vocab) for i in ids)


@pytest.mark.parametrize("text", ROUNDTRIP)
def test_encode_decode_round_trip_is_lossless(tok, text) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_no_unk_on_a_script_never_seen_in_training(tok) -> None:
    # Tamil was not in CORPUS; byte-level still round-trips it exactly
    tamil = "தமிழ் ஒரு செம்மொழி ஆகும்."
    assert tok.decode(tok.encode(tamil)) == tamil


def test_special_tokens_have_reserved_low_ids(tok) -> None:
    assert [tok.vocab[s] for s in SPECIALS] == [0, 1, 2, 3]


def test_decode_skips_special_tokens(tok) -> None:
    ids = [tok.vocab["<bos>"]] + tok.encode("hello") + [tok.vocab["<eos>"]]
    assert tok.decode(ids) == "hello"


def test_save_load_round_trip_and_encode_matches(tok, tmp_path) -> None:
    tok.save(str(tmp_path))
    loaded = Tokenizer.load(str(tmp_path)).check_integrity()
    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.encode("the monsoon") == tok.encode("the monsoon")


def test_serialization_is_byte_identical(tok, tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    tok.save(str(a))
    Tokenizer.load(str(a)).save(str(b))
    for name in ("vocab.json", "merges.txt", "special_tokens.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_integrity_rejects_a_missing_byte(tok) -> None:
    from feature3_tokenizer.byte_level import BYTE_TO_UNICODE
    broken = dict(tok.vocab)
    freed = broken.pop(BYTE_TO_UNICODE[65])   # drop the symbol for byte 'A'...
    broken["<placeholder>"] = freed           # ...but keep ids contiguous
    with pytest.raises(ValueError, match="missing from the vocab"):
        Tokenizer(broken, tok.merges, tok.special_tokens).check_integrity()


def test_integrity_rejects_duplicate_ids(tok) -> None:
    broken = dict(tok.vocab)
    first = next(iter(broken))
    broken[first] = 1   # collide with <bos>'s id
    with pytest.raises(ValueError, match="contiguous"):
        Tokenizer(broken, tok.merges, tok.special_tokens).check_integrity()


def test_content_hash_is_stable_and_sensitive(tok) -> None:
    assert tok.content_hash() == tok.content_hash()
    # a one-token change must change the hash
    changed = dict(tok.vocab)
    changed["<extra>"] = len(changed)
    other = Tokenizer(changed, tok.merges, tok.special_tokens)
    assert other.content_hash() != tok.content_hash()
