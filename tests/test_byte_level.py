"""Epic 3.1 tests — the byte-level foundation. Fully offline."""

from __future__ import annotations

import pytest

from feature3_tokenizer.byte_level import (
    BYTE_TO_UNICODE,
    UNICODE_TO_BYTE,
    byte_decode,
    byte_encode,
    decode_from_symbols,
    encode_to_symbols,
    pretokenize,
    pretokens_as_symbols,
)

SAMPLES = {
    "web": "The monsoon usually reaches Kerala around the first of June.",
    "code": "def add(a, b):\n    return a + b  # sum\n",
    "math": "Let x = 3.14; then 2*x > 6 and x^2 ≈ 9.87.",
    "hindi": "हिन्दी भारत की एक प्रमुख भाषा है।",
    "bn": "বাংলা ভাষা দক্ষিণ এশিয়ার একটি প্রধান ভাষা।",
    "tamil": "தமிழ் ஒரு செம்மொழி ஆகும்.",
    "mix": "email a@b.com 😀 ₹100 — naïve café",
}


@pytest.mark.parametrize("name", list(SAMPLES))
def test_byte_round_trip_is_lossless(name) -> None:
    s = SAMPLES[name]
    assert byte_decode(byte_encode(s)) == s


@pytest.mark.parametrize("name", list(SAMPLES))
def test_symbol_round_trip_is_lossless(name) -> None:
    s = SAMPLES[name]
    assert decode_from_symbols(encode_to_symbols(s)) == s


def test_byte_to_unicode_is_a_bijection_over_256_printable_chars() -> None:
    assert len(BYTE_TO_UNICODE) == 256
    values = list(BYTE_TO_UNICODE.values())
    assert len(set(values)) == 256                 # all distinct
    assert all(len(v) == 1 and not v.isspace() for v in values)  # single, printable
    assert UNICODE_TO_BYTE == {v: k for k, v in BYTE_TO_UNICODE.items()}


@pytest.mark.parametrize("name", list(SAMPLES))
def test_pretokenize_is_lossless(name) -> None:
    s = SAMPLES[name]
    assert "".join(pretokenize(s)) == s


def test_leading_space_attaches_to_the_word() -> None:
    assert pretokenize("the monsoon reaches") == ["the", " monsoon", " reaches"]


def test_a_whitespace_run_is_its_own_piece() -> None:
    assert pretokenize("a\n\nb") == ["a", "\n\n", "b"]


def test_indic_word_survives_as_one_pretoken() -> None:
    assert pretokenize("हिन्दी भाषा") == ["हिन्दी", " भाषा"]
    assert pretokenize("বাংলা ভাষা") == ["বাংলা", " ভাষা"]


def test_pretokens_as_symbols_matches_manual_mapping() -> None:
    text = "hi there"
    assert pretokens_as_symbols(text) == [encode_to_symbols(pt) for pt in pretokenize(text)]


def test_everything_is_deterministic() -> None:
    s = SAMPLES["mix"]
    assert byte_encode(s) == byte_encode(s)
    assert encode_to_symbols(s) == encode_to_symbols(s)
    assert pretokenize(s) == pretokenize(s)
