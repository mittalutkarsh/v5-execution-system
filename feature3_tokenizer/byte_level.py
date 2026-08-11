"""Epic 3.1 — the byte-level foundation for a byte-level BPE tokenizer.

No merges yet: just (a) a lossless byte layer, (b) a lossless printable-symbol
view of those bytes (the GPT-2 byte<->unicode trick, so vocab/merges can be
stored and diffed as text), and (c) a pre-tokenizer that splits text into words
without ever splitting inside one.

Because the base alphabet is all 256 byte values, EVERY lane is representable
with no <unk> and nothing to fragment -- the India-first payoff: Devanagari,
Bengali and Tamil (and emoji, and code) all survive losslessly. Pure stdlib,
fully deterministic.
"""

from __future__ import annotations

import re

__all__ = [
    "BYTE_TO_UNICODE",
    "UNICODE_TO_BYTE",
    "PRETOKEN_RE",
    "byte_encode",
    "byte_decode",
    "encode_to_symbols",
    "decode_from_symbols",
    "pretokenize",
    "pretokens_as_symbols",
]


def _bytes_to_unicode() -> dict[int, str]:
    """The classic GPT-2 construction: each of the 256 bytes -> one printable char."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_UNICODE: dict[int, str] = _bytes_to_unicode()
UNICODE_TO_BYTE: dict[str, int] = {c: b for b, c in BYTE_TO_UNICODE.items()}

# A single leading space attaches to the following word (GPT-2 style); a run of
# whitespace is its own piece; a "word" is a maximal run of non-whitespace, so an
# Indic syllable and its combining marks are never split and no later merge can
# cross a whitespace boundary.
PRETOKEN_RE = re.compile(r" ?\S+|\s+")


def byte_encode(text: str) -> list[int]:
    """The UTF-8 bytes of `text` as ints 0..255."""
    return list(text.encode("utf-8"))


def byte_decode(byte_ids: list[int]) -> str:
    """Inverse of byte_encode."""
    return bytes(byte_ids).decode("utf-8")


def encode_to_symbols(text: str) -> str:
    """Map each UTF-8 byte of `text` to its printable symbol char and join."""
    return "".join(BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))


def decode_from_symbols(symbols: str) -> str:
    """Inverse of encode_to_symbols: symbols -> bytes -> text."""
    return bytes(UNICODE_TO_BYTE[c] for c in symbols).decode("utf-8")


def pretokenize(text: str) -> list[str]:
    """Split `text` into word-pieces. Lossless: "".join(pretokenize(t)) == t."""
    return PRETOKEN_RE.findall(text)


def pretokens_as_symbols(text: str) -> list[str]:
    """Each pre-token rendered in the printable-symbol view (BPE's input unit)."""
    return [encode_to_symbols(pt) for pt in pretokenize(text)]
