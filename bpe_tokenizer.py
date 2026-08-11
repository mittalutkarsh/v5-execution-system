"""Epics 3.3-3.6 — the frozen tokenizer object.

  3.3  serialize / load  vocab.json + merges.txt + special_tokens.json (deterministic bytes)
  3.4  encode(text) -> ids   (greedy merge-by-rank within each pre-token; no <unk>)
  3.5  decode(ids) -> text   (byte-level inverse; decode(encode(x)) == x on every lane)
  3.6  special tokens with reserved ids + an integrity check

A merged/byte token is stored in the printable-symbol view; a special token is a
literal string (e.g. "<pad>") that never arises from encoding ordinary text, so
the two never collide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from byte_level import (
    BYTE_TO_UNICODE,
    decode_from_symbols,
    pretokens_as_symbols,
)
from content_hash import content_hash as _hash_text

__all__ = ["Tokenizer", "VOCAB_FILE", "MERGES_FILE", "SPECIALS_FILE", "MERGES_HEADER"]

VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"
SPECIALS_FILE = "special_tokens.json"
MERGES_HEADER = "# v5-bpe merges v1"

_JSON = {"ensure_ascii": False, "sort_keys": True, "indent": 2}


class Tokenizer:
    """A frozen byte-level BPE tokenizer: encode, decode, serialize, verify."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: Iterable[tuple[str, str]],
        special_tokens: Iterable[str] = (),
    ) -> None:
        self.vocab: dict[str, int] = dict(vocab)
        self.merges: list[tuple[str, str]] = [tuple(m) for m in merges]
        self.special_tokens: list[str] = list(special_tokens)
        self.id_to_token: dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.ranks: dict[tuple[str, str], int] = {m: i for i, m in enumerate(self.merges)}
        self.special_set = set(self.special_tokens)

    # -- 3.4 encode ------------------------------------------------------------

    def _bpe_word(self, symbols: str) -> list[str]:
        """Greedily apply the lowest-rank merge available until none remain."""
        word = list(symbols)
        if len(word) < 2:
            return word
        while True:
            best_rank = None
            best_i = -1
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i < 0:
                return word
            word[best_i : best_i + 2] = [word[best_i] + word[best_i + 1]]

    def encode(self, text: str) -> list[int]:
        """Text -> token ids. Deterministic; byte-level, so never emits <unk>."""
        ids: list[int] = []
        for sym in pretokens_as_symbols(text):
            for tok in self._bpe_word(sym):
                ids.append(self.vocab[tok])
        return ids

    # -- 3.5 decode ------------------------------------------------------------

    def decode(self, ids: Iterable[int]) -> str:
        """Token ids -> text. Special tokens carry no bytes and are skipped."""
        symbols = "".join(
            self.id_to_token[i] for i in ids if self.id_to_token[i] not in self.special_set
        )
        return decode_from_symbols(symbols)

    # -- 3.6 integrity ---------------------------------------------------------

    def check_integrity(self) -> "Tokenizer":
        """Raise ValueError if the tokenizer is not internally consistent."""
        ids = sorted(self.vocab.values())
        if ids != list(range(len(self.vocab))):
            raise ValueError("token ids are not a contiguous 0..N-1 range without duplicates")
        for b in range(256):
            if BYTE_TO_UNICODE[b] not in self.vocab:
                raise ValueError(f"byte {b} ({BYTE_TO_UNICODE[b]!r}) missing from the vocab")
        for a, b in self.merges:
            if a not in self.vocab or b not in self.vocab or (a + b) not in self.vocab:
                raise ValueError(f"merge ({a!r}, {b!r}) references a token not in the vocab")
        for s in self.special_tokens:
            if s not in self.vocab:
                raise ValueError(f"special token {s!r} missing from the vocab")
        byte_syms = set(BYTE_TO_UNICODE.values())
        merge_toks = {a + b for a, b in self.merges}
        for s in self.special_tokens:
            if s in byte_syms or s in merge_toks:
                raise ValueError(f"special token {s!r} collides with a byte/merge token")
        return self

    # -- 3.3 serialize / load --------------------------------------------------

    def save(self, out_dir: str) -> None:
        """Write vocab.json + merges.txt + special_tokens.json (deterministic bytes)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / VOCAB_FILE).open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(self.vocab, **_JSON) + "\n")
        with (out / MERGES_FILE).open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(MERGES_HEADER + "\n")
            for a, b in self.merges:
                fh.write(f"{a} {b}\n")
        with (out / SPECIALS_FILE).open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(self.special_tokens, ensure_ascii=False, indent=2) + "\n")

    @classmethod
    def load(cls, in_dir: str) -> "Tokenizer":
        """Read a tokenizer written by save()."""
        src = Path(in_dir)
        vocab = json.loads((src / VOCAB_FILE).read_text(encoding="utf-8"))
        specials = json.loads((src / SPECIALS_FILE).read_text(encoding="utf-8"))
        merges: list[tuple[str, str]] = []
        for line in (src / MERGES_FILE).read_text(encoding="utf-8").splitlines():
            # skip ONLY the exact header line, never a merge whose first symbol
            # is "#" (the byte '#' maps to the symbol '#', so "# x" is real data)
            if not line or line == MERGES_HEADER:
                continue
            a, b = line.split(" ")
            merges.append((a, b))
        return cls(vocab, merges, specials)

    # -- 3.7 canonical form + content hash ------------------------------------

    def canonical_str(self) -> str:
        """A deterministic string identity of the tokenizer (order-independent)."""
        obj = {
            "special_tokens": list(self.special_tokens),
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
        }
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def content_hash(self) -> str:
        """sha256 over the canonical form; the tokenizer's frozen identity."""
        return _hash_text(self.canonical_str())
