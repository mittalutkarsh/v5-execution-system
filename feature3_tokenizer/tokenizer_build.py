"""Epic 3.7 — build, freeze, and verify the tokenizer.

Ties 3.2-3.6 together: sample the cleaned corpus, train the BPE, run the
integrity check, serialize the artifact, and write a tokenizer_manifest.json
whose content hash is the tokenizer's identity. Re-freezing the same corpus
reproduces the same hash; a one-token change changes it. Every downstream stage
references that hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from feature3_tokenizer.bpe_tokenizer import Tokenizer
from feature3_tokenizer.bpe_train import DOCS_PER_LANE, MAX_CHARS_PER_DOC, sample_clean_corpus, train_bpe

__all__ = [
    "DEFAULT_SPECIALS",
    "DEFAULT_VOCAB_SIZE",
    "MANIFEST_FILE",
    "MANIFEST_KIND",
    "build_frozen_tokenizer",
    "load_frozen_tokenizer",
    "write_manifest",
    "verify_frozen",
]

DEFAULT_SPECIALS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<doc>")
DEFAULT_VOCAB_SIZE: int = 12_000
MANIFEST_FILE = "tokenizer_manifest.json"
MANIFEST_KIND = "tokenizer_manifest"
MANIFEST_VERSION = 1

_JSON = {"ensure_ascii": False, "sort_keys": True, "indent": 2}


def write_manifest(tok: Tokenizer, out_dir: str, *, trained_on: str) -> dict[str, Any]:
    """Write tokenizer_manifest.json and return the manifest dict."""
    manifest = {
        "kind": MANIFEST_KIND,
        "version": MANIFEST_VERSION,
        "hash": tok.content_hash(),
        "vocab_size": len(tok.vocab),
        "n_merges": len(tok.merges),
        "special_tokens": list(tok.special_tokens),
        "base": "bytes-256",
        "trained_on": trained_on,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / MANIFEST_FILE).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, **_JSON) + "\n")
    return manifest


def build_frozen_tokenizer(
    *,
    clean_root: str = "data/clean",
    out_dir: str = "tokenizer",
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    special_tokens: Sequence[str] = DEFAULT_SPECIALS,
    docs_per_lane: int = DOCS_PER_LANE,
    max_chars: int = MAX_CHARS_PER_DOC,
    texts: Sequence[str] | None = None,
) -> Tokenizer:
    """Train -> integrity-check -> serialize -> manifest. Returns the tokenizer.

    `texts` overrides corpus sampling (used by tests); otherwise a bounded,
    lane-balanced sample of `clean_root` is used.
    """
    sample = list(texts) if texts is not None else sample_clean_corpus(
        clean_root, docs_per_lane=docs_per_lane, max_chars=max_chars
    )
    vocab, merges = train_bpe(sample, vocab_size=vocab_size, special_tokens=special_tokens)
    tok = Tokenizer(vocab, merges, special_tokens).check_integrity()
    tok.save(out_dir)
    write_manifest(tok, out_dir, trained_on=clean_root)
    return tok


def load_frozen_tokenizer(out_dir: str = "tokenizer") -> Tokenizer:
    """Load a frozen tokenizer and re-run its integrity check."""
    return Tokenizer.load(out_dir).check_integrity()


def verify_frozen(out_dir: str = "tokenizer") -> bool:
    """True iff the on-disk tokenizer's recomputed hash matches its manifest."""
    manifest = json.loads((Path(out_dir) / MANIFEST_FILE).read_text(encoding="utf-8"))
    tok = load_frozen_tokenizer(out_dir)
    return tok.content_hash() == manifest["hash"]
