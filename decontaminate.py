"""Epic 2.6 — decontaminate train against eval + contrastive.

Builds the set of n-gram (default 13-token) shingles from the eval documents
and the contrastive continuations, then drops any TRAIN document that shares at
least K of them. Only train documents can be dropped; eval and contrastive are
protected — they define the benchmark, so they are never removed here.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

__all__ = ["NGRAM_N", "MIN_OVERLAP", "ngrams", "build_contam_ngrams", "decontaminate"]

NGRAM_N: int = 13
MIN_OVERLAP: int = 1  # sharing even one 13-gram with eval is contamination

_WORD = re.compile(r"\w+", re.UNICODE)


def ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    """The set of lowercased word n-grams in `text` (empty if too short)."""
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_contam_ngrams(
    eval_docs: Iterable[dict[str, Any]],
    contrastive_pairs: Sequence[Any] = (),
    *,
    n: int = NGRAM_N,
) -> set[tuple[str, ...]]:
    """Every n-gram that appears in the eval set or a contrastive continuation."""
    contam: set[tuple[str, ...]] = set()
    for doc in eval_docs:
        contam |= ngrams(doc["text"], n)
    for pair in contrastive_pairs:
        contam |= ngrams(f"{pair.prefix} {pair.y_plus}", n)
        contam |= ngrams(f"{pair.prefix} {pair.y_minus}", n)
    return contam


def decontaminate(
    train_docs: Iterable[dict[str, Any]],
    eval_docs: Iterable[dict[str, Any]],
    contrastive_pairs: Sequence[Any] = (),
    *,
    n: int = NGRAM_N,
    min_overlap: int = MIN_OVERLAP,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop train docs overlapping eval/contrastive. Returns (kept, drops)."""
    contam = build_contam_ngrams(eval_docs, contrastive_pairs, n=n)
    kept: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for doc in train_docs:
        overlap = len(ngrams(doc["text"], n) & contam)
        if overlap >= min_overlap:
            drops.append(
                {
                    "id": doc["id"],
                    "stage": "decontam",
                    "reason": f"shares {overlap} {n}-gram(s) with eval/contrastive",
                    "overlap": overlap,
                }
            )
        else:
            kept.append(doc)
    return kept, drops
