"""Epic 2.6 — decontaminate train against eval + contrastive.

Builds the set of n-gram (default 13-token) shingles from the eval documents
and the contrastive continuations, then drops any TRAIN document that shares at
least K of them. Only train documents can be dropped; eval and contrastive are
protected — they define the benchmark, so they are never removed here.

Two refinements over a naive overlap check:

  * Boilerplate guard. An n-gram that recurs across many eval documents is a
    template / navigation phrase, not benchmark content. Such high
    document-frequency n-grams are excluded from the contamination set, so a
    train document is not discarded merely for sharing Wikipedia boilerplate
    with the eval slice — a real risk for the small, template-heavy Indic and
    multilingual lanes we most want to keep.

  * Auditability. Every drop records which eval/contrastive source(s) it
    matched and a sample overlapping n-gram, so a human can tell genuine
    leakage from a false positive instead of trusting a bare count.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from text_tokens import words as _words

__all__ = [
    "NGRAM_N",
    "MIN_OVERLAP",
    "MAX_EVAL_DOC_FREQ",
    "ngrams",
    "build_contam_index",
    "contaminating_ngrams",
    "decontaminate",
]

NGRAM_N: int = 13
MIN_OVERLAP: int = 1        # sharing even one *content* 13-gram is contamination
MAX_EVAL_DOC_FREQ: float = 0.30  # n-grams recurring across >30% of eval docs = boilerplate
_MAX_MATCHED_REPORTED: int = 5   # cap the matched-source list in a drop record


def ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    """The set of lowercased word n-grams in `text` (empty if too short)."""
    toks = [w.lower() for w in _words(text)]
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def build_contam_index(
    eval_docs: Iterable[dict[str, Any]],
    contrastive_pairs: Sequence[Any] = (),
    *,
    n: int = NGRAM_N,
) -> dict[tuple[str, ...], set[str]]:
    """Map each n-gram -> the set of eval/contrastive source ids it appears in."""
    index: dict[tuple[str, ...], set[str]] = {}

    def add(source_id: str, text: str) -> None:
        for gram in ngrams(text, n):
            index.setdefault(gram, set()).add(source_id)

    for doc in eval_docs:
        add(doc["id"], doc["text"])
    for pair in contrastive_pairs:
        add(pair.id, f"{pair.prefix} {pair.y_plus}")
        add(pair.id, f"{pair.prefix} {pair.y_minus}")
    return index


def contaminating_ngrams(
    index: dict[tuple[str, ...], set[str]],
    n_sources: int,
    *,
    max_doc_freq: float = MAX_EVAL_DOC_FREQ,
) -> dict[tuple[str, ...], set[str]]:
    """The contaminating n-grams: those NOT recurring across too many sources.

    An n-gram found in more than `max_doc_freq` of the eval/contrastive sources
    is boilerplate (a shared template), so it is dropped from the set — it must
    not, on its own, condemn a train document.
    """
    if n_sources <= 0:
        return {}
    cap = max(1, int(max_doc_freq * n_sources))
    return {gram: ids for gram, ids in index.items() if len(ids) <= cap}


def decontaminate(
    train_docs: Iterable[dict[str, Any]],
    eval_docs: Iterable[dict[str, Any]],
    contrastive_pairs: Sequence[Any] = (),
    *,
    n: int = NGRAM_N,
    min_overlap: int = MIN_OVERLAP,
    max_eval_doc_freq: float = MAX_EVAL_DOC_FREQ,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop train docs overlapping eval/contrastive content. Returns (kept, drops).

    Each drop records the overlap count, the matched source id(s), and a sample
    overlapping n-gram, so a false positive can be told from real leakage.
    """
    eval_docs = list(eval_docs)
    contrastive_pairs = list(contrastive_pairs)
    index = build_contam_index(eval_docs, contrastive_pairs, n=n)
    n_sources = len(eval_docs) + len(contrastive_pairs)
    contam = contaminating_ngrams(index, n_sources, max_doc_freq=max_eval_doc_freq)

    kept: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for doc in train_docs:
        matched = ngrams(doc["text"], n) & contam.keys()
        if len(matched) >= min_overlap:
            sources: set[str] = set()
            for gram in matched:
                sources |= contam[gram]
            sample = " ".join(min(matched))  # deterministic sample n-gram
            drops.append(
                {
                    "id": doc["id"],
                    "stage": "decontam",
                    "reason": f"shares {len(matched)} {n}-gram(s) with eval/contrastive",
                    "overlap": len(matched),
                    "matched_sources": sorted(sources)[:_MAX_MATCHED_REPORTED],
                    "sample_ngram": sample,
                }
            )
        else:
            kept.append(doc)
    return kept, drops
