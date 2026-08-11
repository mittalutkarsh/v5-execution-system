"""Epic 3.2 — a deterministic byte-level BPE trainer.

Counts adjacent symbol-pair frequencies inside pre-tokens, merges the most
frequent pair, and repeats until the target vocab size. The tie-break is total
and fixed (highest count, then the lexicographically smallest pair), so the
learned (vocab, merges) are reproducible regardless of dict iteration order --
no wall-clock, no randomness.

Performance: pair counts are maintained incrementally and a lazy max-heap picks
the best pair, so training is near-linear in the corpus rather than rescanning
every pair on every merge. This is what makes a 12k-merge run finish in seconds.
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any, Iterable

from byte_level import BYTE_TO_UNICODE, pretokens_as_symbols

__all__ = ["train_bpe", "sample_clean_corpus", "DOCS_PER_LANE", "MAX_CHARS_PER_DOC"]

DOCS_PER_LANE: int = 300
MAX_CHARS_PER_DOC: int = 2000


def train_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int,
    special_tokens: Iterable[str] = (),
) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Learn a byte-level BPE vocab + ordered merges from `texts`.

    Returns (vocab, merges) where vocab maps token -> contiguous id (special
    tokens first, then the 256 byte symbols, then merges in learn order) and
    merges is the ordered list of (a, b) pairs.
    """
    specials = list(special_tokens)

    # 1. word (pre-token) frequencies, each a tuple of byte-symbols
    word_freq: dict[tuple[str, ...], int] = {}
    for text in texts:
        for sym in pretokens_as_symbols(text):
            if not sym:
                continue
            w = tuple(sym)
            word_freq[w] = word_freq.get(w, 0) + 1

    # 2. base vocab: specials, then all 256 byte symbols (every byte is covered)
    vocab: dict[str, int] = {}
    for tok in specials + [BYTE_TO_UNICODE[b] for b in range(256)]:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    special_set = set(specials)
    target_merges = max(0, vocab_size - len(vocab))

    words: list[list[str]] = [list(w) for w in word_freq]
    counts: list[int] = [word_freq[w] for w in word_freq]

    pair_count: dict[tuple[str, str], int] = {}
    heap: list[tuple[int, tuple[str, str]]] = []  # (-count, pair), lazily validated

    def touch(pair: tuple[str, str], delta: int) -> None:
        c = pair_count.get(pair, 0) + delta
        pair_count[pair] = c
        if c > 0:
            heapq.heappush(heap, (-c, pair))

    def apply_word_pairs(i: int, sign: int) -> None:
        w, c = words[i], counts[i]
        for j in range(len(w) - 1):
            touch((w[j], w[j + 1]), sign * c)

    pair_where: dict[tuple[str, str], set[int]] = {}
    for i in range(len(words)):
        w, c = words[i], counts[i]
        for j in range(len(w) - 1):
            p = (w[j], w[j + 1])
            pair_count[p] = pair_count.get(p, 0) + c
            pair_where.setdefault(p, set()).add(i)
    for p, c in pair_count.items():
        heapq.heappush(heap, (-c, p))

    merges: list[tuple[str, str]] = []
    while len(merges) < target_merges:
        # pop the current best pair (max count, then smallest pair); skip stale
        best = None
        while heap:
            negc, p = heapq.heappop(heap)
            if pair_count.get(p, 0) == -negc and -negc > 0:
                best = p
                break
        if best is None:
            break

        a, b = best
        merged = a + b
        if merged in special_set or merged in vocab:
            pair_count[best] = 0  # disable this colliding pair and re-select
            continue

        for i in list(pair_where.get(best, ())):
            w = words[i]
            if len(w) < 2:
                continue
            apply_word_pairs(i, -1)  # remove this word's old pair contributions
            new: list[str] = []
            k = 0
            while k < len(w):
                if k < len(w) - 1 and w[k] == a and w[k + 1] == b:
                    new.append(merged)
                    k += 2
                else:
                    new.append(w[k])
                    k += 1
            words[i] = new
            apply_word_pairs(i, +1)  # add the new pair contributions
            for j in range(len(new) - 1):
                if new[j] == merged or new[j + 1] == merged:
                    pair_where.setdefault((new[j], new[j + 1]), set()).add(i)

        pair_count[best] = 0
        vocab[merged] = len(vocab)
        merges.append((a, b))

    return vocab, merges


def sample_clean_corpus(
    clean_root: str = "data/clean",
    *,
    docs_per_lane: int = DOCS_PER_LANE,
    max_chars: int = MAX_CHARS_PER_DOC,
) -> list[str]:
    """A bounded, lane-balanced, deterministic sample of the cleaned corpus.

    Up to `docs_per_lane` documents per lane, each truncated to `max_chars`, in
    a fixed order (lanes sorted, documents in file order). Reading order is
    stable, so the training sample -- and therefore the tokenizer -- is
    reproducible.
    """
    root = Path(clean_root)
    per_lane: dict[str, list[str]] = {}
    for docs_file in sorted(root.rglob("documents.jsonl")):
        for line in docs_file.read_text(encoding="utf-8").splitlines():
            doc: dict[str, Any] = json.loads(line)
            lane = doc.get("lane", "?")
            bucket = per_lane.setdefault(lane, [])
            if len(bucket) >= docs_per_lane:
                break
            bucket.append(doc["text"][:max_chars])
    return [t for lane in sorted(per_lane) for t in per_lane[lane]]
