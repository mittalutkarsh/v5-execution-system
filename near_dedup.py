"""Epic 2.4 — near-duplicate removal via MinHash + LSH. Pure Python, deterministic.

No external library. A stable base hash (zlib.crc32) over word n-gram shingles,
NUM_PERM fixed permutations generated once from a fixed seed, and banded LSH so
candidates are found without an all-pairs scan (near-linear on the corpus). The
first document of each near-duplicate cluster (in the order given) is kept; the
rest are dropped, each naming its survivor and the estimated Jaccard.
"""

from __future__ import annotations

import random
import re
import zlib
from typing import Any, Iterable

__all__ = [
    "NUM_PERM",
    "BANDS",
    "ROWS",
    "SHINGLE_N",
    "THRESHOLD",
    "shingles",
    "minhash",
    "est_jaccard",
    "dedup_near",
]

NUM_PERM: int = 32
BANDS: int = 16
ROWS: int = 2          # BANDS * ROWS == NUM_PERM
SHINGLE_N: int = 3     # word tri-grams
THRESHOLD: float = 0.8
_MERSENNE = (1 << 61) - 1
_SEED = "v5-near-dedup-2026"

_WORD = re.compile(r"\w+", re.UNICODE)

# Permutation coefficients, generated ONCE from a fixed seed -> deterministic.
_rng = random.Random(_SEED)
_PERMS: tuple[tuple[int, int], ...] = tuple(
    (_rng.randrange(1, _MERSENNE), _rng.randrange(0, _MERSENNE)) for _ in range(NUM_PERM)
)
_MAX = _MERSENNE  # sentinel for "no shingle"


def shingles(text: str, n: int = SHINGLE_N) -> set[int]:
    """Stable crc32 hashes of the lowercased word n-grams of `text`."""
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < n:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return {zlib.crc32(g.encode("utf-8")) & 0xFFFFFFFF for g in grams}


def minhash(text: str) -> tuple[int, ...]:
    """The NUM_PERM-long MinHash signature of `text`."""
    hs = shingles(text)
    if not hs:
        return tuple(_MAX for _ in range(NUM_PERM))
    sig = []
    for a, b in _PERMS:
        sig.append(min(((a * h + b) % _MERSENNE) for h in hs))
    return tuple(sig)


def est_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Estimated Jaccard similarity = fraction of matching signature slots."""
    if not sig_a:
        return 0.0
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


def _bands(sig: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
    """(band_index, band_slice) keys for LSH bucketing."""
    return [(b, sig[b * ROWS : (b + 1) * ROWS]) for b in range(BANDS)]


def dedup_near(
    docs: Iterable[dict[str, Any]],
    *,
    threshold: float = THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first of each near-duplicate cluster; drop the rest.

    Online LSH: each doc is compared only to already-kept docs that share a
    band bucket. Returns (kept, drops); a drop is
    {id, stage:"near_dup", reason, near:<survivor id>, est_jaccard}.
    """
    kept: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}  # band-key -> kept indices
    sigs: list[tuple[int, ...]] = []

    for doc in docs:
        sig = minhash(doc["text"])
        candidates: set[int] = set()
        band_keys = _bands(sig)
        for key in band_keys:
            candidates.update(buckets.get(key, ()))

        survivor = None
        best = 0.0
        for idx in candidates:
            j = est_jaccard(sig, sigs[idx])
            if j >= threshold and j >= best:
                best, survivor = j, idx

        if survivor is not None:
            drops.append(
                {
                    "id": doc["id"],
                    "stage": "near_dup",
                    "reason": f"~{best:.2f} Jaccard to an earlier document",
                    "near": kept[survivor]["id"],
                    "est_jaccard": round(best, 4),
                }
            )
        else:
            new_idx = len(kept)
            kept.append(doc)
            sigs.append(sig)
            for key in band_keys:
                buckets.setdefault(key, []).append(new_idx)
    return kept, drops
