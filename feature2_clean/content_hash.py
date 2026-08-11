"""Epic 2.2 — the content hasher (born here) + exact-duplicate removal.

content_hash is sha256 over the NFC-normalised UTF-8 bytes, so it is stable
across runs and machines and independent of dict order or wall-clock.
dedup_exact keeps the first occurrence of each hash in the order given, and
records every later exact duplicate as a drop naming its survivor.
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Any, Iterable

__all__ = ["content_hash", "dedup_exact"]


def content_hash(text: str) -> str:
    """sha256 over the NFC-normalised UTF-8 bytes of `text`."""
    canonical = unicodedata.normalize("NFC", text).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def dedup_exact(
    docs: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first doc per content_hash; drop later exact duplicates.

    Returns (kept, drops). A drop is
    {id, stage:"exact_dup", reason, duplicate_of:<survivor id>}.
    """
    kept: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    survivor_by_hash: dict[str, str] = {}

    for doc in docs:
        digest = content_hash(doc["text"])
        survivor = survivor_by_hash.get(digest)
        if survivor is None:
            survivor_by_hash[digest] = doc["id"]
            kept.append(doc)
        else:
            drops.append(
                {
                    "id": doc["id"],
                    "stage": "exact_dup",
                    "reason": "identical content to an earlier document",
                    "duplicate_of": survivor,
                }
            )
    return kept, drops
