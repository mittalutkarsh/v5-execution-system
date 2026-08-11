"""Epic 2.3 — quality filter. Deterministic heuristics with named thresholds.

Lenient by design: the goal is to drop obvious junk (too short, symbol spam,
single-token repetition), not to reshape the corpus. Each check returns the
first failing reason so a drop can say exactly why.
"""

from __future__ import annotations

from typing import Any, Iterable

from feature2_clean.text_tokens import is_word_char
from feature2_clean.text_tokens import words as _words

__all__ = [
    "MIN_CHARS",
    "MIN_WORDS",
    "MAX_SYMBOL_RATIO",
    "MAX_WORD_REPETITION",
    "quality_ok",
    "filter_quality",
]

MIN_CHARS: int = 20              # shorter than this is not a document
MIN_WORDS: int = 5
MAX_SYMBOL_RATIO: float = 0.5    # fraction of non-word (punctuation/symbol) chars
MAX_WORD_REPETITION: float = 0.6  # fraction of tokens that are the single commonest word


def quality_ok(text: str) -> tuple[bool, str | None]:
    """(True, None) if the text passes, else (False, reason)."""
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return False, f"too short: {len(stripped)} chars < {MIN_CHARS}"

    words = _words(stripped)
    if len(words) < MIN_WORDS:
        return False, f"too few words: {len(words)} < {MIN_WORDS}"

    # symbol ratio: combining marks (Indic vowel-signs/virama) are word content,
    # not symbols, so classify via the shared script-aware rule -- otherwise
    # ordinary Devanagari/Bengali/Tamil text reads as majority-"symbol".
    non_space = [c for c in stripped if not c.isspace()]
    if non_space:
        symbols = sum(1 for c in non_space if not is_word_char(c))
        ratio = symbols / len(non_space)
        if ratio > MAX_SYMBOL_RATIO:
            return False, f"symbol-heavy: {ratio:.2f} > {MAX_SYMBOL_RATIO}"

    # single-token repetition: e.g. "buy buy buy buy ..."
    counts: dict[str, int] = {}
    for w in words:
        key = w.lower()
        counts[key] = counts.get(key, 0) + 1
    top = max(counts.values())
    rep = top / len(words)
    if rep > MAX_WORD_REPETITION:
        return False, f"repetitive: one token is {rep:.2f} of the text > {MAX_WORD_REPETITION}"

    return True, None


def filter_quality(
    docs: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep passing docs; drop the rest recording {id, stage:"quality", reason}."""
    kept: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    for doc in docs:
        ok, reason = quality_ok(doc["text"])
        if ok:
            kept.append(doc)
        else:
            drops.append({"id": doc["id"], "stage": "quality", "reason": reason})
    return kept, drops
