"""Epic 2.1 — canonical text normalization. Pure, deterministic, idempotent.

Unicode NFC, drop control characters, normalise newlines, collapse runs of
spaces/tabs to a single space (tabs become spaces; only \\n is kept as
structure), strip per-line trailing spaces and overall ends.

Deliberate choice for an India-first corpus: only category "Cc" (control) is
dropped. Format characters (category "Cf") such as the zero-width joiner
U+200D are KEPT, because they carry meaning in Devanagari, Bengali, Tamil and
other Indic scripts, and dropping them would silently corrupt the text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from corpus_schema import Document

__all__ = ["normalize_text", "normalize_document"]

_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Return the canonical form of `text`. normalize_text(normalize_text(x)) == normalize_text(x)."""
    text = unicodedata.normalize("NFC", text)
    # normalise newlines first so the control-char filter sees only \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # drop control chars (Cc) except tab/newline; keep format chars (Cf) for Indic scripts
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc"
    )
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip(" ") for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def normalize_document(doc: Document) -> Document:
    """A Document with normalised text; every other field unchanged."""
    return replace(doc, text=normalize_text(doc.text))
