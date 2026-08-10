"""Epic 2.5 — PII scrub. Deterministic, idempotent, conservative.

Redacts email addresses and phone numbers to fixed placeholders. Patterns are
deliberately narrow to avoid over-redaction: the placeholders contain no digits
or "@", so scrubbing an already-scrubbed text changes nothing more.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from corpus_schema import Document

__all__ = ["EMAIL_PLACEHOLDER", "PHONE_PLACEHOLDER", "scrub_pii", "scrub_document"]

EMAIL_PLACEHOLDER = "[EMAIL]"
PHONE_PLACEHOLDER = "[PHONE]"

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# A run of digits with separators (spaces, tabs, dots, dashes, parens),
# optionally a leading +. The separator class deliberately EXCLUDES newlines: a
# phone number lives on one line, and matching across "\n" would swallow whole
# multi-line number columns (equations, data tables) in the math/code lanes. The
# 7-digit floor is enforced in the callback, so short numbers like a year (2024)
# or a chapter (7) are left alone.
_PHONE = re.compile(r"(?<!\w)\+?\d[\d \t().\-]{5,}\d(?!\w)")


def scrub_pii(text: str) -> tuple[str, int]:
    """Return (scrubbed_text, n_redactions). Idempotent."""
    n = 0

    def _email(_m: re.Match) -> str:
        nonlocal n
        n += 1
        return EMAIL_PLACEHOLDER

    def _phone(m: re.Match) -> str:
        nonlocal n
        # require at least 7 digits so short number groups are not redacted
        if sum(c.isdigit() for c in m.group()) < 7:
            return m.group()
        n += 1
        return PHONE_PLACEHOLDER

    text = _EMAIL.sub(_email, text)
    text = _PHONE.sub(_phone, text)
    return text, n


def scrub_document(doc: Document) -> tuple[Document, int]:
    """A Document with scrubbed text, and the number of redactions made."""
    scrubbed, n = scrub_pii(doc.text)
    return replace(doc, text=scrubbed), n
