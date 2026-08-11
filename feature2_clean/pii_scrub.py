"""Epic 2.5 — PII scrub. Deterministic, idempotent, conservative.

Redacts email addresses and phone numbers to fixed placeholders. Patterns are
deliberately narrow to avoid over-redaction: the placeholders contain no digits
or "@", so scrubbing an already-scrubbed text changes nothing more.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from feature1_collect.corpus_schema import Document

__all__ = ["EMAIL_PLACEHOLDER", "PHONE_PLACEHOLDER", "scrub_pii", "scrub_document"]

EMAIL_PLACEHOLDER = "[EMAIL]"
PHONE_PLACEHOLDER = "[PHONE]"

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Phone numbers are redacted only when they carry an explicit dialing marker:
#   * a leading "+" (international format), or
#   * a parenthesized area/trunk code "(...)".
# Bare digit runs -- version strings, IDs, dates, and numeric data that fill the
# code/math/web lanes -- are deliberately LEFT ALONE, because treating every
# 7-digit sequence as a phone number destroys real content. The separator class
# excludes newlines (a phone lives on one line), and the 7-digit floor is still
# enforced in the callback so short groups like a year (2024) are ignored.
_SEP = r"[\d \t().\-]"
_PHONE = re.compile(
    r"(?<!\w)(?:"
    r"\+\d" + _SEP + r"{5,}\d"           # +CC then a single-line digit run
    r"|\(\d{2,4}\)[ \t]?" + _SEP + r"{4,}\d"  # (area code) then more digits
    r")(?!\w)"
)


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
