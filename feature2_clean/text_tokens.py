"""Shared, script-aware word tokenization for the cleaning stages.

Python's ``\\w`` (even with re.UNICODE) treats Unicode combining marks
(categories Mn/Mc/Me) as NON-word characters. For Indic scripts that is
destructive: a Bengali or Devanagari syllable is a base consonant plus
vowel-sign / virama marks, so ``\\w+`` shreds "বাংলা" into single-consonant
fragments (~1.3 chars each). Every downstream count computed on those fragments
-- word length, n-gram overlap, MinHash shingles, symbol ratio -- is then
garbage for exactly the India-first lanes we care most about.

A word here is a maximal run of letters, digits and their combining marks
(Unicode categories L*, N*, M*), plus the zero-width joiner that binds Indic
conjuncts. Punctuation, symbols and whitespace separate words. On pure-ASCII /
Latin text this is identical to ``\\w+`` (there are no marks to reattach), so
the English/code/math lanes are unaffected.
"""

from __future__ import annotations

import re

__all__ = ["WORD_RE", "words", "is_word_char"]

# Combining marks that ``\w`` omits. Covers the scripts present in the corpus
# (Devanagari, Bengali, Tamil) plus generic Latin/combining diacritics. To add a
# script, append its mark ranges here.
_MARKS = (
    "̀-ͯ"                                        # combining diacritical marks (general)
    "ऀ-ःऺ-ॏ॑-ॗॢॣ"  # Devanagari
    "ঁ-ঃ়া-্ৗৢৣ"    # Bengali
    "ஂா-்ௗ"                             # Tamil
    "‍"                                               # zero-width joiner (Indic conjuncts)
)
WORD_RE = re.compile(r"[\w" + _MARKS + r"]+", re.UNICODE)


def words(text: str) -> list[str]:
    """The words of `text`: runs of letters/digits and their combining marks."""
    return WORD_RE.findall(text)


def is_word_char(ch: str) -> bool:
    """True for a letter, digit, combining mark or ZWJ; False for punctuation/symbols."""
    return WORD_RE.fullmatch(ch) is not None
