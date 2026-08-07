"""Epic 1.1 — the corpus data model, and nothing else.

Two record types, their validators, and three literal examples. No file I/O,
no network, no tokenizer, no runner. Importing this module touches nothing
outside itself, so it is deterministic by construction.

The `source` field is a provenance LABEL -- a string recording where a
document is claimed to have come from. Nothing here resolves it, downloads
it, or opens it. Acquisition is a later micro-step; this file only says what
a record looks like once something else has produced one.

Design note: construction stays permissive and validation is a separate call.
A corpus loader wants to read a bad row, report it by id, and keep going --
not die inside a dataclass constructor. So `Document(...)` never raises on a
bad lane; `validate_document(...)` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "Document",
    "ContrastivePair",
    "Record",
    "LANES",
    "PROVENANCE_TIERS",
    "SPLITS",
    "VANTAGE",
    "CHAUVINISM",
    "validate_document",
    "validate_contrastive",
    "validate",
    "EXAMPLE_DOCUMENTS",
    "EXAMPLE_PAIRS",
    "EXAMPLES",
]

# --------------------------------------------------------------------------
# Closed vocabularies. frozenset so they cannot be mutated by an importer;
# error messages sort them so the text is stable across runs.
# --------------------------------------------------------------------------

LANES: Final[frozenset[str]] = frozenset(
    {"web", "code", "math", "indic", "multilingual"}
)
# PROVENANCE_TIERS -- trust grades. v1 convention, expected to change.
#
# The tier is a judgement about a SOURCE, not about a document. You cannot
# read a billion documents; you can decide once per acquisition route and
# stamp every document that route produces. The axis below is rights first,
# then review, then filtering:
#
#   T0  Rights explicit and training permitted. Origin identifiable per
#       document. A human reviewed a sample. Eval-eligible.
#   T1  Published dataset with a stated licence and a documented cleaning
#       pipeline. Origin identifiable. Not reviewed by us. Eval-eligible.
#   T2  Open crawl of permissive-by-default sources, deduped, quality-
#       filtered, decontaminated against eval. Rights plausible but not
#       confirmed per document. Train only.
#   T3  Everything else: rights unclear, or dedup/decontamination missing,
#       or quality checks failed. Train only, and usually dropped from late
#       curriculum phases.
#
# Three rules that go with it:
#   1. A source's tier is a CEILING. Per-document checks can demote below
#      it; nothing can promote above it, because promotion needs rights
#      knowledge and human review, which no text classifier recovers.
#   2. Failing eval decontamination drops a document. It does not demote it.
#   3. split="eval" requires T0 or T1.
#
# Of these, rule 3 IS enforced here (validate_document: eval requires T0/T1) --
# it needs only the document's own split and tier, no registry. Rules 1 (tier
# ceiling) and 2 (decontamination drop) need a source registry and are enforced
# in a later step. Otherwise this file only checks that each string is one of
# the four labels; the label set is what other code depends on and is not
# expected to change.
PROVENANCE_TIERS: Final[frozenset[str]] = frozenset({"T0", "T1", "T2", "T3"})
SPLITS: Final[frozenset[str]] = frozenset({"train", "eval"})

VANTAGE: Final[str] = "indian_plus_western_minus"
CHAUVINISM: Final[str] = "none"


# --------------------------------------------------------------------------
# Records
#
# frozen   : a corpus record is a fact about the data, not a mutable buffer
# slots    : there will be millions of these, so skip the per-instance dict
# kw_only  : Document is six strings in a row -- positional construction would
#            silently accept lane and provenance_tier swapped
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Document:
    """One unit of pretraining text, tagged with where it came from."""

    id: str
    lane: str              # one of LANES
    provenance_tier: str   # one of PROVENANCE_TIERS
    split: str             # one of SPLITS
    source: str            # a provenance LABEL, not a URL or path; never fetched
    text: str              # the document body, already in memory


@dataclass(frozen=True, slots=True, kw_only=True)
class ContrastivePair:
    """One prefix with two continuations: Indian vantage vs Western default.

    y_plus is the preferred continuation, y_minus the default a
    Western-centric model tends to emit. Neither disparages anyone -- the
    contrast is about vantage point, which is what `chauvinism="none"`
    asserts and what validation enforces.
    """

    id: str
    topic: str
    prefix: str
    y_plus: str    # Indian-vantage continuation
    y_minus: str   # Western-default continuation
    vantage: str   # must equal VANTAGE
    chauvinism: str  # must equal CHAUVINISM


Record = Document | ContrastivePair


# --------------------------------------------------------------------------
# Field checks. Every failure is a ValueError, including a wrong type, so a
# caller has exactly one exception to catch per record.
# --------------------------------------------------------------------------


def _require_str(value: object, field: str, *, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{where}: {field} must be a str, got {type(value).__name__}"
        )
    if not value.strip():
        raise ValueError(f"{where}: {field} must be a non-empty string")
    return value


def _require_member(
    value: object, field: str, allowed: frozenset[str], *, where: str
) -> str:
    text = _require_str(value, field, where=where)
    if text not in allowed:
        raise ValueError(
            f"{where}: {field} must be one of {sorted(allowed)}, got {text!r}"
        )
    return text


def _require_exact(value: object, field: str, expected: str, *, where: str) -> str:
    text = _require_str(value, field, where=where)
    if text != expected:
        raise ValueError(
            f"{where}: {field} must be {expected!r}, got {text!r}"
        )
    return text


# --------------------------------------------------------------------------
# Validators. Each returns the record it was given, so a construction and its
# check can be written on one line.
# --------------------------------------------------------------------------


def validate_document(doc: Document) -> Document:
    """Raise ValueError on any invalid field, else return `doc` unchanged."""
    if not isinstance(doc, Document):
        raise ValueError(f"expected a Document, got {type(doc).__name__}")

    # id first, so every later message can name the offending record
    doc_id = _require_str(doc.id, "id", where="Document")
    where = f"Document {doc_id!r}"

    _require_member(doc.lane, "lane", LANES, where=where)
    _require_member(doc.provenance_tier, "provenance_tier", PROVENANCE_TIERS, where=where)
    _require_member(doc.split, "split", SPLITS, where=where)
    _require_str(doc.source, "source", where=where)
    _require_str(doc.text, "text", where=where)

    # Rule 3: eval data must be high-trust (see PROVENANCE_TIERS). This is a
    # per-document invariant -- it reads only this document's own split and
    # tier, so it belongs here rather than in a later registry step.
    if doc.split == "eval" and doc.provenance_tier not in {"T0", "T1"}:
        raise ValueError(
            f"{where}: split='eval' requires provenance_tier T0 or T1, "
            f"got {doc.provenance_tier!r}"
        )
    return doc


def validate_contrastive(pair: ContrastivePair) -> ContrastivePair:
    """Raise ValueError on any invalid field, else return `pair` unchanged."""
    if not isinstance(pair, ContrastivePair):
        raise ValueError(f"expected a ContrastivePair, got {type(pair).__name__}")

    pair_id = _require_str(pair.id, "id", where="ContrastivePair")
    where = f"ContrastivePair {pair_id!r}"

    _require_str(pair.topic, "topic", where=where)
    _require_str(pair.prefix, "prefix", where=where)
    _require_str(pair.y_plus, "y_plus", where=where)
    _require_str(pair.y_minus, "y_minus", where=where)
    _require_exact(pair.vantage, "vantage", VANTAGE, where=where)
    _require_exact(pair.chauvinism, "chauvinism", CHAUVINISM, where=where)

    if pair.y_plus.strip() == pair.y_minus.strip():
        raise ValueError(
            f"{where}: y_plus and y_minus are identical, so the pair "
            f"carries no contrast"
        )
    return pair


def validate(record: Record) -> Record:
    """Dispatch to the right validator by record type."""
    if isinstance(record, Document):
        return validate_document(record)
    if isinstance(record, ContrastivePair):
        return validate_contrastive(record)
    raise ValueError(f"unknown record type: {type(record).__name__}")


# --------------------------------------------------------------------------
# Examples. Hand-written literals, NOT samples of any real corpus. The text
# below was composed for this file; `source` is set to "fixture/handwritten"
# so nothing here can be mistaken for provenance. A real row would carry the
# actual origin string, e.g. "commoncrawl/CC-MAIN-2024-33" or
# "ai4bharat/sangraha:hi" -- but note that `source` is only ever a LABEL.
# Nothing in this module resolves it, fetches it, or opens it.
#
# The tier values below are chosen to exercise two different enum members and
# to satisfy rule 3 (eval requires T0/T1). They are not a demonstration of
# tiering policy -- a handwritten fixture is not an acquisition route, so no
# tier really applies to it.
# --------------------------------------------------------------------------

DOC_WEB: Final[Document] = Document(
    id="doc-web-000001",
    lane="web",
    provenance_tier="T2",
    split="train",
    source="fixture/handwritten",
    text=(
        "The southwest monsoon usually reaches the Kerala coast around 1 June, "
        "but the India Meteorological Department declares onset only once "
        "rainfall, wind field and satellite thresholds are met together."
    ),
)

DOC_MATH: Final[Document] = Document(
    id="doc-math-000001",
    lane="math",
    provenance_tier="T0",
    split="eval",
    source="fixture/handwritten",
    text=(
        "Problem. Show that among any 8 integers, two have a difference "
        "divisible by 7. Sketch: the residues mod 7 give 7 pigeonholes for "
        "8 pigeons, so some residue repeats."
    ),
)

PAIR_FINANCIAL_YEAR: Final[ContrastivePair] = ContrastivePair(
    id="pair-fiscal-000001",
    topic="financial year",
    prefix="For a company registered in Mumbai, the financial year begins in",
    y_plus=(
        "April and closes on 31 March, the period the Companies Act, 2013 "
        "fixes for Indian companies."
    ),
    y_minus=(
        "January and closes on 31 December, following the calendar year."
    ),
    vantage=VANTAGE,
    chauvinism=CHAUVINISM,
)

EXAMPLE_DOCUMENTS: Final[tuple[Document, ...]] = (DOC_WEB, DOC_MATH)
EXAMPLE_PAIRS: Final[tuple[ContrastivePair, ...]] = (PAIR_FINANCIAL_YEAR,)
EXAMPLES: Final[tuple[Record, ...]] = (*EXAMPLE_DOCUMENTS, *EXAMPLE_PAIRS)
