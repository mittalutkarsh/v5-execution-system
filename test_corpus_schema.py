"""Epic 1.1 tests. Run with: pytest -q"""

from __future__ import annotations

import dataclasses

import pytest

from corpus_schema import (
    EXAMPLE_DOCUMENTS,
    EXAMPLE_PAIRS,
    EXAMPLES,
    ContrastivePair,
    Document,
    validate,
    validate_contrastive,
    validate_document,
)


def test_examples_validate() -> None:
    """Every shipped example passes its own validator."""
    assert len(EXAMPLE_DOCUMENTS) == 2
    assert len(EXAMPLE_PAIRS) == 1
    assert len(EXAMPLES) == 3
    for record in EXAMPLES:
        assert validate(record) is record


def test_bad_lane_raises() -> None:
    """A lane outside the closed vocabulary is rejected."""
    doc = Document(
        id="doc-bad-lane",
        lane="tweets",          # not in LANES
        provenance_tier="T1",
        split="train",
        source="somewhere",
        text="some text",
    )
    with pytest.raises(ValueError, match="lane"):
        validate_document(doc)


def test_chauvinism_not_none_raises() -> None:
    """chauvinism must be exactly "none"."""
    pair = ContrastivePair(
        id="pair-bad-chauvinism",
        topic="financial year",
        prefix="For a company registered in Mumbai, the financial year begins in",
        y_plus="April and closes on 31 March.",
        y_minus="January and closes on 31 December.",
        vantage="indian_plus_western_minus",
        chauvinism="mild",      # must be "none"
    )
    with pytest.raises(ValueError, match="chauvinism"):
        validate_contrastive(pair)


def test_records_are_frozen() -> None:
    """Records are facts about the data, not mutable buffers."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        EXAMPLE_DOCUMENTS[0].lane = "code"  # type: ignore[misc]
