"""Epic 1.8 tests. Run with: pytest -q"""

from __future__ import annotations

import dataclasses

import pytest

from contrastive_pairs import CONTRASTIVE_PAIRS, validate_all
from corpus_schema import CHAUVINISM, VANTAGE


def test_corpus_is_large_enough() -> None:
    assert len(CONTRASTIVE_PAIRS) >= 30


def test_validate_all_passes() -> None:
    assert validate_all(CONTRASTIVE_PAIRS) == CONTRASTIVE_PAIRS


def test_ids_are_unique_and_well_formed() -> None:
    ids = [pair.id for pair in CONTRASTIVE_PAIRS]
    assert len(ids) == len(set(ids))
    assert ids == [f"pair-{i:04d}" for i in range(1, len(ids) + 1)]


def test_every_pair_declares_the_expected_vantage() -> None:
    assert all(pair.vantage == VANTAGE for pair in CONTRASTIVE_PAIRS)


def test_no_pair_is_chauvinistic() -> None:
    assert all(pair.chauvinism == CHAUVINISM for pair in CONTRASTIVE_PAIRS)


def test_continuations_differ_in_every_pair() -> None:
    for pair in CONTRASTIVE_PAIRS:
        assert pair.y_plus != pair.y_minus, pair.id
        assert pair.y_plus.strip() != pair.y_minus.strip(), pair.id


def test_topics_are_distinct() -> None:
    """The spec asks for distinct contested topics, not variations on one."""
    topics = [pair.topic for pair in CONTRASTIVE_PAIRS]
    assert len(topics) == len(set(topics))


def test_prefix_is_shared_and_non_trivial() -> None:
    """One prefix per pair, carrying enough context to make the fork meaningful."""
    for pair in CONTRASTIVE_PAIRS:
        assert pair.prefix.strip(), pair.id
        assert len(pair.prefix.split()) >= 3, pair.id


def test_duplicate_id_is_rejected() -> None:
    first = CONTRASTIVE_PAIRS[0]
    clashing = dataclasses.replace(CONTRASTIVE_PAIRS[1], id=first.id)
    with pytest.raises(ValueError, match="duplicate id"):
        validate_all((first, clashing))


def test_invalid_pair_is_rejected() -> None:
    broken = dataclasses.replace(CONTRASTIVE_PAIRS[0], chauvinism="mild")
    with pytest.raises(ValueError, match="chauvinism"):
        validate_all((broken,))


def test_identical_continuations_are_rejected() -> None:
    same = dataclasses.replace(CONTRASTIVE_PAIRS[0], y_minus=CONTRASTIVE_PAIRS[0].y_plus)
    with pytest.raises(ValueError, match="identical"):
        validate_all((same,))


def test_default_argument_validates_the_shipped_corpus() -> None:
    assert validate_all() == CONTRASTIVE_PAIRS
