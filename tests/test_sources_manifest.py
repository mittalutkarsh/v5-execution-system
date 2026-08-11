"""Epic 1.2 tests. Run with: pytest -q"""

from __future__ import annotations

import dataclasses

import pytest

from feature1_collect.corpus_schema import LANES
from feature1_collect.sources_manifest import (
    POOL_TARGET_TOKENS,
    POOL_TOLERANCE,
    SOURCES,
    eval_eligible,
    lane_totals,
    validate_sources,
)

T1_WIKI_IDS = {
    "indic-wiki-hi",
    "indic-wiki-bn",
    "indic-wiki-ta",
    "mling-wiki-es",
    "mling-wiki-fr",
}
T2_IDS = {"web-fineweb", "code-github", "math-openwebmath"}


def test_manifest_validates() -> None:
    assert validate_sources(SOURCES) == SOURCES


def test_total_within_tolerance() -> None:
    total = sum(s.target_tokens for s in SOURCES)
    assert abs(total - POOL_TARGET_TOKENS) <= POOL_TOLERANCE * POOL_TARGET_TOKENS
    assert total == 10_000_000  # exact, as declared


def test_lane_totals_cover_all_five_lanes() -> None:
    totals = lane_totals(SOURCES)
    assert set(totals) == set(LANES)
    assert totals == {
        "code": 2_000_000,
        "indic": 2_200_000,
        "math": 1_200_000,
        "multilingual": 600_000,
        "web": 4_000_000,
    }
    assert sum(totals.values()) == POOL_TARGET_TOKENS


def test_source_ids_unique() -> None:
    ids = [s.source_id for s in SOURCES]
    assert len(ids) == len(set(ids)) == 8


def test_every_revision_unpinned() -> None:
    """1.2 declares; 1.3 pins the snapshot."""
    assert all(s.revision == "" for s in SOURCES)


def test_eval_eligible_is_t1_wikipedia_only() -> None:
    got = eval_eligible(SOURCES)
    assert {s.source_id for s in got} == T1_WIKI_IDS
    assert all(s.provenance_tier == "T1" for s in got)
    assert all(s.dataset == "wikimedia/wikipedia" for s in got)
    # the T2 web/code/math sources are excluded
    excluded = {s.source_id for s in SOURCES} - {s.source_id for s in got}
    assert excluded == T2_IDS


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("license", "", "license"),
        ("license", "   ", "license"),
        ("target_tokens", 0, "target_tokens"),
        ("target_tokens", -1, "target_tokens"),
        ("gated", True, "gated"),
        ("provenance_tier", "T9", "provenance_tier"),
        ("lane", "tweets", "lane"),
        ("dataset", "", "dataset"),
        ("source_id", "", "source_id"),
    ],
)
def test_bad_field_raises(field: str, value: object, match: str) -> None:
    broken = (dataclasses.replace(SOURCES[0], **{field: value}),) + SOURCES[1:]
    with pytest.raises(ValueError, match=match):
        validate_sources(broken)


def test_duplicate_source_id_raises() -> None:
    dupe = dataclasses.replace(SOURCES[1], source_id=SOURCES[0].source_id)
    broken = (SOURCES[0], dupe) + SOURCES[2:]
    with pytest.raises(ValueError, match="duplicate"):
        validate_sources(broken)


def test_missing_lane_raises() -> None:
    broken = tuple(s for s in SOURCES if s.lane != "code")
    with pytest.raises(ValueError, match="lanes with no source"):
        validate_sources(broken)


def test_total_outside_tolerance_raises() -> None:
    shrunk = dataclasses.replace(SOURCES[0], target_tokens=1)
    with pytest.raises(ValueError, match="tolerance"):
        validate_sources((shrunk,) + SOURCES[1:])


def test_empty_manifest_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_sources(())


# --------------------------------------------------------------------------
# Epics 1.4-1.7: the upstream text column
# --------------------------------------------------------------------------


def test_every_source_declares_a_text_field() -> None:
    assert all(s.text_field.strip() for s in SOURCES)


def test_code_github_reads_the_code_column() -> None:
    """The code source (CodeSearchNet) puts the source under "code"."""
    code = next(s for s in SOURCES if s.source_id == "code-github")
    assert code.text_field == "code"


def test_every_other_source_reads_text() -> None:
    others = [s for s in SOURCES if s.source_id != "code-github"]
    assert others, "expected sources besides code-github"
    assert all(s.text_field == "text" for s in others)


def test_empty_text_field_raises() -> None:
    broken = (dataclasses.replace(SOURCES[0], text_field=""),) + SOURCES[1:]
    with pytest.raises(ValueError, match="text_field"):
        validate_sources(broken)


def test_whitespace_text_field_raises() -> None:
    broken = (dataclasses.replace(SOURCES[0], text_field="   "),) + SOURCES[1:]
    with pytest.raises(ValueError, match="text_field"):
        validate_sources(broken)
