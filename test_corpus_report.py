"""Epic 1.11 tests. Fully offline. Run with: pytest -q"""

from __future__ import annotations

import json

import pytest

from corpus_loader import corpus_counts
from corpus_report import (
    REPORT_KIND,
    REPORT_VERSION,
    build_summary,
    load_corpus_summary,
    write_corpus_summary,
)
from corpus_schema import LANES
from sources_manifest import SOURCES

WEB_ID = "web-fineweb"      # T2, lane "web"
WIKI_ID = "indic-wiki-hi"   # T1, lane "indic" -- the only eval-eligible one here
N_WEB, N_WIKI = 5, 6
EVAL_INDICES = (1, 3)
JSON_KW = dict(sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture
def fixture_sources():
    return tuple(s for s in SOURCES if s.source_id in (WEB_ID, WIKI_ID))


def doc_text(source_id: str, i: int) -> str:
    return f"{source_id} document {i:03d}. " + "body text for the corpus. " * (i % 3 + 1)


def build_corpus(tmp_path, sources):
    """A data/raw plus data/eval laid out the way 1.3 and 1.9 leave them."""
    raw, evl = tmp_path / "raw", tmp_path / "eval"
    counts = {WEB_ID: N_WEB, WIKI_ID: N_WIKI}

    for source in sources:
        directory = raw / source.source_id
        directory.mkdir(parents=True)
        with (directory / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for i in range(counts[source.source_id]):
                fh.write(json.dumps({
                    "id": f"{source.source_id}-{i:07d}",
                    "lane": source.lane,
                    "provenance_tier": source.provenance_tier,
                    "split": "train",
                    "source": f"{source.dataset}@pinned-test",
                    "text": doc_text(source.source_id, i),
                }, **JSON_KW) + "\n")

    evl.mkdir(parents=True)
    with (evl / "eval_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({
            "kind": "header", "seed": "v5-eval-2026", "target_fraction": 0.015,
            "selected_count": len(EVAL_INDICES), "fingerprint": "deadbeef",
        }, **JSON_KW) + "\n")
        for i in EVAL_INDICES:
            fh.write(json.dumps({
                "id": f"{WIKI_ID}-{i:07d}", "source_id": WIKI_ID, "lane": "indic",
                "provenance_tier": "T1", "split": "eval", "est_tokens": 10,
            }, **JSON_KW) + "\n")
    return raw, evl


@pytest.fixture
def corpus(tmp_path, fixture_sources):
    raw, evl = build_corpus(tmp_path, fixture_sources)
    return {"raw_root": str(raw), "eval_root": str(evl), "sources": fixture_sources}


# --------------------------------------------------------------------------
# build_summary
# --------------------------------------------------------------------------


def test_report_is_labelled(corpus) -> None:
    report = build_summary(**corpus)
    assert report["kind"] == REPORT_KIND
    assert report["version"] == REPORT_VERSION


def test_totals_match_corpus_counts(corpus) -> None:
    report = build_summary(**corpus)
    assert report["totals"] == corpus_counts(**corpus)["totals"]


def test_lanes_cover_every_lane(corpus) -> None:
    report = build_summary(**corpus)
    assert set(report["lanes"]) == set(LANES)
    counts = corpus_counts(**corpus)
    for lane in LANES:
        assert report["lanes"][lane] == counts[lane]


def test_contrastive_matches_corpus_counts(corpus) -> None:
    report = build_summary(**corpus)
    assert report["contrastive"] == corpus_counts(**corpus)["contrastive"]
    assert report["contrastive"]["pairs"] >= 30


def test_train_and_eval_partition_the_totals(corpus) -> None:
    report = build_summary(**corpus)
    totals, derived = report["totals"], report["derived"]
    assert derived["train_docs"] + totals["eval_docs"] == totals["docs"]
    assert derived["train_tokens"] + totals["eval_tokens"] == totals["tokens"]


def test_eval_token_fraction_is_rounded_correctly(corpus) -> None:
    report = build_summary(**corpus)
    totals = report["totals"]
    assert report["derived"]["eval_token_fraction"] == round(
        totals["eval_tokens"] / totals["tokens"], 6
    )


def test_lane_shares_sum_to_one(corpus) -> None:
    report = build_summary(**corpus)
    shares = report["derived"]["lane_train_token_share"]
    assert set(shares) == set(LANES)
    # five values each rounded to 6 places -> at most 2.5e-6 of drift
    assert abs(sum(shares.values()) - 1.0) < 1e-5


def test_lane_shares_track_lane_train_tokens(corpus) -> None:
    report = build_summary(**corpus)
    train_tokens = report["derived"]["train_tokens"]
    for lane, share in report["derived"]["lane_train_token_share"].items():
        expected = round(report["lanes"][lane]["train_tokens"] / train_tokens, 6)
        assert share == expected


def test_empty_corpus_does_not_divide_by_zero(tmp_path, fixture_sources) -> None:
    """No raw dirs at all: every ratio is 0.0 rather than an exception."""
    report = build_summary(
        raw_root=str(tmp_path / "absent"),
        eval_root=str(tmp_path / "absent"),
        sources=fixture_sources,
    )
    assert report["totals"]["tokens"] == 0
    assert report["derived"]["eval_token_fraction"] == 0.0
    assert set(report["derived"]["lane_train_token_share"]) == set(LANES)
    assert all(v == 0.0 for v in report["derived"]["lane_train_token_share"].values())


# --------------------------------------------------------------------------
# write / load
# --------------------------------------------------------------------------


def test_written_file_round_trips(tmp_path, corpus) -> None:
    out = tmp_path / "out" / "corpus_summary.json"
    returned = write_corpus_summary(**corpus, out_path=str(out))
    assert out.exists()
    assert load_corpus_summary(str(out)) == returned
    assert load_corpus_summary(str(out)) == build_summary(**corpus)


def test_written_file_is_valid_json_ending_in_a_newline(tmp_path, corpus) -> None:
    out = tmp_path / "corpus_summary.json"
    write_corpus_summary(**corpus, out_path=str(out))
    raw = out.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw)["kind"] == REPORT_KIND
    assert "\r" not in raw, "CRLF would break byte-identity across platforms"


def test_parent_directories_are_created(tmp_path, corpus) -> None:
    out = tmp_path / "deep" / "nested" / "corpus_summary.json"
    write_corpus_summary(**corpus, out_path=str(out))
    assert out.exists()


def test_regeneration_is_byte_identical(tmp_path, corpus) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_corpus_summary(**corpus, out_path=str(a))
    write_corpus_summary(**corpus, out_path=str(b))
    assert a.read_bytes() == b.read_bytes()


def test_rewriting_the_same_path_leaves_it_unchanged(tmp_path, corpus) -> None:
    out = tmp_path / "corpus_summary.json"
    write_corpus_summary(**corpus, out_path=str(out))
    before = out.read_bytes()
    write_corpus_summary(**corpus, out_path=str(out))
    assert out.read_bytes() == before


def test_report_carries_no_timestamp(tmp_path, corpus) -> None:
    """A generation time would make every rerun a diff."""
    out = tmp_path / "corpus_summary.json"
    write_corpus_summary(**corpus, out_path=str(out))
    text = out.read_text(encoding="utf-8").lower()
    for volatile in ("timestamp", "generated_at", "created", "hostname", "run_id"):
        assert volatile not in text
