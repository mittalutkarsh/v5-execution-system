"""Epic 1.3 tests. Fully offline -- neither `datasets` nor `huggingface_hub`
is imported, installed, or reachable. Run with: pytest -q
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys

import pytest

from corpus_schema import Document, validate_document
from fetch import estimate_tokens, fetch_source
from sources_manifest import SOURCES

TARGET = 200
N_DOCS = 50


@pytest.fixture
def source():
    """web-fineweb, but pinned and with a target small enough to cap early."""
    web = next(s for s in SOURCES if s.source_id == "web-fineweb")
    return dataclasses.replace(web, target_tokens=TARGET, revision="pinned-test")


def fake_docs(n: int = N_DOCS) -> list[dict[str, str]]:
    """Deterministic stand-in for a streamed dataset."""
    return [
        {"text": f"Document {i:02d}. The quick brown fox jumps over the lazy dog."}
        for i in range(n)
    ]


def exploding_iter():
    """A stream that fails loudly if anything touches it."""
    raise AssertionError("doc_iter was consumed on a cache hit")
    yield  # pragma: no cover


def read_docs(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------
# estimate_tokens
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "a", "abcd", "hello world", "x" * 999, "नमस्ते दुनिया", "🙂🙂🙂"],
)
def test_estimate_tokens_matches_bytes_over_four(text: str) -> None:
    assert estimate_tokens(text) == max(1, len(text.encode("utf-8")) // 4)


def test_estimate_tokens_counts_bytes_not_characters() -> None:
    """Devanagari costs more per character than ASCII, and should."""
    assert estimate_tokens("नमस्ते") > estimate_tokens("namaste")


def test_estimate_tokens_is_monotonic() -> None:
    previous = 0
    for length in range(0, 400, 7):
        current = estimate_tokens("x" * length)
        assert current >= previous
        previous = current


def test_estimate_tokens_never_zero() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a") == 1


# --------------------------------------------------------------------------
# capping
# --------------------------------------------------------------------------


def test_fetch_caps_at_target(tmp_path, source) -> None:
    summary = fetch_source(source, out_root=str(tmp_path), doc_iter=fake_docs())

    assert summary["cached"] is False
    assert summary["est_tokens"] >= TARGET, "should stop at or past the target"
    assert summary["doc_count"] < N_DOCS, "should not consume the whole stream"

    # and it stopped as soon as it crossed, not later
    docs = read_docs(summary["path"])
    assert len(docs) == summary["doc_count"]
    before_last = sum(estimate_tokens(d["text"]) for d in docs[:-1])
    assert before_last < TARGET


def test_fetch_writes_one_json_object_per_line(tmp_path, source) -> None:
    summary = fetch_source(source, out_root=str(tmp_path), doc_iter=fake_docs())
    raw = open(summary["path"], "r", encoding="utf-8").read()
    assert raw.endswith("\n")
    assert raw.count("\n") == summary["doc_count"]


def test_accepts_plain_strings_too(tmp_path, source) -> None:
    texts = [d["text"] for d in fake_docs()]
    summary = fetch_source(source, out_root=str(tmp_path), doc_iter=texts)
    assert summary["doc_count"] > 0
    assert summary["est_tokens"] >= TARGET


def test_blank_texts_are_skipped_not_fatal(tmp_path, source) -> None:
    stream = [{"text": ""}, {"text": "   "}] + fake_docs()
    summary = fetch_source(source, out_root=str(tmp_path), doc_iter=stream)
    assert summary["skipped"] == 2
    assert summary["doc_count"] > 0


# --------------------------------------------------------------------------
# every line rebuilds into a valid Document
# --------------------------------------------------------------------------


def test_every_line_is_a_valid_document(tmp_path, source) -> None:
    summary = fetch_source(source, out_root=str(tmp_path), doc_iter=fake_docs())
    for row in read_docs(summary["path"]):
        document = Document(**row)
        assert validate_document(document) is document
        assert document.lane == source.lane
        assert document.provenance_tier == source.provenance_tier
        assert document.split == "train"
        assert document.source == f"{source.dataset}@pinned-test"
        assert document.id.startswith("web-fineweb-")


# --------------------------------------------------------------------------
# the log
# --------------------------------------------------------------------------


def test_log_sha256_and_bytes_match_recompute(tmp_path, source) -> None:
    summary = fetch_source(source, out_root=str(tmp_path), doc_iter=fake_docs())

    log_lines = [
        json.loads(line)
        for line in open(tmp_path / "fetch_log.jsonl", encoding="utf-8")
        if line.strip()
    ]
    assert len(log_lines) == 1
    record = log_lines[0]

    payload = open(summary["path"], "rb").read()
    assert record["bytes"] == len(payload)
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["sha256"] == summary["sha256"]
    assert record["revision"] == "pinned-test"
    assert record["target_tokens"] == TARGET
    assert record["dataset"] == source.dataset
    assert record["config"] == source.config


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------


def test_second_call_is_a_cached_noop(tmp_path, source) -> None:
    first = fetch_source(source, out_root=str(tmp_path), doc_iter=fake_docs())
    before = open(first["path"], "rb").read()

    # the exploding iterator proves the stream is never touched
    second = fetch_source(source, out_root=str(tmp_path), doc_iter=exploding_iter())

    assert second["cached"] is True
    assert second["sha256"] == first["sha256"]
    assert second["bytes"] == first["bytes"]
    assert second["doc_count"] == first["doc_count"]
    assert open(first["path"], "rb").read() == before, "file must be untouched"

    # and no second log line was appended
    log = [
        line
        for line in open(tmp_path / "fetch_log.jsonl", encoding="utf-8")
        if line.strip()
    ]
    assert len(log) == 1


def test_force_refetches_and_appends_a_log_line(tmp_path, source) -> None:
    first = fetch_source(source, out_root=str(tmp_path), doc_iter=fake_docs())
    again = fetch_source(
        source, out_root=str(tmp_path), doc_iter=fake_docs(), force=True
    )
    assert again["cached"] is False
    assert again["sha256"] == first["sha256"], "same inputs, same bytes"
    log = [
        line
        for line in open(tmp_path / "fetch_log.jsonl", encoding="utf-8")
        if line.strip()
    ]
    assert len(log) == 2


def test_partial_file_without_log_record_is_refetched(tmp_path, source) -> None:
    """A crashed run leaves a file but no log line; it must not be trusted."""
    doc_dir = tmp_path / source.source_id
    doc_dir.mkdir(parents=True)
    (doc_dir / "documents.jsonl").write_text("{ truncated", encoding="utf-8")

    summary = fetch_source(source, out_root=str(tmp_path), doc_iter=fake_docs())
    assert summary["cached"] is False
    assert summary["doc_count"] > 0


# --------------------------------------------------------------------------
# reproducibility and offline-ness
# --------------------------------------------------------------------------


def test_two_runs_are_byte_identical(tmp_path, source) -> None:
    one = fetch_source(source, out_root=str(tmp_path / "a"), doc_iter=fake_docs())
    two = fetch_source(source, out_root=str(tmp_path / "b"), doc_iter=fake_docs())
    assert one["sha256"] == two["sha256"]
    assert open(one["path"], "rb").read() == open(two["path"], "rb").read()


def test_unpinned_injected_fetch_is_refused(tmp_path, source) -> None:
    unpinned = dataclasses.replace(source, revision="")
    with pytest.raises(ValueError, match="reproducible"):
        fetch_source(unpinned, out_root=str(tmp_path), doc_iter=fake_docs())


def test_no_network_libraries_were_imported() -> None:
    assert "datasets" not in sys.modules
    assert "huggingface_hub" not in sys.modules
