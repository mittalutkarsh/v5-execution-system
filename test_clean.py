"""Feature 2 (Clean & filter) tests — stages 2.1-2.6 and the 2.7 pipeline.
Fully offline. Run with: pytest -q
"""

from __future__ import annotations

import json

import pytest

from content_hash import content_hash, dedup_exact
from corpus_schema import Document, validate_document
from decontaminate import decontaminate, ngrams
from near_dedup import dedup_near, est_jaccard, minhash
from pii_scrub import EMAIL_PLACEHOLDER, PHONE_PLACEHOLDER, scrub_pii
from quality_filter import quality_ok
from text_normalize import normalize_document, normalize_text

JSON_KW = dict(sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def row(doc_id, text, lane="web", tier="T2", split="train"):
    return {"id": doc_id, "lane": lane, "provenance_tier": tier,
            "split": split, "source": "src@rev", "text": text}


# --------------------------------------------------------------------------
# 2.1 normalize
# --------------------------------------------------------------------------


def test_normalize_collapses_whitespace_and_trims() -> None:
    assert normalize_text("  a    b\t\tc  ") == "a b c"


def test_normalize_is_idempotent() -> None:
    messy = "  Ｈello\r\n\r\n\r\nworld \x00\x07  trailing   "
    once = normalize_text(messy)
    assert normalize_text(once) == once


def test_normalize_drops_control_chars_and_collapses_tabs_keeping_newlines() -> None:
    assert "\x00" not in normalize_text("a\x00b")
    assert normalize_text("a\nb\tc") == "a\nb c"   # tab collapses to a space; newline kept


def test_normalize_keeps_zero_width_joiner_for_indic() -> None:
    """ZWJ (U+200D, category Cf) is meaningful in Indic scripts and must survive."""
    text = "क्‍ष"  # a Devanagari conjunct using ZWJ
    assert "‍" in normalize_text(text)


def test_normalize_applies_nfc() -> None:
    decomposed = "é"       # e + combining acute
    assert normalize_text(decomposed) == "é"  # é


def test_normalize_document_preserves_other_fields() -> None:
    doc = Document(id="d1", lane="web", provenance_tier="T2", split="train",
                   source="s@r", text="  hi   there  ")
    out = normalize_document(doc)
    assert out.text == "hi there"
    assert (out.id, out.lane, out.provenance_tier, out.split, out.source) == \
           (doc.id, doc.lane, doc.provenance_tier, doc.split, doc.source)


# --------------------------------------------------------------------------
# 2.2 content hash + exact dedup
# --------------------------------------------------------------------------


def test_content_hash_is_stable_and_equal_for_equal_text() -> None:
    assert content_hash("hello world") == content_hash("hello world")
    assert content_hash("a") != content_hash("b")
    assert len(content_hash("x")) == 64


def test_dedup_exact_keeps_first_and_names_survivor() -> None:
    docs = [row("a", "same text here"), row("b", "unique one"),
            row("c", "same text here")]
    kept, drops = dedup_exact(docs)
    assert [d["id"] for d in kept] == ["a", "b"]
    assert drops == [{"id": "c", "stage": "exact_dup",
                      "reason": "identical content to an earlier document",
                      "duplicate_of": "a"}]


# --------------------------------------------------------------------------
# 2.3 quality
# --------------------------------------------------------------------------


def test_quality_accepts_a_normal_doc() -> None:
    ok, reason = quality_ok("The monsoon usually reaches Kerala around the first of June.")
    assert ok and reason is None


def test_quality_rejects_too_short() -> None:
    ok, reason = quality_ok("hi")
    assert not ok and "short" in reason


def test_quality_rejects_symbol_spam() -> None:
    # five real words so it clears the word-count check, then heavy symbols
    ok, reason = quality_ok("alpha beta gamma delta epsilon " + "#$%@^&*()" * 6)
    assert not ok and "symbol" in reason


def test_quality_rejects_single_token_repetition() -> None:
    ok, reason = quality_ok("buy " * 40)
    assert not ok and "repetitive" in reason


# --------------------------------------------------------------------------
# 2.4 near-dup
# --------------------------------------------------------------------------


def test_minhash_is_deterministic() -> None:
    assert minhash("the quick brown fox jumps") == minhash("the quick brown fox jumps")


def test_near_identical_documents_are_deduped() -> None:
    base = "the southwest monsoon reaches the kerala coast around the first of june each year"
    docs = [row("a", base),
            row("b", base + " reliably"),      # near-identical
            row("c", "an entirely unrelated sentence about cricket and hockey")]
    kept, drops = dedup_near(docs, threshold=0.6)
    kept_ids = {d["id"] for d in kept}
    assert "a" in kept_ids and "c" in kept_ids
    assert [d["id"] for d in drops] == ["b"]
    assert drops[0]["near"] == "a" and drops[0]["stage"] == "near_dup"


def test_unrelated_documents_are_both_kept() -> None:
    docs = [row("a", "cricket is the centre of the sport's global economy"),
            row("b", "the financial year in india begins in april")]
    kept, drops = dedup_near(docs)
    assert len(kept) == 2 and drops == []


def test_dedup_near_is_deterministic() -> None:
    docs = [row(str(i), f"document number {i} about various unrelated topics") for i in range(10)]
    a = [d["id"] for d in dedup_near(docs)[0]]
    b = [d["id"] for d in dedup_near(list(docs))[0]]
    assert a == b


# --------------------------------------------------------------------------
# 2.5 PII
# --------------------------------------------------------------------------


def test_pii_redacts_email_and_phone() -> None:
    text, n = scrub_pii("write to alice@example.com or call +91 98765 43210 today")
    assert EMAIL_PLACEHOLDER in text and PHONE_PLACEHOLDER in text
    assert "alice@example.com" not in text
    assert n == 2


def test_pii_leaves_ordinary_text_alone() -> None:
    text, n = scrub_pii("the year 2024 had 365 days and chapter 7 was short")
    assert n == 0 and text == "the year 2024 had 365 days and chapter 7 was short"


def test_pii_is_idempotent() -> None:
    once, _ = scrub_pii("mail me at bob@site.org")
    twice, n = scrub_pii(once)
    assert twice == once and n == 0


# --------------------------------------------------------------------------
# 2.6 decontamination
# --------------------------------------------------------------------------


def test_ngrams_below_n_is_empty() -> None:
    assert ngrams("only three words here", n=13) == set()


def test_contaminated_train_doc_is_dropped() -> None:
    span = " ".join(f"word{i}" for i in range(20))
    eval_docs = [row("e-1", span, tier="T1", split="eval")]
    train = [row("t-1", "clean preamble " + span + " clean tail"),
             row("t-2", "a completely different unrelated training document here")]
    kept, drops = decontaminate(train, eval_docs, ())
    assert [d["id"] for d in kept] == ["t-2"]
    assert drops[0]["id"] == "t-1" and drops[0]["stage"] == "decontam"


def test_decontam_leaves_clean_train_alone() -> None:
    eval_docs = [row("e-1", " ".join(f"tok{i}" for i in range(20)), tier="T1", split="eval")]
    train = [row("t-1", "nothing in common with the eval set at all here friends")]
    kept, drops = decontaminate(train, eval_docs, ())
    assert len(kept) == 1 and drops == []


# --------------------------------------------------------------------------
# 2.7 pipeline
# --------------------------------------------------------------------------


@pytest.fixture
def cleanable(tmp_path):
    """A tiny raw+eval corpus with a duplicate, junk, PII, and a contaminated doc."""
    from sources_manifest import SOURCES
    web = next(s for s in SOURCES if s.source_id == "web-fineweb")     # T2
    wiki = next(s for s in SOURCES if s.source_id == "indic-wiki-hi")  # T1
    raw, evl = tmp_path / "raw", tmp_path / "eval"

    eval_span = " ".join(f"benchword{i}" for i in range(20))
    web_docs = [
        ("web-fineweb-0000000", "The southwest monsoon reaches the Kerala coast in early June each year here."),
        ("web-fineweb-0000001", "The southwest monsoon reaches the Kerala coast in early June each year here."),  # exact dup
        ("web-fineweb-0000002", "Contact us at team@example.com for the full monsoon onset report and details."),
        ("web-fineweb-0000003", "leaked preamble " + eval_span + " trailing text of the contaminated document"),
        ("web-fineweb-0000004", "##"),  # junk (too short / symbol)
    ]
    (raw / web.source_id).mkdir(parents=True)
    with (raw / web.source_id / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for did, text in web_docs:
            fh.write(json.dumps(row(did, text, lane="web", tier="T2"), **JSON_KW) + "\n")

    (raw / wiki.source_id).mkdir(parents=True)
    with (raw / wiki.source_id / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row("indic-wiki-hi-0000000",
                                "A perfectly ordinary Hindi Wikipedia paragraph about the monsoon season.",
                                lane="indic", tier="T1"), **JSON_KW) + "\n")

    evl.mkdir(parents=True)
    with (evl / "eval_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"kind": "header", "fingerprint": "x"}, **JSON_KW) + "\n")
        # eval doc lives in the wiki lane; give it the contaminating span
    # put the eval doc as a second wiki doc and mark it eval via the manifest
    with (raw / wiki.source_id / "documents.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row("indic-wiki-hi-0000001", eval_span, lane="indic", tier="T1"),
                            **JSON_KW) + "\n")
    with (evl / "eval_manifest.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"id": "indic-wiki-hi-0000001", "source_id": "indic-wiki-hi",
                             "lane": "indic", "provenance_tier": "T1", "split": "eval",
                             "est_tokens": 20}, **JSON_KW) + "\n")

    return {"raw_root": str(raw), "eval_root": str(evl), "sources": (web, wiki),
            "clean_root": str(tmp_path / "clean")}


def test_pipeline_drops_each_kind_and_reports(cleanable) -> None:
    from clean_pipeline import clean_corpus
    report = clean_corpus(**cleanable, contrastive=())
    stages = {s["stage"]: s for s in report["stages"]}
    assert stages["exact_dup"]["dropped"] == 1
    assert stages["quality"]["dropped"] >= 1
    assert stages["decontam"]["dropped"] == 1
    assert report["pii_redactions"] >= 1
    assert report["kind"] == "cleaning_report"
    # every drop names a stage and a reason
    for d in report["drops"]:
        assert d["stage"] and d["reason"]


def test_pipeline_writes_cleaned_docs_that_validate(cleanable) -> None:
    from clean_pipeline import clean_corpus
    from pathlib import Path
    clean_corpus(**cleanable, contrastive=())
    clean = Path(cleanable["clean_root"])
    seen = 0
    for path in clean.rglob("documents.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            doc = Document(**json.loads(line))
            assert validate_document(doc) is doc
            assert "@example.com" not in doc.text   # PII scrubbed
            seen += 1
    assert seen >= 1


def test_pipeline_is_idempotent(cleanable) -> None:
    from clean_pipeline import clean_corpus
    from pathlib import Path
    clean_corpus(**cleanable, contrastive=())
    report_path = Path(cleanable["clean_root"]) / "cleaning_report.json"
    first = report_path.read_bytes()
    clean_corpus(**cleanable, contrastive=())
    assert report_path.read_bytes() == first


def test_pipeline_does_not_mutate_raw(cleanable) -> None:
    from clean_pipeline import clean_corpus
    from pathlib import Path
    raw = Path(cleanable["raw_root"])
    before = {p: p.read_bytes() for p in sorted(raw.rglob("*.jsonl"))}
    clean_corpus(**cleanable, contrastive=())
    after = {p: p.read_bytes() for p in sorted(raw.rglob("*.jsonl"))}
    assert before == after
