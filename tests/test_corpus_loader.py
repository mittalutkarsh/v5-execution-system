"""Epic 1.10 tests. Fully offline. Run with: pytest -q"""

from __future__ import annotations

import json

import pytest

from feature1_collect.contrastive_pairs import CONTRASTIVE_PAIRS
from feature1_collect.corpus_loader import (
    CorpusView,
    corpus_counts,
    iter_documents,
    load_corpus,
    load_eval_ids,
)
from feature1_collect.corpus_schema import LANES, validate_document
from feature1_collect.fetch import estimate_tokens
from feature1_collect.sources_manifest import SOURCES

WEB_ID = "web-fineweb"      # T2, lane "web"      -- never eval-eligible
WIKI_ID = "indic-wiki-hi"   # T1, lane "indic"    -- eval-eligible
N_WEB, N_WIKI = 5, 6
JSON_KW = dict(sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@pytest.fixture
def fixture_sources():
    return tuple(s for s in SOURCES if s.source_id in (WEB_ID, WIKI_ID))


def doc_text(source_id: str, i: int) -> str:
    return f"{source_id} document {i:03d}. " + "body text for the corpus. " * (i % 3 + 1)


def build_corpus(tmp_path, sources, eval_indices=(1, 3), extra_eval_ids=()):
    """A fake data/raw plus a data/eval manifest, laid out as 1.3 and 1.9 leave them."""
    raw, evl = tmp_path / "raw", tmp_path / "eval"
    counts = {WEB_ID: N_WEB, WIKI_ID: N_WIKI}

    for source in sources:
        d = raw / source.source_id
        d.mkdir(parents=True)
        with (d / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for i in range(counts[source.source_id]):
                fh.write(json.dumps({
                    "id": f"{source.source_id}-{i:07d}",
                    "lane": source.lane,
                    "provenance_tier": source.provenance_tier,
                    "split": "train",                 # what fetch.py wrote
                    "source": f"{source.dataset}@pinned-test",
                    "text": doc_text(source.source_id, i),
                }, **JSON_KW) + "\n")

    # eval entries are drawn only from the T1 wiki source
    eval_ids = [f"{WIKI_ID}-{i:07d}" for i in eval_indices] + list(extra_eval_ids)
    evl.mkdir(parents=True)
    with (evl / "eval_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({
            "kind": "header", "seed": "v5-eval-2026", "target_fraction": 0.015,
            "selected_count": len(eval_ids), "fingerprint": "deadbeef",
        }, **JSON_KW) + "\n")
        for doc_id in eval_ids:
            fh.write(json.dumps({
                "id": doc_id, "source_id": WIKI_ID, "lane": "indic",
                "provenance_tier": "T1", "split": "eval", "est_tokens": 10,
            }, **JSON_KW) + "\n")
    return raw, evl, frozenset(eval_ids)


# --------------------------------------------------------------------------
# load_eval_ids
# --------------------------------------------------------------------------


def test_load_eval_ids_skips_the_header(tmp_path, fixture_sources) -> None:
    _, evl, expected = build_corpus(tmp_path, fixture_sources)
    got = load_eval_ids(str(evl))
    assert got == expected
    assert not any("header" in i for i in got)


def test_missing_manifest_means_no_eval(tmp_path) -> None:
    assert load_eval_ids(str(tmp_path / "nothing-here")) == frozenset()


# --------------------------------------------------------------------------
# iter_documents
# --------------------------------------------------------------------------


def test_yields_every_raw_document_exactly_once(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    loaded = list(iter_documents(raw_root=str(raw), eval_root=str(evl),
                                 sources=fixture_sources))
    ids = [l.document.id for l in loaded]
    assert len(ids) == N_WEB + N_WIKI
    assert len(set(ids)) == len(ids)


def test_split_is_routed_by_the_manifest(tmp_path, fixture_sources) -> None:
    raw, evl, expected = build_corpus(tmp_path, fixture_sources)
    loaded = list(iter_documents(raw_root=str(raw), eval_root=str(evl),
                                 sources=fixture_sources))
    got_eval = {l.document.id for l in loaded if l.document.split == "eval"}
    got_train = {l.document.id for l in loaded if l.document.split == "train"}
    assert got_eval == expected
    assert not (got_eval & got_train), "a document is eval XOR train"
    assert got_eval | got_train == {l.document.id for l in loaded}


def test_raw_split_value_is_overridden(tmp_path, fixture_sources) -> None:
    """Every raw row says "train"; the manifest still wins."""
    raw, evl, expected = build_corpus(tmp_path, fixture_sources)
    for row in (json.loads(l) for l in
                open(raw / WIKI_ID / "documents.jsonl", encoding="utf-8")):
        assert row["split"] == "train"
    loaded = list(iter_documents(raw_root=str(raw), eval_root=str(evl),
                                 sources=fixture_sources))
    assert {l.document.id for l in loaded if l.document.split == "eval"} == expected


def test_every_document_validates_and_is_sized(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    for loaded in iter_documents(raw_root=str(raw), eval_root=str(evl),
                                 sources=fixture_sources):
        assert validate_document(loaded.document) is loaded.document
        assert loaded.est_tokens > 0
        assert loaded.est_tokens == estimate_tokens(loaded.document.text)


def test_eval_documents_are_all_eligible_tiers(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    tiers = {l.document.provenance_tier for l in
             iter_documents(raw_root=str(raw), eval_root=str(evl), sources=fixture_sources)
             if l.document.split == "eval"}
    assert tiers <= {"T0", "T1"}


def test_iteration_is_deterministic(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    def run():
        return [(l.document.id, l.document.split, l.est_tokens)
                for l in iter_documents(raw_root=str(raw), eval_root=str(evl),
                                        sources=fixture_sources)]
    assert run() == run()


def test_source_order_does_not_depend_on_input_order(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    forward = [l.document.id for l in iter_documents(
        raw_root=str(raw), eval_root=str(evl), sources=fixture_sources)]
    backward = [l.document.id for l in iter_documents(
        raw_root=str(raw), eval_root=str(evl), sources=tuple(reversed(fixture_sources)))]
    assert forward == backward


def test_absent_source_file_is_skipped(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    extra = tuple(s for s in SOURCES if s.source_id == "code-github")
    loaded = list(iter_documents(raw_root=str(raw), eval_root=str(evl),
                                 sources=fixture_sources + extra))
    assert len(loaded) == N_WEB + N_WIKI


def test_manifest_id_absent_from_raw_raises(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources,
                               extra_eval_ids=(f"{WIKI_ID}-9999999",))
    with pytest.raises(ValueError, match="absent from"):
        list(iter_documents(raw_root=str(raw), eval_root=str(evl),
                            sources=fixture_sources))


# --------------------------------------------------------------------------
# corpus_counts
# --------------------------------------------------------------------------


def test_counts_match_the_fixture(tmp_path, fixture_sources) -> None:
    raw, evl, expected = build_corpus(tmp_path, fixture_sources)
    counts = corpus_counts(raw_root=str(raw), eval_root=str(evl),
                           sources=fixture_sources)

    web_tokens = sum(estimate_tokens(doc_text(WEB_ID, i)) for i in range(N_WEB))
    eval_idx = {int(i.rsplit("-", 1)[1]) for i in expected}
    wiki_eval = sum(estimate_tokens(doc_text(WIKI_ID, i)) for i in eval_idx)
    wiki_train = sum(estimate_tokens(doc_text(WIKI_ID, i))
                     for i in range(N_WIKI) if i not in eval_idx)

    assert counts["web"] == {"train_docs": N_WEB, "eval_docs": 0,
                             "train_tokens": web_tokens, "eval_tokens": 0}
    assert counts["indic"] == {"train_docs": N_WIKI - len(eval_idx),
                               "eval_docs": len(eval_idx),
                               "train_tokens": wiki_train, "eval_tokens": wiki_eval}
    assert counts["code"]["train_docs"] == 0, "unfed lanes still appear"
    assert set(LANES) <= set(counts)


def test_counts_totals_add_up(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    counts = corpus_counts(raw_root=str(raw), eval_root=str(evl),
                           sources=fixture_sources)
    lanes = [v for k, v in counts.items() if k in LANES]
    assert counts["totals"]["docs"] == sum(b["train_docs"] + b["eval_docs"] for b in lanes)
    assert counts["totals"]["tokens"] == sum(b["train_tokens"] + b["eval_tokens"] for b in lanes)
    assert counts["totals"]["eval_docs"] == sum(b["eval_docs"] for b in lanes)
    assert counts["totals"]["docs"] == N_WEB + N_WIKI


def test_contrastive_entry(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    counts = corpus_counts(raw_root=str(raw), eval_root=str(evl),
                           sources=fixture_sources)
    expected = sum(estimate_tokens(f"{p.prefix} {p.y_plus}")
                   + estimate_tokens(f"{p.prefix} {p.y_minus}")
                   for p in CONTRASTIVE_PAIRS)
    assert counts["contrastive"] == {"pairs": len(CONTRASTIVE_PAIRS),
                                     "est_tokens": expected}
    assert counts["contrastive"]["pairs"] >= 30


# --------------------------------------------------------------------------
# load_corpus
# --------------------------------------------------------------------------


def test_view_agrees_with_the_standalone_functions(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    view = load_corpus(raw_root=str(raw), eval_root=str(evl), sources=fixture_sources)
    assert isinstance(view, CorpusView)

    direct = [(l.document.id, l.document.split) for l in iter_documents(
        raw_root=str(raw), eval_root=str(evl), sources=fixture_sources)]
    viewed = [(l.document.id, l.document.split) for l in view.documents()]
    assert viewed == direct
    assert view.counts() == corpus_counts(raw_root=str(raw), eval_root=str(evl),
                                          sources=fixture_sources)
    assert view.contrastive == CONTRASTIVE_PAIRS


def test_view_documents_returns_a_fresh_iterator(tmp_path, fixture_sources) -> None:
    raw, evl, _ = build_corpus(tmp_path, fixture_sources)
    view = load_corpus(raw_root=str(raw), eval_root=str(evl), sources=fixture_sources)
    assert len(list(view.documents())) == len(list(view.documents())) == N_WEB + N_WIKI
