"""Epic 1.9 tests. Fully offline. Run with: pytest -q"""

from __future__ import annotations

import json

import pytest

from feature1_collect.corpus_schema import Document, validate_document
from feature1_collect.eval_split import EVAL_TARGET_FRACTION, carve_eval, select_eval
from feature1_collect.fetch import estimate_tokens
from feature1_collect.sources_manifest import SOURCES

POOL = 200_000
OTHER_SEED = "different-seed-2027"


def fake_candidates(n: int = 50, tier: str = "T1") -> list[dict]:
    """Deterministic eval candidates with varied sizes."""
    return [
        {
            "id": f"indic-wiki-hi-{i:07d}",
            "source_id": "indic-wiki-hi",
            "lane": "indic",
            "provenance_tier": tier,
            "est_tokens": 100 + (i * 7) % 50,
        }
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# select_eval — the pure part
# --------------------------------------------------------------------------


def test_selection_stops_at_or_just_past_target() -> None:
    cands = fake_candidates()
    result = select_eval(cands, total_pool_tokens=POOL)

    assert result["target_tokens"] == round(EVAL_TARGET_FRACTION * POOL)
    assert result["selected_tokens"] >= result["target_tokens"]
    # overshoot is bounded by the largest single candidate
    largest = max(c["est_tokens"] for c in cands)
    assert result["selected_tokens"] - result["target_tokens"] < largest


def test_selected_is_a_subset_of_candidates() -> None:
    cands = fake_candidates()
    result = select_eval(cands, total_pool_tokens=POOL)
    assert set(result["eval_ids"]) <= {c["id"] for c in cands}
    assert len(result["eval_ids"]) < len(cands), "should not take everything"


def test_eval_ids_are_sorted_and_unique() -> None:
    result = select_eval(fake_candidates(), total_pool_tokens=POOL)
    ids = result["eval_ids"]
    assert list(ids) == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_same_seed_is_reproducible() -> None:
    a = select_eval(fake_candidates(), total_pool_tokens=POOL)
    b = select_eval(fake_candidates(), total_pool_tokens=POOL)
    assert a["eval_ids"] == b["eval_ids"]
    assert a["fingerprint"] == b["fingerprint"]


def test_candidate_order_does_not_matter() -> None:
    """Selection is keyed by hash, not by input order."""
    forward = select_eval(fake_candidates(), total_pool_tokens=POOL)
    backward = select_eval(list(reversed(fake_candidates())), total_pool_tokens=POOL)
    assert forward["eval_ids"] == backward["eval_ids"]
    assert forward["fingerprint"] == backward["fingerprint"]


def test_different_seeds_give_different_splits() -> None:
    a = select_eval(fake_candidates(), total_pool_tokens=POOL)
    b = select_eval(fake_candidates(), total_pool_tokens=POOL, seed=OTHER_SEED)
    assert a["fingerprint"] != b["fingerprint"]
    assert a["eval_ids"] != b["eval_ids"]


def test_non_selected_are_disjoint_from_eval() -> None:
    cands = fake_candidates()
    result = select_eval(cands, total_pool_tokens=POOL)
    chosen = set(result["eval_ids"])
    train = [c["id"] for c in cands if c["id"] not in chosen]
    assert not (set(train) & chosen), "a doc is eval XOR train"
    assert len(train) + len(chosen) == len(cands)


def test_ineligible_tier_raises() -> None:
    cands = fake_candidates()
    cands[7]["provenance_tier"] = "T2"
    with pytest.raises(ValueError, match="not eval-eligible"):
        select_eval(cands, total_pool_tokens=POOL)


def test_duplicate_candidate_id_raises() -> None:
    cands = fake_candidates()
    cands[3]["id"] = cands[0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        select_eval(cands, total_pool_tokens=POOL)


def test_zero_target_selects_nothing() -> None:
    result = select_eval(fake_candidates(), total_pool_tokens=0)
    assert result["eval_ids"] == ()
    assert result["selected_tokens"] == 0


# --------------------------------------------------------------------------
# carve_eval — the I/O part
# --------------------------------------------------------------------------

EVAL_SOURCE_IDS = ("indic-wiki-hi", "mling-wiki-es")


@pytest.fixture
def eligible_sources():
    """A two-source SUBSET. eval_eligible must work without validate_sources."""
    return tuple(s for s in SOURCES if s.source_id in EVAL_SOURCE_IDS)


def build_raw_root(tmp_path, sources, docs_per_source: int = 12):
    """Lay out a fake data/raw the way fetch.py would have left it."""
    raw = tmp_path / "raw"
    total = 0
    log_lines = []
    for source in sources:
        source_dir = raw / source.source_id
        source_dir.mkdir(parents=True)
        est = 0
        with (source_dir / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for i in range(docs_per_source):
                text = f"{source.source_id} document {i:03d}. " + "content here. " * (i % 4 + 1)
                doc = {
                    "id": f"{source.source_id}-{i:07d}",
                    "lane": source.lane,
                    "provenance_tier": source.provenance_tier,
                    "split": "train",
                    "source": f"{source.dataset}@pinned-test",
                    "text": text,
                }
                fh.write(json.dumps(doc, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
                est += estimate_tokens(text)
        total += est
        log_lines.append({"source_id": source.source_id, "est_tokens": est})
    with (raw / "fetch_log.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for line in log_lines:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
    return raw, total


def read_manifest(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_carve_writes_header_and_eval_lines(tmp_path, eligible_sources) -> None:
    raw, _ = build_raw_root(tmp_path, eligible_sources)
    summary = carve_eval(
        raw_root=str(raw), eval_root=str(tmp_path / "eval"), sources=eligible_sources
    )

    lines = read_manifest(summary["manifest"])
    header, entries = lines[0], lines[1:]

    assert header["kind"] == "header"
    assert header["seed"] == summary["seed"]
    assert header["fingerprint"] == summary["fingerprint"]
    assert header["selected_count"] == len(entries) == summary["selected_count"]
    assert entries, "expected at least one eval document"
    for entry in entries:
        assert entry["split"] == "eval"
        assert entry["provenance_tier"] == "T1"


def test_every_eval_document_validates(tmp_path, eligible_sources) -> None:
    raw, _ = build_raw_root(tmp_path, eligible_sources)
    out = tmp_path / "eval"
    carve_eval(raw_root=str(raw), eval_root=str(out), sources=eligible_sources)

    seen = 0
    for source in eligible_sources:
        path = out / source.source_id / "documents.jsonl"
        if not path.exists():
            continue
        for row in read_manifest(path):
            document = Document(**row)
            assert validate_document(document) is document
            assert document.split == "eval"
            seen += 1
    assert seen == len(read_manifest(out / "eval_manifest.jsonl")) - 1


def test_eval_ids_come_only_from_eligible_sources(tmp_path, eligible_sources) -> None:
    raw, _ = build_raw_root(tmp_path, eligible_sources)
    summary = carve_eval(
        raw_root=str(raw), eval_root=str(tmp_path / "eval"), sources=eligible_sources
    )
    entries = read_manifest(summary["manifest"])[1:]
    assert {e["source_id"] for e in entries} <= set(EVAL_SOURCE_IDS)


def test_raw_files_are_never_mutated(tmp_path, eligible_sources) -> None:
    """The whole premise: eval is recorded, not moved."""
    raw, _ = build_raw_root(tmp_path, eligible_sources)
    before = {p: p.read_bytes() for p in sorted(raw.rglob("*.jsonl"))}
    carve_eval(raw_root=str(raw), eval_root=str(tmp_path / "eval"), sources=eligible_sources)
    after = {p: p.read_bytes() for p in sorted(raw.rglob("*.jsonl"))}
    assert before == after
    for row in read_manifest(raw / EVAL_SOURCE_IDS[0] / "documents.jsonl"):
        assert row["split"] == "train"


def test_second_call_is_cached(tmp_path, eligible_sources) -> None:
    raw, _ = build_raw_root(tmp_path, eligible_sources)
    out = tmp_path / "eval"
    first = carve_eval(raw_root=str(raw), eval_root=str(out), sources=eligible_sources)
    payload = (out / "eval_manifest.jsonl").read_bytes()

    second = carve_eval(raw_root=str(raw), eval_root=str(out), sources=eligible_sources)
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["fingerprint"] == first["fingerprint"]
    assert (out / "eval_manifest.jsonl").read_bytes() == payload


def test_a_new_seed_is_not_cached(tmp_path, eligible_sources) -> None:
    raw, _ = build_raw_root(tmp_path, eligible_sources)
    out = tmp_path / "eval"
    first = carve_eval(raw_root=str(raw), eval_root=str(out), sources=eligible_sources)
    again = carve_eval(
        raw_root=str(raw), eval_root=str(out), sources=eligible_sources, seed=OTHER_SEED
    )
    assert again["cached"] is False
    assert again["fingerprint"] != first["fingerprint"]


def test_ineligible_source_contributes_no_candidates(tmp_path) -> None:
    """A T2 source on disk is simply never a candidate."""
    web = next(s for s in SOURCES if s.source_id == "web-fineweb")   # T2
    wiki = next(s for s in SOURCES if s.source_id == "indic-wiki-hi")  # T1
    raw, _ = build_raw_root(tmp_path, (web, wiki))
    summary = carve_eval(
        raw_root=str(raw), eval_root=str(tmp_path / "eval"), sources=(web, wiki)
    )
    entries = read_manifest(summary["manifest"])[1:]
    assert {e["source_id"] for e in entries} == {"indic-wiki-hi"}
    assert not (tmp_path / "eval" / "web-fineweb").exists()
