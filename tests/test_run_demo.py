"""Epic 1.12 tests. Fully offline. Run with: pytest -q"""

from __future__ import annotations

import json

import pytest

from feature1_collect.corpus_loader import corpus_counts
from feature1_collect.corpus_schema import LANES
from run_demo import ARTIFACTS_ROOT, RunLog, run
from feature1_collect.sources_manifest import SOURCES

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
    """data/raw and data/eval, laid out the way 1.3 and 1.9 leave them."""
    raw, evl = tmp_path / "raw", tmp_path / "eval"
    per_source = {WEB_ID: N_WEB, WIKI_ID: N_WIKI}

    for source in sources:
        directory = raw / source.source_id
        directory.mkdir(parents=True)
        with (directory / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for i in range(per_source[source.source_id]):
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


def do_run(corpus, artifacts_root):
    clean_root = str(artifacts_root) + "-clean"
    code = run(**corpus, artifacts_root=str(artifacts_root), clean_root=clean_root,
               tokenizer_dir=str(artifacts_root) + "-tok", vocab_size=400,
               shard_root=str(artifacts_root) + "-shards", n_steps=4)
    return code, (artifacts_root / "run.log").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# RunLog, on its own
# --------------------------------------------------------------------------


def test_runlog_line_format(tmp_path) -> None:
    path = tmp_path / "run.log"
    log = RunLog(path)
    log.info("run_start")
    log.info("corpus_lane", lane="web", train_docs=5)
    log.passed("corpus_loaded", total=11, eval=2)
    log.close()

    assert path.read_text(encoding="utf-8").splitlines() == [
        "[INFO] run_start",
        "[INFO] corpus_lane lane=web train_docs=5",
        "[PASS] corpus_loaded total=11 eval=2",
    ]


def test_runlog_records_events(tmp_path) -> None:
    log = RunLog(tmp_path / "run.log")
    log.info("run_start")
    log.info("corpus_lane", lane="web", train_docs=5)
    log.passed("corpus_loaded", total=11)
    log.close()

    assert log.events == [
        {"level": "INFO", "msg": "run_start"},
        {"level": "INFO", "msg": "corpus_lane", "lane": "web", "train_docs": 5},
        {"level": "PASS", "event": "corpus_loaded", "total": 11},
    ]


def test_runlog_carries_no_timestamp(tmp_path) -> None:
    log = RunLog(tmp_path / "run.log")
    log.info("run_start")
    log.close()
    assert (tmp_path / "run.log").read_text(encoding="utf-8") == "[INFO] run_start\n"


def test_runlog_refuses_an_absolute_path(tmp_path) -> None:
    """The determinism guarantee depends on this never slipping through."""
    log = RunLog(tmp_path / "run.log")
    with pytest.raises(ValueError, match="absolute path"):
        log.info("wrote", file="/home/someone/artifacts/corpus_summary.json")
    log.close()


def test_runlog_allows_relative_paths_with_slashes(tmp_path) -> None:
    """A dataset name has slashes but is not machine-specific."""
    log = RunLog(tmp_path / "run.log")
    log.info("source", dataset="wikimedia/wikipedia")
    log.close()
    assert "dataset=wikimedia/wikipedia" in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_runlog_starts_fresh_each_run(tmp_path) -> None:
    path = tmp_path / "run.log"
    first = RunLog(path); first.info("old_run"); first.close()
    second = RunLog(path); second.info("new_run"); second.close()
    text = path.read_text(encoding="utf-8")
    assert "old_run" not in text and "new_run" in text


# --------------------------------------------------------------------------
# run()
# --------------------------------------------------------------------------


def test_run_returns_zero(tmp_path, corpus) -> None:
    code, _ = do_run(corpus, tmp_path / "artifacts")
    assert code == 0


def test_run_log_has_the_pass_line(tmp_path, corpus) -> None:
    _, text = do_run(corpus, tmp_path / "artifacts")
    assert any(line.startswith("[PASS] corpus_loaded") for line in text.splitlines())


def test_run_log_has_one_lane_line_per_lane(tmp_path, corpus) -> None:
    _, text = do_run(corpus, tmp_path / "artifacts")
    lane_lines = [l for l in text.splitlines() if l.startswith("[INFO] corpus_lane")]
    assert len(lane_lines) == len(LANES)
    for lane in LANES:
        assert any(f"lane={lane} " in l for l in lane_lines), lane


def test_run_log_reports_the_fixture_counts(tmp_path, corpus) -> None:
    counts = corpus_counts(**corpus)
    totals = counts["totals"]
    _, text = do_run(corpus, tmp_path / "artifacts")
    assert f"total={totals['docs']}" in text
    assert f"eval={totals['eval_docs']}" in text
    assert f"contrastive={counts['contrastive']['pairs']}" in text


def test_run_log_brackets_the_stages(tmp_path, corpus) -> None:
    _, text = do_run(corpus, tmp_path / "artifacts")
    lines = text.splitlines()
    assert lines[0] == "[INFO] run_start"
    assert lines[-1] == "[INFO] run_complete"


def test_summary_artifact_is_written(tmp_path, corpus) -> None:
    artifacts = tmp_path / "artifacts"
    do_run(corpus, artifacts)
    summary = artifacts / "manifests" / "corpus_summary.json"
    assert summary.exists()
    assert json.loads(summary.read_text(encoding="utf-8"))["kind"] == "corpus_summary"


def test_log_mentions_the_summary_by_basename_only(tmp_path, corpus) -> None:
    _, text = do_run(corpus, tmp_path / "artifacts")
    assert "file=corpus_summary.json" in text
    assert str(tmp_path) not in text, "no absolute path may reach the log"


def test_two_runs_produce_byte_identical_logs(tmp_path, corpus) -> None:
    """Different artifact roots, same bytes -- no clock, no paths."""
    a, b = tmp_path / "artifacts-a", tmp_path / "deeper" / "artifacts-b"
    run(**corpus, artifacts_root=str(a), clean_root=str(tmp_path / "clean-a"),
        tokenizer_dir=str(tmp_path / "tok-a"), vocab_size=400, shard_root=str(tmp_path / "sh-a"), n_steps=4)
    run(**corpus, artifacts_root=str(b), clean_root=str(tmp_path / "clean-b"),
        tokenizer_dir=str(tmp_path / "tok-b"), vocab_size=400, shard_root=str(tmp_path / "sh-b"), n_steps=4)
    assert (a / "run.log").read_bytes() == (b / "run.log").read_bytes()


def test_run_log_has_the_clean_pass_line(tmp_path, corpus) -> None:
    _, text = do_run(corpus, tmp_path / "artifacts")
    assert any(line.startswith("[PASS] corpus_cleaned") for line in text.splitlines())
    assert "[INFO] clean_stage stage=normalize" in text
    assert "[INFO] clean_stage stage=decontam" in text


def test_run_log_has_the_tokenizer_pass_line(tmp_path, corpus) -> None:
    _, text = do_run(corpus, tmp_path / "artifacts")
    pass_lines = [l for l in text.splitlines() if l.startswith("[PASS] tokenizer_frozen")]
    assert len(pass_lines) == 1
    assert "vocab=" in pass_lines[0] and "hash=" in pass_lines[0]
    assert "[INFO] tokenizer_roundtrip lanes_checked=" in text


def test_run_log_has_the_shards_pass_line(tmp_path, corpus) -> None:
    _, text = do_run(corpus, tmp_path / "artifacts")
    pass_lines = [l for l in text.splitlines() if l.startswith("[PASS] shards_written")]
    assert len(pass_lines) == 1
    assert "verified=True" in pass_lines[0] and "tokens=" in pass_lines[0]


def test_run_log_has_downstream_pass_lines(tmp_path, corpus) -> None:
    _, text = do_run(corpus, tmp_path / "artifacts")
    for event in ("eval_shard_blocked", "mixture_compiled", "opus_selected",
                  "sequences_packed", "batch_stream_ready", "trained",
                  "contrastive_delta_s"):
        assert any(l.startswith(f"[PASS] {event}") for l in text.splitlines()), event


def test_run_creates_the_manifests_dir(tmp_path, corpus) -> None:
    artifacts = tmp_path / "nested" / "artifacts"
    do_run(corpus, artifacts)
    assert (artifacts / "manifests").is_dir()


def test_artifacts_root_default_is_stable() -> None:
    assert ARTIFACTS_ROOT == "submission_artifacts"
