"""Epic 3.7 tests — build, freeze, manifest, verify. Fully offline."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizer_build import (
    DEFAULT_SPECIALS,
    MANIFEST_FILE,
    build_frozen_tokenizer,
    verify_frozen,
    write_manifest,
)

CORPUS = [
    "the monsoon reaches kerala in june every year",
    "हिन्दी भारत की एक प्रमुख भाषा है",
    "বাংলা ভাষা দক্ষিণ এশিয়ার একটি ভাষা",
    "def add(a, b): return a + b  # sum",
] * 30


def _build(tmp, vocab_size=500):
    return build_frozen_tokenizer(texts=CORPUS, out_dir=str(tmp), vocab_size=vocab_size)


def test_build_writes_all_four_artifacts(tmp_path) -> None:
    _build(tmp_path)
    for name in ("vocab.json", "merges.txt", "special_tokens.json", MANIFEST_FILE):
        assert (tmp_path / name).exists()


def test_manifest_matches_the_tokenizer(tmp_path) -> None:
    tok = _build(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["kind"] == "tokenizer_manifest"
    assert manifest["hash"] == tok.content_hash()
    assert manifest["vocab_size"] == len(tok.vocab)
    assert manifest["n_merges"] == len(tok.merges)
    assert manifest["special_tokens"] == list(DEFAULT_SPECIALS)


def test_verify_frozen_is_true_after_build(tmp_path) -> None:
    _build(tmp_path)
    assert verify_frozen(str(tmp_path)) is True


def test_verify_frozen_detects_tampering(tmp_path) -> None:
    _build(tmp_path)
    # corrupt the manifest hash
    path = tmp_path / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["hash"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_frozen(str(tmp_path)) is False


def test_rebuild_reproduces_the_same_hash(tmp_path) -> None:
    a = _build(tmp_path / "a")
    b = _build(tmp_path / "b")
    assert a.content_hash() == b.content_hash()
    # and the serialized bytes match
    assert (tmp_path / "a" / "vocab.json").read_bytes() == (tmp_path / "b" / "vocab.json").read_bytes()
    assert (tmp_path / "a" / "merges.txt").read_bytes() == (tmp_path / "b" / "merges.txt").read_bytes()


def test_save_load_preserves_merges_that_start_with_hash(tmp_path) -> None:
    """A merge whose first symbol is '#' must survive reload (the '#' byte maps
    to the '#' symbol, so it must not be mistaken for a merges.txt comment)."""
    corpus = ["### heading ### block ###", "a # b # c # d"] * 60
    tok = build_frozen_tokenizer(texts=corpus, out_dir=str(tmp_path), vocab_size=320)
    assert any(a.startswith("#") for a, b in tok.merges)   # such a merge exists
    assert verify_frozen(str(tmp_path)) is True            # and reloads identically


def test_trained_on_reference_is_relative(tmp_path) -> None:
    tok = build_frozen_tokenizer(texts=CORPUS, out_dir=str(tmp_path),
                                 vocab_size=400, clean_root="data/clean")
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["trained_on"] == "data/clean"   # no absolute path leaks in
