"""Feature 4 tests — shards, manifests, index, immutability. Fully offline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature3_tokenizer.bpe_train import train_bpe
from feature3_tokenizer.bpe_tokenizer import Tokenizer
from feature4_shards.shard_index import build_index, load_index, verify_shards
from feature4_shards.shard_writer import DTYPE, read_shard, write_shards

SPECIALS = ("<pad>", "<bos>", "<eos>", "<doc>")
TRAIN_CORPUS = [
    "the monsoon reaches kerala in june every single year without fail here now",
    "बिल्ली और चूहा एक साथ खेलते हैं रोज सुबह",
    "def add(a, b): return a + b  # a small helper",
] * 20


@pytest.fixture
def tok():
    vocab, merges = train_bpe(TRAIN_CORPUS, vocab_size=600, special_tokens=SPECIALS)
    return Tokenizer(vocab, merges, SPECIALS)


def _records():
    return [
        {"split": "train", "lane": "web", "provenance_tier": "T2", "id": "web-0", "text": "the monsoon reaches kerala"},
        {"split": "train", "lane": "web", "provenance_tier": "T2", "id": "web-1", "text": "cricket dominates the market"},
        {"split": "train", "lane": "indic", "provenance_tier": "T1", "id": "indic-0", "text": "बिल्ली और चूहा"},
        {"split": "eval", "lane": "indic", "provenance_tier": "T1", "id": "indic-eval-0", "text": "एक प्रधान भाषा"},
    ]


def _write(tok, tmp_path, shard_tokens=1 << 16):
    mans = write_shards(_records(), tokenizer=tok, shard_root=str(tmp_path), shard_tokens=shard_tokens)
    idx = build_index(mans, tokenizer_hash=tok.content_hash(), shard_root=str(tmp_path))
    return mans, idx


def test_shards_are_written_and_reloadable(tok, tmp_path) -> None:
    mans, _ = _write(tok, tmp_path)
    assert mans, "at least one shard"
    for m in mans:
        arr = read_shard(tmp_path / m["file"])
        assert arr.dtype == np.dtype(DTYPE)
        assert arr.size == m["n_tokens"]


def test_each_doc_is_wrapped_in_bos_eos(tok, tmp_path) -> None:
    mans, _ = _write(tok, tmp_path)
    bos, eos = tok.vocab["<bos>"], tok.vocab["<eos>"]
    for m in mans:
        arr = read_shard(tmp_path / m["file"])
        for d in m["docs"]:
            span = arr[d["start"]: d["start"] + d["length"]]
            assert span[0] == bos and span[-1] == eos


def test_shard_filename_is_content_addressed(tok, tmp_path) -> None:
    mans, _ = _write(tok, tmp_path)
    for m in mans:
        assert m["hash"][:12] in m["file"]           # name carries the hash
        assert m["tokenizer_hash"] == tok.content_hash()


def test_manifest_records_provenance_and_docs(tok, tmp_path) -> None:
    mans, _ = _write(tok, tmp_path)
    indic = next(m for m in mans if m["lane"] == "indic" and m["split"] == "train")
    assert indic["provenance_tiers"] == ["T1"]
    assert "tier:T1" in indic["tags"] and "split:train" in indic["tags"]
    assert [d["doc_id"] for d in indic["docs"]] == ["indic-0"]


def test_index_totals_and_split_partition(tok, tmp_path) -> None:
    _, idx = _write(tok, tmp_path)
    assert idx["kind"] == "shard_index"
    assert idx["total_tokens"] == sum(s["n_tokens"] for s in idx["shards"])
    assert set(idx["by_split"]) == {"train", "eval"}


def test_verify_passes_on_untouched_shards(tok, tmp_path) -> None:
    _write(tok, tmp_path)
    result = verify_shards(str(tmp_path))
    assert result["ok"] is True and not result["mismatches"]


def test_verify_detects_a_mutated_shard(tok, tmp_path) -> None:
    _, idx = _write(tok, tmp_path)
    victim = tmp_path / idx["shards"][0]["file"]
    data = bytearray(victim.read_bytes())
    data[0] ^= 0xFF                                   # flip a byte -> immutability broken
    victim.write_bytes(bytes(data))
    result = verify_shards(str(tmp_path))
    assert result["ok"] is False and result["mismatches"]


def test_whole_docs_never_split_across_shards(tok, tmp_path) -> None:
    # a tiny shard cap forces multiple shards; each doc stays intact
    mans, _ = _write(tok, tmp_path, shard_tokens=8)
    bos, eos = tok.vocab["<bos>"], tok.vocab["<eos>"]
    for m in mans:
        arr = read_shard(tmp_path / m["file"])
        for d in m["docs"]:
            span = arr[d["start"]: d["start"] + d["length"]]
            assert span[0] == bos and span[-1] == eos   # never a fragment


def test_shard_writing_is_deterministic(tok, tmp_path) -> None:
    a = write_shards(_records(), tokenizer=tok, shard_root=str(tmp_path / "a"))
    b = write_shards(_records(), tokenizer=tok, shard_root=str(tmp_path / "b"))
    assert [m["hash"] for m in a] == [m["hash"] for m in b]
