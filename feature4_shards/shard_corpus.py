"""Feature 4 convenience — shard the whole corpus (train from data/clean, eval
from the loader) with the frozen tokenizer, then build + verify the index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from feature1_collect.corpus_loader import iter_documents
from feature1_collect.sources_manifest import SOURCES, LaneSource
from feature3_tokenizer.tokenizer_build import load_frozen_tokenizer
from feature4_shards.shard_index import build_index, verify_shards
from feature4_shards.shard_writer import SHARD_TOKENS, write_shards

__all__ = ["shard_corpus", "train_records", "eval_records"]


def train_records(clean_root: str = "data/clean") -> Iterator[dict[str, Any]]:
    """Cleaned train documents as shard records, in deterministic order."""
    for docs_file in sorted(Path(clean_root).rglob("documents.jsonl")):
        for line in docs_file.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            yield {"split": "train", "lane": d["lane"],
                   "provenance_tier": d["provenance_tier"], "id": d["id"], "text": d["text"]}


def eval_records(
    *, raw_root: str, eval_root: str, sources: Sequence[LaneSource]
) -> Iterator[dict[str, Any]]:
    """Held-out eval documents (via the loader) as shard records."""
    for loaded in iter_documents(raw_root=raw_root, eval_root=eval_root, sources=sources):
        doc = loaded.document
        if doc.split == "eval":
            yield {"split": "eval", "lane": doc.lane,
                   "provenance_tier": doc.provenance_tier, "id": doc.id, "text": doc.text}


def shard_corpus(
    *,
    clean_root: str = "data/clean",
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
    tokenizer_dir: str = "tokenizer",
    shard_root: str = "data/shards",
    shard_tokens: int = SHARD_TOKENS,
) -> dict[str, Any]:
    """Tokenize + shard the corpus, write the index, verify, and return the index."""
    tok = load_frozen_tokenizer(tokenizer_dir)
    records = list(train_records(clean_root)) + list(
        eval_records(raw_root=raw_root, eval_root=eval_root, sources=sources)
    )
    manifests = write_shards(records, tokenizer=tok, shard_root=shard_root, shard_tokens=shard_tokens)
    index = build_index(manifests, tokenizer_hash=tok.content_hash(), shard_root=shard_root)
    return index
