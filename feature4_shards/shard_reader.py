"""Feature 4 helper — stream documents (with their tokens) back out of shards.

Reads the shard index, and for each shard its manifest doc spans, yielding one
record per document in a deterministic order (index order, then doc order within
a shard). This is how downstream stages (packer, batch stream) recover per-doc
token sequences from the immutable shard files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from feature4_shards.shard_index import load_index
from feature4_shards.shard_writer import read_shard

__all__ = ["iter_docs"]


def iter_docs(shard_root: str, *, split: str = "train") -> Iterator[dict[str, Any]]:
    """Yield {doc_id, lane, tokens} for every doc in shards of `split`."""
    root = Path(shard_root)
    index = load_index(shard_root)
    for s in index["shards"]:
        if s["split"] != split:
            continue
        arr = read_shard(root / s["file"])
        man_path = root / s["split"] / s["lane"] / f"{s['shard_id']}.manifest.json"
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        for d in manifest["docs"]:
            start, length = d["start"], d["length"]
            yield {
                "doc_id": d["doc_id"],
                "lane": s["lane"],
                "tokens": arr[start:start + length].tolist(),
            }
