"""Epics 4.3-4.4 — the shard-set index, plus immutability / re-hash verification.

build_index() writes one shard_index.json summarizing every shard (totals per
lane and split). verify_shards() re-reads each shard file, recomputes its
sha256, and checks it against both the index and the per-shard manifest -- the
immutability proof: if a single byte changed, the hash would not match.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

__all__ = ["INDEX_KIND", "INDEX_FILE", "build_index", "load_index", "verify_shards"]

INDEX_KIND = "shard_index"
INDEX_FILE = "shard_index.json"

_JSON = {"ensure_ascii": False, "sort_keys": True, "indent": 2}


def build_index(
    manifests: Iterable[dict[str, Any]],
    *,
    tokenizer_hash: str,
    shard_root: str,
) -> dict[str, Any]:
    """Write shard_index.json from the shard manifests and return it."""
    manifests = sorted(manifests, key=lambda m: m["shard_id"])
    shards = [
        {
            "shard_id": m["shard_id"], "hash": m["hash"], "file": m["file"],
            "n_tokens": m["n_tokens"], "n_docs": m["n_docs"],
            "split": m["split"], "lane": m["lane"],
        }
        for m in manifests
    ]
    by_lane: dict[str, dict[str, int]] = {}
    by_split: dict[str, dict[str, int]] = {}
    for m in manifests:
        for table, key in ((by_lane, m["lane"]), (by_split, m["split"])):
            b = table.setdefault(key, {"shards": 0, "tokens": 0, "docs": 0})
            b["shards"] += 1
            b["tokens"] += m["n_tokens"]
            b["docs"] += m["n_docs"]

    index = {
        "kind": INDEX_KIND,
        "tokenizer_hash": tokenizer_hash,
        "n_shards": len(shards),
        "total_tokens": sum(s["n_tokens"] for s in shards),
        "total_docs": sum(s["n_docs"] for s in shards),
        "by_split": {k: by_split[k] for k in sorted(by_split)},
        "by_lane": {k: by_lane[k] for k in sorted(by_lane)},
        "shards": shards,
    }
    root = Path(shard_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / INDEX_FILE).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(index, **_JSON) + "\n")
    return index


def load_index(shard_root: str) -> dict[str, Any]:
    return json.loads((Path(shard_root) / INDEX_FILE).read_text(encoding="utf-8"))


def verify_shards(shard_root: str) -> dict[str, Any]:
    """Re-hash every shard and confirm it matches the index and its manifest."""
    root = Path(shard_root)
    index = load_index(shard_root)
    mismatches: list[str] = []
    total_tokens = 0
    for s in index["shards"]:
        data = (root / s["file"]).read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != s["hash"]:
            mismatches.append(f"{s['shard_id']}: index hash mismatch")
        man_path = root / s["split"] / s["lane"] / f"{s['shard_id']}.manifest.json"
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        if manifest["hash"] != got:
            mismatches.append(f"{s['shard_id']}: manifest hash mismatch")
        total_tokens += len(data) // 2   # uint16 -> 2 bytes per token
    return {
        "ok": not mismatches and total_tokens == index["total_tokens"],
        "n_shards": index["n_shards"],
        "total_tokens": index["total_tokens"],
        "mismatches": mismatches,
    }
