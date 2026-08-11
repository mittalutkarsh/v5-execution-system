"""Epics 4.1-4.2 — immutable, content-addressed tokenized shards + manifests.

Each cleaned document is tokenized (with the frozen tokenizer) and wrapped as
<bos> ... <eos>. Whole documents are packed into fixed-size shards -- a document
is never split across a shard boundary, so provenance and packing stay clean.
Each shard is a uint16 token array written to a file whose name carries the
sha256 of its bytes (content-addressed), beside a per-shard manifest recording
the hash, token/doc counts, lane, provenance tiers, tags, the source doc ids,
and the tokenizer hash it was produced with. Shards are immutable: nothing
rewrites a shard once written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = ["DTYPE", "SHARD_TOKENS", "MANIFEST_KIND", "write_shards", "read_shard"]

DTYPE = "uint16"                 # vocab 12k < 65536, so a token fits in uint16
SHARD_TOKENS = 1 << 16           # 65,536 tokens: the target max per shard
MANIFEST_KIND = "shard_manifest"

_JSON = {"ensure_ascii": False, "sort_keys": True, "indent": 2}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_shard(path: str | Path) -> np.ndarray:
    """Load a shard file back into a uint16 token array."""
    return np.frombuffer(Path(path).read_bytes(), dtype=DTYPE)


def write_shards(
    records: Iterable[dict[str, Any]],
    *,
    tokenizer: Any,
    shard_root: str,
    shard_tokens: int = SHARD_TOKENS,
) -> list[dict[str, Any]]:
    """Tokenize `records` and write fixed-size shards. Returns the manifests.

    Each record is {split, lane, provenance_tier, id, text}. Shards are grouped
    by (split, lane) in sorted order so the output is deterministic.
    """
    bos = tokenizer.vocab["<bos>"]
    eos = tokenizer.vocab["<eos>"]
    tok_hash = tokenizer.content_hash()
    root = Path(shard_root)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["split"], r["lane"]), []).append(r)

    manifests: list[dict[str, Any]] = []
    for split, lane in sorted(groups):
        buf_tokens: list[int] = []
        buf_docs: list[dict[str, Any]] = []
        shard_idx = 0

        def flush() -> None:
            nonlocal buf_tokens, buf_docs, shard_idx
            if not buf_tokens:
                return
            arr = np.array(buf_tokens, dtype=DTYPE)
            data = arr.tobytes()
            h = _sha256(data)
            shard_id = f"{split}-{lane}-{shard_idx:04d}"
            out_dir = root / split / lane
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{shard_id}-{h[:12]}.bin"
            (out_dir / fname).write_bytes(data)
            tiers = sorted({d["provenance_tier"] for d in buf_docs})
            manifest = {
                "kind": MANIFEST_KIND,
                "shard_id": shard_id,
                "file": f"{split}/{lane}/{fname}",
                "hash": h,
                "dtype": DTYPE,
                "n_tokens": int(arr.size),
                "n_docs": len(buf_docs),
                "split": split,
                "lane": lane,
                "provenance_tiers": tiers,
                "tags": [f"split:{split}", f"lane:{lane}"] + [f"tier:{t}" for t in tiers],
                "tokenizer_hash": tok_hash,
                "docs": buf_docs,
            }
            with (out_dir / f"{shard_id}.manifest.json").open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(manifest, **_JSON) + "\n")
            manifests.append(manifest)
            buf_tokens = []
            buf_docs = []
            shard_idx += 1

        for r in groups[(split, lane)]:
            ids = [bos] + tokenizer.encode(r["text"]) + [eos]
            if buf_tokens and len(buf_tokens) + len(ids) > shard_tokens:
                flush()
            start = len(buf_tokens)
            buf_tokens.extend(ids)
            buf_docs.append({
                "doc_id": r["id"],
                "provenance_tier": r["provenance_tier"],
                "start": start,
                "length": len(ids),
            })
        flush()

    return manifests
