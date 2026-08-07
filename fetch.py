"""Epic 1.3 — the first real fetch, for ONE source (web-fineweb).

The other lanes are deliberately not fetched here.

Design: every part of this module that matters -- capping, id minting,
validation, JSON writing, hashing, logging -- runs with no network at all,
because the document stream is injectable. Pass `doc_iter` and nothing is
downloaded. Only `main()`, and only when `doc_iter` is None, reaches
HuggingFace, and `datasets` / `huggingface_hub` are imported inside that
branch so the tests need neither installed.

Reproducibility: a pinned revision fixes what the upstream yields, streaming
fixes the order, and the token cap fixes where we stop. Same three, same
bytes -- so documents.jsonl is byte-identical across runs, its sha256 is
stable, and a second call is a cache no-op rather than a re-download.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from corpus_schema import Document, validate_document
from sources_manifest import SOURCES, LaneSource, validate_sources

__all__ = ["estimate_tokens", "fetch_source", "main"]

# One JSON object per line, written the same way every time. sort_keys makes
# field order independent of dataclass field order; separators drop the
# whitespace that would otherwise vary; ensure_ascii=False keeps Devanagari
# and friends as themselves rather than \uXXXX escapes.
_JSON_KW: dict[str, Any] = {
    "sort_keys": True,
    "ensure_ascii": False,
    "separators": (",", ":"),
}

_DOCUMENTS_FILE = "documents.jsonl"
_LOG_FILE = "fetch_log.jsonl"


def estimate_tokens(text: str) -> int:
    """Pre-tokenizer byte estimate: roughly four UTF-8 bytes per token.

    Deliberately crude. It exists so the fetch can stop near a target without
    a tokenizer, and it is measured in BYTES, not characters -- so scripts
    outside Latin-1 cost more per character, which is the truth of it.
    """
    return max(1, len(text.encode("utf-8")) // 4)


def _extract_text(item: Any, *, index: int) -> str:
    """Accept either a plain string or a mapping carrying a "text" key."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        text = item.get("text")
        if isinstance(text, str):
            return text
        raise ValueError(
            f"item {index}: dict has no string 'text' key, got "
            f"{type(text).__name__}"
        )
    raise ValueError(
        f"item {index}: expected str or dict with 'text', got "
        f"{type(item).__name__}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_log_record(log_path: Path, source_id: str) -> dict[str, Any] | None:
    """Most recent fetch_log record for this source_id, or None.

    Last rather than first: a `force=True` refetch appends a new line, and
    the newest one describes what is on disk now.
    """
    if not log_path.exists():
        return None
    found: dict[str, Any] | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("source_id") == source_id:
                found = record
    return found


def _resolve_revision(source: LaneSource, doc_iter: Iterable[Any] | None) -> str:
    """A pinned revision, or the current upstream sha when actually fetching."""
    if source.revision:
        return source.revision
    if doc_iter is not None:
        raise ValueError(
            f"{source.source_id}: revision is empty and doc_iter was supplied, "
            f"so there is nothing to pin against. An injected fetch must "
            f"declare its revision, or the output is not reproducible."
        )
    from huggingface_hub import HfApi  # lazy: only the real fetch needs it

    return HfApi().dataset_info(source.dataset).sha


def _stream_hf(source: LaneSource, revision: str) -> Iterator[str]:
    """Stream `text` fields from the pinned upstream dataset."""
    from datasets import load_dataset  # lazy: tests never import this

    dataset = load_dataset(
        source.dataset,
        name=source.config or None,
        split="train",
        streaming=True,
        revision=revision,
    )
    for row in dataset:
        yield row["text"]


def fetch_source(
    source: LaneSource,
    *,
    out_root: str = "data/raw",
    doc_iter: Iterable[Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Materialise one source to <out_root>/<source_id>/documents.jsonl.

    Returns a summary dict. `cached=True` means nothing was written and the
    stream was never touched.
    """
    root = Path(out_root)
    doc_dir = root / source.source_id
    doc_path = doc_dir / _DOCUMENTS_FILE
    log_path = root / _LOG_FILE

    # (a) Cache. Both conditions are required, and that pairing is what makes
    # a crashed run safe: the log line is appended only after the file is
    # complete, so a half-written documents.jsonl has no record and is
    # correctly refetched rather than trusted.
    if not force and doc_path.exists():
        record = _last_log_record(log_path, source.source_id)
        if record is not None:
            return {
                "source_id": source.source_id,
                "revision": record["revision"],
                "path": str(doc_path),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "doc_count": record["doc_count"],
                "est_tokens": record["est_tokens"],
                "skipped": record.get("skipped", 0),
                "cached": True,
            }

    # (b) Revision, recorded into every document's `source` field.
    revision = _resolve_revision(source, doc_iter)

    # (c) Where the documents come from.
    stream: Iterable[Any] = doc_iter if doc_iter is not None else _stream_hf(source, revision)

    # (d) Write until the token cap is crossed.
    doc_dir.mkdir(parents=True, exist_ok=True)
    origin = f"{source.dataset}@{revision}"
    est_tokens = 0
    doc_count = 0
    skipped = 0

    # newline="\n" matters: without it Windows would translate to CRLF and the
    # same fetch would hash differently on a different machine.
    with doc_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, item in enumerate(stream):
            text = _extract_text(item, index=index)
            if not text.strip():
                # Real web dumps contain blanks. An empty document fails
                # validate_document, and dying at document 300,000 of a 4M
                # token fetch is worse than counting the skip.
                skipped += 1
                continue

            document = Document(
                id=f"{source.source_id}-{index:07d}",
                lane=source.lane,
                provenance_tier=source.provenance_tier,
                split="train",
                source=origin,
                text=text,
            )
            validate_document(document)

            handle.write(json.dumps(asdict(document), **_JSON_KW))
            handle.write("\n")

            doc_count += 1
            est_tokens += estimate_tokens(text)
            if est_tokens >= source.target_tokens:
                break

    # (e) Hash the finished file, then record it.
    file_bytes = doc_path.stat().st_size
    digest = _sha256(doc_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "source_id": source.source_id,
                    "dataset": source.dataset,
                    "config": source.config,
                    "revision": revision,
                    "path": str(doc_path),
                    "bytes": file_bytes,
                    "sha256": digest,
                    "doc_count": doc_count,
                    "skipped": skipped,
                    "est_tokens": est_tokens,
                    "target_tokens": source.target_tokens,
                },
                **_JSON_KW,
            )
        )
        handle.write("\n")

    # (f) Summary.
    return {
        "source_id": source.source_id,
        "revision": revision,
        "path": str(doc_path),
        "bytes": file_bytes,
        "sha256": digest,
        "doc_count": doc_count,
        "est_tokens": est_tokens,
        "skipped": skipped,
        "cached": False,
    }


def main() -> int:
    """Real fetch of web-fineweb only. This is the branch that hits the network."""
    validate_sources(SOURCES)
    source = next(s for s in SOURCES if s.source_id == "web-fineweb")
    summary = fetch_source(source)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
