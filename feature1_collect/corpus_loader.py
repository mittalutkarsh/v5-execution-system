"""Epic 1.10 — the corpus loader. One deterministic reader, no writes.

Yields every fetched document across every lane, with `split` routed by the
eval manifest: an id recorded there becomes split="eval", everything else
becomes split="train". A document is eval XOR train, decided by lookup rather
than by where the bytes live — which is why 1.9 never had to move a file.

Whatever `split` the raw file says is IGNORED and overwritten. The raw files
all say "train" because that is what fetch.py wrote; the manifest is the only
authority on eval membership.

Reads only. Nothing here creates, modifies, or deletes a file.

Two things worth knowing before relying on it:

  * `iter_documents` is a GENERATOR. The consistency check for manifest ids
    that never appeared in raw runs after the last yield, so it only fires on
    full exhaustion. `list(iter_documents(...))` raises; taking two documents
    and walking away does not.
  * `validate_document` does NOT enforce rule 3. It checks that a tier is one
    of the four legal strings, so a T3 document with split="eval" passes it.
    Rule 3 is a separate constraint and gets its own guard below, matching
    the one in eval_split.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Sequence

from feature1_collect.contrastive_pairs import CONTRASTIVE_PAIRS
from feature1_collect.corpus_schema import LANES, ContrastivePair, Document, validate_document
from feature1_collect.fetch import estimate_tokens
from feature1_collect.sources_manifest import EVAL_TIERS, SOURCES, LaneSource

__all__ = [
    "LoadedDocument",
    "CorpusView",
    "CONTRASTIVE_PAIRS",
    "load_eval_ids",
    "iter_documents",
    "corpus_counts",
    "load_corpus",
]

_DOCUMENTS_FILE = "documents.jsonl"
_MANIFEST_FILE = "eval_manifest.jsonl"


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """One document as the loader hands it out, with its size already measured."""

    document: Document
    est_tokens: int


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream one JSON object per line. Nothing is held after it is yielded."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_eval_ids(eval_root: str = "data/eval") -> frozenset[str]:
    """The ids Epic 1.9 recorded as eval. A missing manifest means none."""
    manifest = Path(eval_root) / _MANIFEST_FILE
    if not manifest.exists():
        return frozenset()
    return frozenset(
        row["id"] for row in _read_jsonl(manifest) if row.get("kind") != "header"
    )


def iter_documents(
    *,
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
) -> Iterator[LoadedDocument]:
    """Every fetched document, in a fixed order, with split routed by manifest.

    Order is sources by source_id, then file order within each source, so two
    runs over unchanged inputs yield an identical sequence.

    Raises ValueError if the manifest names an id that never appeared in raw --
    a manifest and a corpus that disagree is a consistency bug, and silently
    training on a corpus whose eval set is partly missing is worse than
    stopping. Note this fires only once the generator is exhausted.
    """
    raw = Path(raw_root)
    eval_ids = load_eval_ids(eval_root)
    seen_eval: set[str] = set()

    for source in sorted(sources, key=lambda s: s.source_id):
        path = raw / source.source_id / _DOCUMENTS_FILE
        if not path.exists():
            continue

        for row in _read_jsonl(path):
            doc_id = row["id"]
            is_eval = doc_id in eval_ids
            if is_eval:
                seen_eval.add(doc_id)
                # Rule 3, explicitly. validate_document below will not catch
                # this -- it accepts any of the four tiers on an eval doc.
                if row["provenance_tier"] not in EVAL_TIERS:
                    raise ValueError(
                        f"{doc_id}: manifest routes a "
                        f"{row['provenance_tier']!r} document to eval; rule 3 "
                        f"allows only {sorted(EVAL_TIERS)}"
                    )

            document = Document(
                id=doc_id,
                lane=row["lane"],
                provenance_tier=row["provenance_tier"],
                split="eval" if is_eval else "train",
                source=row["source"],
                text=row["text"],
            )
            validate_document(document)
            yield LoadedDocument(
                document=document, est_tokens=estimate_tokens(row["text"])
            )

    missing = eval_ids - seen_eval
    if missing:
        shown = sorted(missing)[:5]
        raise ValueError(
            f"eval manifest references {len(missing)} id(s) absent from "
            f"{raw_root}: {shown}{' …' if len(missing) > len(shown) else ''}"
        )


def _contrastive_tokens(
    pairs: Sequence[ContrastivePair] = CONTRASTIVE_PAIRS,
) -> int:
    """Both continuations count: each pair is scored as two full sequences."""
    return sum(
        estimate_tokens(f"{pair.prefix} {pair.y_plus}")
        + estimate_tokens(f"{pair.prefix} {pair.y_minus}")
        for pair in pairs
    )


def corpus_counts(
    *,
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
) -> dict[str, Any]:
    """One pass over the corpus, tallied per lane.

    Every lane in LANES appears even when nothing feeds it, so a gap reads as
    a zero rather than a missing key, and the dict shape is stable.
    """
    lanes: dict[str, dict[str, int]] = {
        lane: {
            "train_docs": 0,
            "eval_docs": 0,
            "train_tokens": 0,
            "eval_tokens": 0,
        }
        for lane in sorted(LANES)
    }

    for loaded in iter_documents(
        raw_root=raw_root, eval_root=eval_root, sources=sources
    ):
        bucket = lanes.setdefault(
            loaded.document.lane,
            {"train_docs": 0, "eval_docs": 0, "train_tokens": 0, "eval_tokens": 0},
        )
        prefix = "eval" if loaded.document.split == "eval" else "train"
        bucket[f"{prefix}_docs"] += 1
        bucket[f"{prefix}_tokens"] += loaded.est_tokens

    counts: dict[str, Any] = dict(lanes)
    counts["contrastive"] = {
        "pairs": len(CONTRASTIVE_PAIRS),
        "est_tokens": _contrastive_tokens(),
    }
    counts["totals"] = {
        "docs": sum(b["train_docs"] + b["eval_docs"] for b in lanes.values()),
        "tokens": sum(b["train_tokens"] + b["eval_tokens"] for b in lanes.values()),
        "eval_docs": sum(b["eval_docs"] for b in lanes.values()),
        "eval_tokens": sum(b["eval_tokens"] for b in lanes.values()),
    }
    return counts


@dataclass(frozen=True, slots=True)
class CorpusView:
    """A bound view of one corpus on disk. Holds paths, not data."""

    raw_root: str
    eval_root: str
    sources: tuple[LaneSource, ...]
    contrastive: tuple[ContrastivePair, ...]

    def documents(self) -> Iterator[LoadedDocument]:
        """A fresh iterator each call -- the view holds no state."""
        return iter_documents(
            raw_root=self.raw_root,
            eval_root=self.eval_root,
            sources=self.sources,
        )

    def counts(self) -> dict[str, Any]:
        return corpus_counts(
            raw_root=self.raw_root,
            eval_root=self.eval_root,
            sources=self.sources,
        )


def load_corpus(
    *,
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
) -> CorpusView:
    """Bind a corpus on disk. Reads nothing until documents() or counts()."""
    return CorpusView(
        raw_root=raw_root,
        eval_root=eval_root,
        sources=tuple(sources),
        contrastive=CONTRASTIVE_PAIRS,
    )
