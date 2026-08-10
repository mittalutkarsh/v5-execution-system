"""Epic 2.7 — the cleaning pipeline. Composes 2.1-2.6, records every drop.

Reads the corpus through the loader, runs the six stages over the TRAIN docs in
a fixed order, and writes a cleaned corpus under data/clean/ plus a
cleaning_report. The raw files are never touched, and eval documents pass
through untouched (they are the benchmark) but are used as the decontamination
reference. Deterministic and idempotent: a second run is byte-identical.

Stage order (each consumes the previous stage's survivors):
  normalize -> exact-dedup -> quality -> near-dedup -> PII scrub -> decontaminate
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from contrastive_pairs import CONTRASTIVE_PAIRS
from content_hash import dedup_exact
from corpus_loader import iter_documents
from decontaminate import decontaminate
from near_dedup import dedup_near
from pii_scrub import scrub_pii
from quality_filter import filter_quality
from sources_manifest import SOURCES, LaneSource
from text_normalize import normalize_text

__all__ = ["REPORT_KIND", "REPORT_VERSION", "clean_corpus"]

REPORT_KIND = "cleaning_report"
REPORT_VERSION = 1
_DOCUMENTS_FILE = "documents.jsonl"
_REPORT_FILE = "cleaning_report.json"

_JSON_LINE = {"sort_keys": True, "ensure_ascii": False, "separators": (",", ":")}
_JSON_DOC = {"sort_keys": True, "indent": 2, "ensure_ascii": False}


def _source_id(doc_id: str) -> str:
    """Recover the source_id from a document id (f'{source_id}-{index:07d}')."""
    return doc_id.rsplit("-", 1)[0]


def clean_corpus(
    *,
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
    clean_root: str = "data/clean",
    contrastive: Sequence[Any] = CONTRASTIVE_PAIRS,
) -> dict[str, Any]:
    """Clean the train corpus, write it + a report under clean_root, return the report."""
    train: list[dict[str, Any]] = []
    eval_docs: list[dict[str, Any]] = []
    for loaded in iter_documents(raw_root=raw_root, eval_root=eval_root, sources=sources):
        row = asdict(loaded.document)
        (eval_docs if row["split"] == "eval" else train).append(row)

    stages: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []

    def record(stage: str, before: int, kept: int, dropped: list[dict[str, Any]], **extra: Any) -> None:
        stages.append({"stage": stage, "input": before, "kept": kept,
                       "dropped": len(dropped), **extra})
        drops.extend(dropped)

    n_in = len(train)

    # 1. normalize (modifies text, drops nothing)
    for row in train:
        row["text"] = normalize_text(row["text"])
    record("normalize", n_in, len(train), [])

    # 2. exact-duplicate removal
    before = len(train)
    train, d = dedup_exact(train)
    record("exact_dup", before, len(train), d)

    # 3. quality filter
    before = len(train)
    train, d = filter_quality(train)
    record("quality", before, len(train), d)

    # 4. near-duplicate removal
    before = len(train)
    train, d = dedup_near(train)
    record("near_dup", before, len(train), d)

    # 5. PII scrub (modifies text, drops nothing)
    before = len(train)
    redactions = 0
    for row in train:
        row["text"], n = scrub_pii(row["text"])
        redactions += n
    record("pii", before, len(train), [], redactions=redactions)

    # 6. decontaminate against eval + contrastive
    before = len(train)
    train, d = decontaminate(train, eval_docs, contrastive)
    record("decontam", before, len(train), d)

    # write cleaned train docs, grouped by source
    out = Path(clean_root)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in train:
        by_source.setdefault(_source_id(row["id"]), []).append(row)
    for source_id, rows in sorted(by_source.items()):
        d = out / source_id
        d.mkdir(parents=True, exist_ok=True)
        with (d / _DOCUMENTS_FILE).open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, **_JSON_LINE) + "\n")

    report = {
        "kind": REPORT_KIND,
        "version": REPORT_VERSION,
        "stages": stages,
        "pii_redactions": redactions,
        "drops": drops,
        "totals": {
            "train_in": n_in,
            "train_out": len(train),
            "dropped": n_in - len(train),
            "eval_docs": len(eval_docs),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / _REPORT_FILE).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, **_JSON_DOC) + "\n")
    return report
