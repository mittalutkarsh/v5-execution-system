"""Epic 1.11 — a deterministic, regenerable corpus summary.

Reads the corpus through `corpus_counts` and writes one JSON file. No network,
no mutation of anything the earlier epics produced.

Deterministic means byte-identical: regenerate over an unchanged corpus and the
file does not move a single byte, so `git diff` stays empty and a changed
summary always means the corpus actually changed.

That property is fragile in one specific way, worth stating because it is the
first thing anyone adds to a report: there is deliberately NO timestamp, no
hostname, no run id, no library version. Any of those would make every
regeneration a diff, and the file would stop being useful as a change signal.
If you later want provenance of that kind, put it in a sibling file rather than
in here.

Floats are rounded to six places before serialisation for the same reason --
an unrounded ratio can differ in its last bits across platforms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Sequence

from corpus_loader import corpus_counts
from corpus_schema import LANES
from sources_manifest import SOURCES, LaneSource

__all__ = [
    "REPORT_KIND",
    "REPORT_VERSION",
    "build_summary",
    "write_corpus_summary",
    "load_corpus_summary",
]

REPORT_KIND: Final[str] = "corpus_summary"
REPORT_VERSION: Final[int] = 1

_ROUND = 6

_JSON_KW: dict[str, Any] = {
    "sort_keys": True,
    "indent": 2,
    "ensure_ascii": False,
}


def build_summary(
    *,
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
) -> dict[str, Any]:
    """Tally the corpus and shape it into the report dict. Writes nothing."""
    counts = corpus_counts(
        raw_root=raw_root, eval_root=eval_root, sources=sources
    )
    lanes = {lane: dict(counts[lane]) for lane in sorted(LANES)}
    totals = dict(counts["totals"])

    # Train totals are derived rather than tallied separately: a document is
    # eval XOR train, so train is exactly what eval is not.
    train_docs = totals["docs"] - totals["eval_docs"]
    train_tokens = totals["tokens"] - totals["eval_tokens"]

    eval_fraction = (
        round(totals["eval_tokens"] / totals["tokens"], _ROUND)
        if totals["tokens"]
        else 0.0
    )
    # An empty corpus gives every lane a share of 0.0 rather than a ZeroDivisionError.
    lane_share = {
        lane: (
            round(lanes[lane]["train_tokens"] / train_tokens, _ROUND)
            if train_tokens
            else 0.0
        )
        for lane in sorted(LANES)
    }

    return {
        "kind": REPORT_KIND,
        "version": REPORT_VERSION,
        "lanes": lanes,
        "contrastive": dict(counts["contrastive"]),
        "totals": totals,
        "derived": {
            "train_docs": train_docs,
            "train_tokens": train_tokens,
            "eval_token_fraction": eval_fraction,
            "lane_train_token_share": lane_share,
        },
    }


def write_corpus_summary(
    *,
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
    out_path: str = "data/corpus_summary.json",
) -> dict[str, Any]:
    """Build the summary and persist it. Returns the report that was written."""
    report = build_summary(
        raw_root=raw_root, eval_root=eval_root, sources=sources
    )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" matters: without it Windows writes CRLF and the same corpus
    # produces a different file on a different machine.
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, **_JSON_KW) + "\n")
    return report


def load_corpus_summary(path: str = "data/corpus_summary.json") -> dict[str, Any]:
    """Read a previously written summary back."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
