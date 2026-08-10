"""Epic 1.12 — the one-command runner. Last epic of Feature 1.

One entry point that runs the pipeline and leaves everything a marker needs
under `submission_artifacts/`. It grows one stage per feature; today it has
exactly one, so the shape matters more than the content.

`run.log` is byte-identical across machines and across artifact locations. Two
things buy that, and both are easy to break later:

  * NO timestamps. A clock in the log makes every run a diff, so a changed log
    would stop meaning a changed pipeline.
  * NO absolute paths. Only basenames are logged, so the same run under
    /home/x/artifacts and /tmp/y/artifacts produces the same bytes. This is
    enforced rather than trusted -- `RunLog` refuses to write a field value
    that looks like an absolute path, because that failure would otherwise be
    invisible until someone diffed two checkouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Sequence

from clean_pipeline import clean_corpus
from corpus_loader import corpus_counts
from corpus_report import write_corpus_summary
from corpus_schema import LANES
from sources_manifest import SOURCES, LaneSource

__all__ = [
    "ARTIFACTS_ROOT",
    "RunLog",
    "stage_load_corpus",
    "stage_clean_corpus",
    "run",
    "main",
]

ARTIFACTS_ROOT: Final[str] = "submission_artifacts"
_RUN_LOG = "run.log"
_MANIFESTS_DIR = "manifests"
_SUMMARY_FILE = "corpus_summary.json"


def _render(value: Any) -> str:
    """Stringify one field value, refusing anything machine-specific."""
    text = str(value)
    looks_absolute = text.startswith("/") or (
        len(text) > 2 and text[1] == ":" and text[2] in "\\/"
    )
    if looks_absolute:
        raise ValueError(
            f"refusing to log the absolute path {text!r}: run.log must stay "
            f"byte-identical wherever artifacts_root lives. Log a basename."
        )
    return text


class RunLog:
    """Append-only run log. Writes to a file and to stdout, and remembers.

    Field order follows the order the caller passed them, which is well
    defined for **kwargs, so the same call site always renders the same line.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.events: list[dict[str, Any]] = []
        # newline="\n" so a Windows run does not produce CRLF and a different file
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def _emit(self, line: str) -> None:
        self._handle.write(line + "\n")
        print(line)

    @staticmethod
    def _suffix(fields: dict[str, Any]) -> str:
        if not fields:
            return ""
        return " " + " ".join(f"{k}={_render(v)}" for k, v in fields.items())

    def info(self, msg: str, **fields: Any) -> None:
        self._emit(f"[INFO] {msg}{self._suffix(fields)}")
        self.events.append({"level": "INFO", "msg": msg, **fields})

    def passed(self, event: str, **fields: Any) -> None:
        self._emit(f"[PASS] {event}{self._suffix(fields)}")
        self.events.append({"level": "PASS", "event": event, **fields})

    def close(self) -> None:
        self._handle.close()


def stage_load_corpus(
    log: RunLog,
    *,
    raw_root: str,
    eval_root: str,
    sources: Sequence[LaneSource],
    summary_path: str | Path,
) -> dict[str, Any]:
    """Stage 1: tally the corpus, persist the summary, report per lane."""
    counts = corpus_counts(
        raw_root=raw_root, eval_root=eval_root, sources=sources
    )

    for lane in sorted(LANES):
        bucket = counts[lane]
        log.info(
            "corpus_lane",
            lane=lane,
            train_docs=bucket["train_docs"],
            eval_docs=bucket["eval_docs"],
            train_tokens=bucket["train_tokens"],
            eval_tokens=bucket["eval_tokens"],
        )

    write_corpus_summary(
        raw_root=raw_root,
        eval_root=eval_root,
        sources=sources,
        out_path=str(summary_path),
    )
    log.info("corpus_summary_written", file=Path(summary_path).name)

    log.passed(
        "corpus_loaded",
        total=counts["totals"]["docs"],
        eval=counts["totals"]["eval_docs"],
        contrastive=counts["contrastive"]["pairs"],
    )
    return counts


def stage_clean_corpus(
    log: RunLog,
    *,
    raw_root: str,
    eval_root: str,
    sources: Sequence[LaneSource],
    clean_root: str,
) -> dict[str, Any]:
    """Stage 2: clean the train corpus and report each stage's drops."""
    report = clean_corpus(
        raw_root=raw_root, eval_root=eval_root, sources=sources, clean_root=clean_root
    )
    for s in report["stages"]:
        log.info(
            "clean_stage",
            stage=s["stage"],
            input=s["input"],
            kept=s["kept"],
            dropped=s["dropped"],
        )
    log.info("pii_redactions", n=report["pii_redactions"])
    totals = report["totals"]
    log.passed(
        "corpus_cleaned",
        kept=totals["train_out"],
        dropped=totals["dropped"],
    )
    return report


def run(
    *,
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
    artifacts_root: str = ARTIFACTS_ROOT,
    clean_root: str = "data/clean",
) -> int:
    """Run every stage. Returns a process exit code: 0 on success."""
    root = Path(artifacts_root)
    manifests = root / _MANIFESTS_DIR
    manifests.mkdir(parents=True, exist_ok=True)

    log = RunLog(root / _RUN_LOG)
    try:
        log.info("run_start")
        stage_load_corpus(
            log,
            raw_root=raw_root,
            eval_root=eval_root,
            sources=sources,
            summary_path=manifests / _SUMMARY_FILE,
        )
        stage_clean_corpus(
            log,
            raw_root=raw_root,
            eval_root=eval_root,
            sources=sources,
            clean_root=clean_root,
        )
        log.info("run_complete")
    finally:
        # even a failing stage leaves a readable log behind
        log.close()
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
