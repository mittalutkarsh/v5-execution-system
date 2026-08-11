"""Epic 1.9 — carve a small, quarantined eval split.

The fetched files are IMMUTABLE. Nothing here rewrites, moves, or deletes
anything under the raw root. Eval membership is RECORDED — an id list plus a
fingerprint — and the loader (later) routes a recorded id to split="eval" and
everything else to split="train". A document is eval XOR train, never both.

Rule 3, from the tier convention in corpus_schema: eval may be drawn only
from T0/T1 sources. That rule is enforced twice on purpose:

  * `select_eval` refuses any candidate outside EVAL_TIERS, and
  * `carve_eval` re-checks each document again at write time.

Note that `validate_document` does NOT cover this. It checks that a tier is
one of the four legal strings, so a T3 document with split="eval" passes it
happily. Rule 3 is a separate constraint and needs its own guard.

Determinism without an RNG: each candidate is keyed by
sha256(f"{seed}:{id}"), the candidates are sorted by that key, and taken in
order until the token target is met. No RNG module, no seeding ritual, no
dependence on dict or filesystem ordering. The same corpus and the same seed
select the same ids on any machine, and the fingerprint proves it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

from feature1_collect.corpus_schema import Document, validate_document
from feature1_collect.fetch import estimate_tokens
from feature1_collect.sources_manifest import EVAL_TIERS, SOURCES, LaneSource, eval_eligible

__all__ = [
    "EVAL_TARGET_FRACTION",
    "DEFAULT_SEED",
    "select_eval",
    "carve_eval",
]

EVAL_TARGET_FRACTION: Final[float] = 0.015  # ~1.5% of the pool
DEFAULT_SEED: Final[str] = "v5-eval-2026"

_MANIFEST_FILE = "eval_manifest.jsonl"
_DOCUMENTS_FILE = "documents.jsonl"
_LOG_FILE = "fetch_log.jsonl"

_JSON_KW: dict[str, Any] = {
    "sort_keys": True,
    "ensure_ascii": False,
    "separators": (",", ":"),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# The pure part: no I/O, no randomness, no globals.
# --------------------------------------------------------------------------


def select_eval(
    candidates: Iterable[dict[str, Any]],
    *,
    total_pool_tokens: int,
    target_fraction: float = EVAL_TARGET_FRACTION,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    """Choose which candidate ids form the eval split.

    `candidates` are dicts carrying id, source_id, lane, provenance_tier and
    est_tokens. Every one must already be eval-eligible; a candidate outside
    EVAL_TIERS raises rather than being silently dropped, because a caller
    that hands over a T2 document has a bug worth surfacing.

    Selection order is sha256(f"{seed}:{id}") — spread across sources without
    an RNG. Returned ids are sorted by id, so the output is stable and
    readable regardless of the order they were chosen in.
    """
    items = tuple(candidates)

    seen: set[str] = set()
    for index, candidate in enumerate(items):
        doc_id = candidate.get("id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"candidate[{index}]: id must be a non-empty string")
        if doc_id in seen:
            raise ValueError(f"duplicate candidate id: {doc_id!r}")
        seen.add(doc_id)

        tier = candidate.get("provenance_tier")
        if tier not in EVAL_TIERS:
            raise ValueError(
                f"candidate {doc_id!r}: provenance_tier {tier!r} is not "
                f"eval-eligible; rule 3 allows only {sorted(EVAL_TIERS)}"
            )

        tokens = candidate.get("est_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError(
                f"candidate {doc_id!r}: est_tokens must be a positive int, "
                f"got {tokens!r}"
            )

    target_tokens = round(target_fraction * total_pool_tokens)
    candidate_tokens = sum(int(c["est_tokens"]) for c in items)

    ordered = sorted(items, key=lambda c: _sha(f"{seed}:{c['id']}"))

    chosen: list[dict[str, Any]] = []
    selected_tokens = 0
    for candidate in ordered:
        if selected_tokens >= target_tokens:
            break
        chosen.append(candidate)
        selected_tokens += int(candidate["est_tokens"])

    eval_ids = tuple(sorted(c["id"] for c in chosen))
    return {
        "eval_ids": eval_ids,
        "selected_tokens": selected_tokens,
        "target_tokens": target_tokens,
        "candidate_tokens": candidate_tokens,
        "seed": seed,
        "fingerprint": _sha("\n".join(eval_ids)),
    }


# --------------------------------------------------------------------------
# The I/O part.
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pool_tokens(raw: Path) -> int:
    """Total tokens fetched, summed over every record in the fetch log."""
    return sum(int(r.get("est_tokens", 0)) for r in _read_jsonl(raw / _LOG_FILE))


def carve_eval(
    *,
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    """Record which fetched documents form the eval split.

    Reads the raw root and writes only under `eval_root`. `sources` may be a
    SUBSET of the manifest -- `eval_eligible` filters by tier and does not
    require a complete manifest, so `validate_sources` is deliberately not
    called here.
    """
    raw, out = Path(raw_root), Path(eval_root)
    eligible = sorted(s.source_id for s in eval_eligible(sources))

    # Pass 1: metadata only. Text is read to size it, then dropped, so peak
    # memory tracks the number of documents rather than the corpus bytes.
    candidates: list[dict[str, Any]] = []
    for source_id in eligible:
        for doc in _read_jsonl(raw / source_id / _DOCUMENTS_FILE):
            candidates.append(
                {
                    "id": doc["id"],
                    "source_id": source_id,
                    "lane": doc["lane"],
                    "provenance_tier": doc["provenance_tier"],
                    "est_tokens": estimate_tokens(doc["text"]),
                }
            )

    total_pool_tokens = _pool_tokens(raw)
    selection = select_eval(
        candidates,
        total_pool_tokens=total_pool_tokens,
        seed=seed,
    )
    eval_ids = set(selection["eval_ids"])
    by_id = {c["id"]: c for c in candidates}

    manifest_path = out / _MANIFEST_FILE
    summary = {
        "eval_root": str(out),
        "manifest": str(manifest_path),
        "seed": seed,
        "target_fraction": EVAL_TARGET_FRACTION,
        "total_pool_tokens": total_pool_tokens,
        "candidate_tokens": selection["candidate_tokens"],
        "target_tokens": selection["target_tokens"],
        "selected_tokens": selection["selected_tokens"],
        "selected_count": len(selection["eval_ids"]),
        "fingerprint": selection["fingerprint"],
        "sources": eligible,
    }

    # Idempotent: an existing manifest whose header fingerprint matches
    # describes exactly this selection, so there is nothing to rewrite.
    if manifest_path.exists():
        existing = _read_jsonl(manifest_path)
        if existing and existing[0].get("fingerprint") == selection["fingerprint"]:
            return {**summary, "cached": True}

    out.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        header = {
            "kind": "header",
            "seed": seed,
            "target_fraction": EVAL_TARGET_FRACTION,
            "target_tokens": selection["target_tokens"],
            "candidate_tokens": selection["candidate_tokens"],
            "selected_tokens": selection["selected_tokens"],
            "selected_count": len(selection["eval_ids"]),
            "fingerprint": selection["fingerprint"],
        }
        handle.write(json.dumps(header, **_JSON_KW) + "\n")
        for doc_id in selection["eval_ids"]:
            candidate = by_id[doc_id]
            handle.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "source_id": candidate["source_id"],
                        "lane": candidate["lane"],
                        "provenance_tier": candidate["provenance_tier"],
                        "split": "eval",
                        "est_tokens": candidate["est_tokens"],
                    },
                    **_JSON_KW,
                )
                + "\n"
            )

    # Pass 2: re-read the raw files and copy out only the chosen documents,
    # with split flipped. The raw files are opened read-only and never touched.
    written = 0
    for source_id in eligible:
        rows = [
            doc
            for doc in _read_jsonl(raw / source_id / _DOCUMENTS_FILE)
            if doc["id"] in eval_ids
        ]
        if not rows:
            continue
        source_dir = out / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        with (source_dir / _DOCUMENTS_FILE).open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for doc in rows:
                # Rule 3 again, explicitly. validate_document below checks that
                # the tier is one of the four legal strings -- it does NOT
                # check eval-eligibility, so a T3 document would pass it.
                if doc["provenance_tier"] not in EVAL_TIERS:
                    raise ValueError(
                        f"{doc['id']}: tier {doc['provenance_tier']!r} reached the "
                        f"eval writer; rule 3 allows only {sorted(EVAL_TIERS)}"
                    )
                document = Document(
                    id=doc["id"],
                    lane=doc["lane"],
                    provenance_tier=doc["provenance_tier"],
                    split="eval",
                    source=doc["source"],
                    text=doc["text"],
                )
                validate_document(document)
                handle.write(
                    json.dumps(
                        {
                            "id": document.id,
                            "lane": document.lane,
                            "provenance_tier": document.provenance_tier,
                            "split": document.split,
                            "source": document.source,
                            "text": document.text,
                        },
                        **_JSON_KW,
                    )
                    + "\n"
                )
                written += 1

    return {**summary, "documents_written": written, "cached": False}
