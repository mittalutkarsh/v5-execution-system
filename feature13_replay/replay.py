"""Feature 13 — replay (epics 13.1-13.3).

Given only the seed and the consumption ledger, replay any interval [a, b): the
batch stream reconstructs each batch's id, sequence indices and content hash
from (seed, index) and they must match the original ledger exactly. This is the
governing invariant made checkable after the fact.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["replay_interval"]


def replay_interval(
    stream: Any, ledger_rows: Sequence[dict[str, Any]], a: int, b: int
) -> dict[str, Any]:
    """Reconstruct batches [a, b) and compare to the recorded ledger rows."""
    by_index = {row["batch_index"]: row for row in ledger_rows}
    mismatches: list[int] = []
    checked = 0
    for i in range(a, b):
        original = by_index.get(i)
        if original is None:
            mismatches.append(i)
            continue
        rebuilt = stream.batch(i)
        if (rebuilt.content_hash != original["content_hash"]
                or list(rebuilt.seq_indices) != original["seq_indices"]):
            mismatches.append(i)
        checked += 1
    return {
        "interval": [a, b],
        "checked": checked,
        "matched": not mismatches,
        "mismatches": mismatches,
    }
