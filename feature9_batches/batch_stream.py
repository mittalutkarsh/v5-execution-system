"""Feature 9 — the batch stream + consumption ledger (epics 9.2-9.5).

A BatchStream turns the packed sequences into an endless, mixture-weighted,
fully reproducible stream of batches. batch(i) draws `batch_size` sequences: for
each slot it picks a lane by the mixture probabilities, then a sequence from
that lane -- all from batch `i`'s dedicated generator, so batch(i) depends only
on (seed, i) and the fixed pool. Each batch carries a content hash over its
token bytes (9.3). A ConsumptionLedger records every consumed batch, append-only
(9.4); seed + a ledger offset then reconstructs any batch and its hash (9.5).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature9_batches.rng import MASTER_SEED, batch_rng

__all__ = ["Batch", "BatchStream", "ConsumptionLedger", "verify_reconstruction"]


@dataclass(frozen=True)
class Batch:
    index: int
    seq_indices: tuple[int, ...]
    batch_id: str
    content_hash: str
    n_tokens: int
    n_loss_positions: int

    def as_ledger_row(self) -> dict[str, Any]:
        return {
            "batch_index": self.index,
            "batch_id": self.batch_id,
            "content_hash": self.content_hash,
            "seq_indices": list(self.seq_indices),
            "n_sequences": len(self.seq_indices),
            "n_tokens": self.n_tokens,
            "n_loss_positions": self.n_loss_positions,
        }


class BatchStream:
    """A reproducible, mixture-weighted stream over a fixed pool of sequences."""

    def __init__(
        self,
        sequences: Sequence[Any],
        *,
        seed: str = MASTER_SEED,
        batch_size: int = 8,
        lane_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.sequences = sequences
        self.seed = seed
        self.batch_size = batch_size
        self._lane_weights = lane_weights

        by_lane: dict[str, list[int]] = {}
        for j, s in enumerate(sequences):
            by_lane.setdefault(s.lane, []).append(j)
        self.by_lane = by_lane
        self.lanes = sorted(by_lane)

        if lane_weights is None:                      # weight lanes by their size
            weights = {lane: float(len(v)) for lane, v in by_lane.items()}
        else:                                         # restrict to present lanes
            weights = {lane: max(0.0, float(lane_weights.get(lane, 0.0))) for lane in self.lanes}
            if sum(weights.values()) <= 0:            # fall back to uniform if empty
                weights = {lane: 1.0 for lane in self.lanes}
        total = sum(weights[lane] for lane in self.lanes)
        self.probs = np.array([weights[lane] / total for lane in self.lanes])

    def __len__(self) -> int:
        return len(self.sequences)

    def reseed(self, seed: str) -> "BatchStream":
        """A new stream over the same pool + mixture but a different seed (for forks)."""
        return BatchStream(
            self.sequences, seed=seed, batch_size=self.batch_size,
            lane_weights=self._lane_weights,
        )

    def batch(self, index: int) -> Batch:
        """Reconstruct batch `index` — a pure function of (seed, index, pool)."""
        rng = batch_rng(self.seed, index)
        chosen: list[int] = []
        for _ in range(self.batch_size):
            lane = self.lanes[int(rng.choice(len(self.lanes), p=self.probs))]
            pool = self.by_lane[lane]
            chosen.append(int(pool[int(rng.integers(len(pool)))]))
        tokens = np.array([self.sequences[j].tokens for j in chosen], dtype=np.uint16)
        h = hashlib.sha256(tokens.tobytes()).hexdigest()
        n_loss = sum(self.sequences[j].n_loss_positions() for j in chosen)
        return Batch(index, tuple(chosen), h[:16], h, int(tokens.size), int(n_loss))

    def tokens_of(self, batch: Batch) -> np.ndarray:
        """Materialize the [batch_size, seq_len] token matrix for a batch."""
        return np.array([self.sequences[j].tokens for j in batch.seq_indices], dtype=np.int64)


class ConsumptionLedger:
    """Append-only record of which batches were consumed, in order."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self._fh = self.path.open("w", encoding="utf-8", newline="\n")

    def record(self, batch: Batch) -> None:
        row = batch.as_ledger_row()
        self.rows.append(row)
        self._fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    @property
    def offset(self) -> int:
        """How many batches have been consumed (the next batch index)."""
        return len(self.rows)


def verify_reconstruction(stream: BatchStream, ledger_rows: Sequence[dict[str, Any]]) -> bool:
    """seed + offset proof: recompute each recorded batch and match its hash."""
    for row in ledger_rows:
        b = stream.batch(row["batch_index"])
        if b.content_hash != row["content_hash"] or list(b.seq_indices) != row["seq_indices"]:
            return False
    return True
