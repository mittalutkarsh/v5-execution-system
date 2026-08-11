"""Feature 5 — the evaluation firewall (epics 5.1-5.3).

Eval shards are quarantined: their ids may NEVER enter a training batch. The
firewall is the single gate every shard id must pass through on its way into the
batch stream. It exposes the train-only pool (Epic 5.2) and an admit() that
raises on an eval shard, and an audit() that proves the two sets are disjoint
(Epic 5.3). It is a hard boundary, enforced, not a convention.
"""

from __future__ import annotations

from typing import Any

__all__ = ["FirewallViolation", "EvalFirewall", "eval_shards_blocked"]


class FirewallViolation(Exception):
    """Raised when an eval (quarantined) shard is offered to a training batch."""


class EvalFirewall:
    """Partitions a shard index into train vs eval and gates admission to train."""

    def __init__(self, index: dict[str, Any]) -> None:
        self.eval_ids = frozenset(
            s["shard_id"] for s in index["shards"] if s["split"] == "eval"
        )
        # preserve index order for the train pool (deterministic)
        self.train_ids = tuple(
            s["shard_id"] for s in index["shards"] if s["split"] == "train"
        )

    def is_eval(self, shard_id: str) -> bool:
        return shard_id in self.eval_ids

    def admit(self, shard_id: str) -> str:
        """Return shard_id if it may train; raise FirewallViolation if it is eval."""
        if shard_id in self.eval_ids:
            raise FirewallViolation(
                f"eval shard {shard_id!r} may not enter a training batch"
            )
        return shard_id

    def train_pool(self) -> tuple[str, ...]:
        """The only shard ids a training batch is allowed to draw from."""
        return self.train_ids

    def audit(self) -> dict[str, Any]:
        """Prove the partition: train and eval are disjoint and fully covered."""
        overlap = sorted(set(self.train_ids) & self.eval_ids)
        return {
            "train_shards": len(self.train_ids),
            "eval_shards": len(self.eval_ids),
            "disjoint": not overlap,
            "overlap": overlap,
            "eval_ids": sorted(self.eval_ids),
        }


def eval_shards_blocked(index: dict[str, Any]) -> dict[str, Any]:
    """Confirm every eval shard is blocked and none leaks into the train pool.

    Returns {blocked, train_shards, ok}; ok is False if any eval id would be
    admitted or appears in the train pool.
    """
    fw = EvalFirewall(index)
    blocked = 0
    for shard_id in fw.eval_ids:
        try:
            fw.admit(shard_id)
        except FirewallViolation:
            blocked += 1
    audit = fw.audit()
    return {
        "blocked": blocked,
        "eval_shards": audit["eval_shards"],
        "train_shards": audit["train_shards"],
        "ok": blocked == audit["eval_shards"] and audit["disjoint"],
    }
