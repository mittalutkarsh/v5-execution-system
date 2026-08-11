"""Feature 14 — fork (epics 14.1-14.3).

Fork restores an earlier checkpoint and starts a NEW branch: same shared history
up to the fork point, then a divergent continuation driven by a different seed.
The branch records its lineage (parent checkpoint hash, step, ledger offset,
branch id, seed) so any run can be traced back to where it split off.
"""

from __future__ import annotations

from typing import Any

from feature11_checkpoint.checkpoint import load_checkpoint
from feature12_resume.resume import train_range

__all__ = ["fork_run"]


def fork_run(
    checkpoint_dir: str,
    parent_stream: Any,
    *,
    branch_id: str,
    fork_seed: str,
    steps: int,
) -> dict[str, Any]:
    """Fork from a checkpoint onto a new-seed branch; return the lineage record."""
    trainer, meta = load_checkpoint(checkpoint_dir)
    offset = meta["ledger_offset"]

    branch_stream = parent_stream.reseed(fork_seed)
    branch_rows = train_range(trainer, branch_stream, offset, offset + steps)

    # divergence proof: at the fork point the branch draws a different batch than
    # the parent would have (different seed -> different sampling)
    diverged = (branch_stream.batch(offset).content_hash
                != parent_stream.batch(offset).content_hash)

    return {
        "kind": "fork_lineage",
        "branch_id": branch_id,
        "seed": fork_seed,
        "diverged_at": offset,
        "diverged": diverged,
        "steps": steps,
        "parent": {
            "checkpoint_hash": meta["model_hash"],
            "step": meta["step"],
            "ledger_offset": offset,
            "seed": meta["seed"],
        },
        "branch_batch_ids": [r["batch_id"] for r in branch_rows],
    }
