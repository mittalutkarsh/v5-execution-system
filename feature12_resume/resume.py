"""Feature 12 — crash + resume (epics 12.1-12.4).

A run is checkpointed, then deliberately crashed at a set batch. Resuming loads
the checkpoint, reads the ledger offset, and continues from exactly that batch --
no batch skipped, none repeated. Because the loss trajectory of the resumed tail
must match a clean run's tail, this proves the checkpoint restored model,
optimizer and RNG exactly, not just the data position.
"""

from __future__ import annotations

from typing import Any, Callable

from feature10_trainer.trainer import Trainer, batch_tensors
from feature11_checkpoint.checkpoint import load_checkpoint, save_checkpoint

__all__ = ["CrashError", "train_range", "crash_and_resume"]


class CrashError(Exception):
    """Raised by the deliberate crash hook at a chosen batch (12.1)."""


def train_range(trainer: Trainer, stream: Any, start: int, end: int) -> list[dict[str, Any]]:
    """Train batches [start, end); return per-step {index, batch_id, loss}."""
    rows: list[dict[str, Any]] = []
    for i in range(start, end):
        batch = stream.batch(i)
        tokens, pos, allowed, lm = batch_tensors(stream, batch)
        loss = trainer.train_step(tokens, pos, allowed, lm)
        rows.append({"index": i, "batch_id": batch.batch_id, "loss": round(loss, 6)})
    return rows


def crash_and_resume(
    make_trainer: Callable[[], Trainer],
    stream: Any,
    *,
    total: int,
    crash_at: int,
    checkpoint_dir: str,
    seed: str,
    crash: bool = True,
) -> dict[str, Any]:
    """Run clean vs crash-then-resume and prove the two trajectories match."""
    # 1. clean reference run
    clean = train_range(make_trainer(), stream, 0, total)

    # 2. run up to the crash point, checkpoint, then crash
    t = make_trainer()
    before = train_range(t, stream, 0, crash_at)
    save_checkpoint(t, step=crash_at, ledger_offset=crash_at, seed=seed, out_dir=checkpoint_dir)
    if crash:
        try:
            raise CrashError(f"deliberate crash at batch {crash_at}")
        except CrashError:
            pass  # a real process would die here; we caught it to continue the demo

    # 3. resume from the checkpoint and finish
    resumed_trainer, meta = load_checkpoint(checkpoint_dir)
    offset = meta["ledger_offset"]
    next_expected = stream.batch(offset)
    after = train_range(resumed_trainer, stream, offset, total)

    combined = before + after
    return {
        "resume_offset": offset,
        "next_batch_id": next_expected.batch_id,
        "no_skip_or_repeat": [r["index"] for r in combined] == list(range(total)),
        "loss_trajectory_matched": [r["loss"] for r in combined] == [r["loss"] for r in clean],
    }
