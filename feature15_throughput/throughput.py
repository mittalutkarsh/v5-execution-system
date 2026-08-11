"""Feature 15 — throughput / packing efficiency (epics 15.1-15.3).

Two kinds of number, kept apart on purpose:

  * DETERMINISTIC efficiency metrics -- packing utilization (real tokens /
    capacity) and loss-bearing token count -- which are safe to log because they
    never vary between runs.
  * WALL-CLOCK throughput (loss-bearing tokens / second), which is
    machine-dependent, so it is written to performance.json only and never put
    in run.log (that would break the byte-identical guarantee).
"""

from __future__ import annotations

import time
from typing import Any, Callable

from feature10_trainer.trainer import batch_tensors

__all__ = ["packing_utilization", "measure_throughput", "build_performance"]


def packing_utilization(packing_report: dict[str, Any]) -> float:
    """Real tokens / capacity (15.1) -- deterministic."""
    return packing_report["packing_efficiency"]


def measure_throughput(make_trainer: Callable[[], Any], stream: Any, *, n_steps: int) -> dict[str, Any]:
    """Time `n_steps` training steps; report loss-bearing tokens per second (15.2)."""
    trainer = make_trainer()
    loss_tokens = 0
    start = time.perf_counter()
    for i in range(n_steps):
        batch = stream.batch(i)
        tokens, pos, allowed, lm = batch_tensors(stream, batch)
        trainer.train_step(tokens, pos, allowed, lm)
        loss_tokens += int(lm.sum())
    seconds = time.perf_counter() - start
    return {
        "steps": n_steps,
        "loss_bearing_tokens": loss_tokens,
        "seconds": round(seconds, 4),
        "loss_tokens_per_sec": round(loss_tokens / seconds, 2) if seconds > 0 else 0.0,
    }


def build_performance(packing_report: dict[str, Any], throughput: dict[str, Any]) -> dict[str, Any]:
    """Assemble performance.json (15.3)."""
    return {
        "kind": "performance",
        "packing_efficiency": round(packing_report["packing_efficiency"], 4),
        "loss_positions_total": packing_report["loss_positions"],
        "real_tokens": packing_report["real_tokens"],
        "padding_tokens": packing_report["padding_tokens"],
        "throughput": throughput,
    }
