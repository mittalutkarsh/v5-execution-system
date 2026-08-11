"""Feature 9 — the deterministic RNG (epic 9.1). The seed discipline is born here.

The governing invariant of the whole system is that a seed plus a ledger offset
reconstructs any batch byte-for-byte. We buy that by making batch `i` a PURE
function of (master_seed, i): each batch gets its own generator seeded by a
hash of the master seed and the batch index, so batch 900 can be rebuilt without
replaying batches 0..899. That is what makes resume, replay and fork exact.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["derive_seed", "batch_rng", "MASTER_SEED"]

MASTER_SEED = "v5-batch-stream-2026"


def derive_seed(master_seed: str, index: int) -> int:
    """A stable 64-bit seed for batch `index`, portable across machines."""
    digest = hashlib.sha256(f"{master_seed}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def batch_rng(master_seed: str, index: int) -> np.random.Generator:
    """A fresh numpy Generator dedicated to batch `index` (PCG64, deterministic)."""
    return np.random.default_rng(derive_seed(master_seed, index))
