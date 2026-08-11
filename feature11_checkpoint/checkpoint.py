"""Feature 11 — checkpoints (epics 11.1-11.4).

A checkpoint captures everything needed to continue a run exactly: the model
weights, the optimizer state, the torch RNG snapshot, and -- crucially -- the
ledger offset (how many batches were consumed). Because a batch is a pure
function of (seed, index), the offset alone pins the data position, so
model + optimizer + rng + offset is a complete, restartable state. Each
checkpoint carries a manifest with a canonical model hash so a restore can be
verified (11.2), and load_checkpoint rebuilds an identical trainer (11.3).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from feature10_trainer.moe_model import ModelConfig
from feature10_trainer.trainer import Trainer

__all__ = ["model_hash", "save_checkpoint", "load_checkpoint", "verify_checkpoint", "MANIFEST_FILE"]

MANIFEST_FILE = "checkpoint_manifest.json"
_JSON = {"ensure_ascii": False, "sort_keys": True, "indent": 2}


def model_hash(model: torch.nn.Module) -> str:
    """A canonical sha256 over the model's parameters (order-independent)."""
    h = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        h.update(name.encode("utf-8"))
        h.update(state[name].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(
    trainer: Trainer, *, step: int, ledger_offset: int, seed: str, out_dir: str,
) -> dict[str, Any]:
    """Write model / optimizer / rng / manifest to out_dir; return the manifest."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.model.state_dict(), d / "model.pt")
    torch.save(trainer.opt.state_dict(), d / "optim.pt")
    torch.save(torch.get_rng_state(), d / "rng.pt")
    manifest = {
        "kind": "checkpoint_manifest",
        "step": step,
        "ledger_offset": ledger_offset,
        "seed": seed,
        "config": dataclasses.asdict(trainer.model.cfg),
        "model_hash": model_hash(trainer.model),
        "n_params": trainer.model.n_params(),
    }
    with (d / MANIFEST_FILE).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, **_JSON) + "\n")
    return manifest


def load_checkpoint(out_dir: str) -> tuple[Trainer, dict[str, Any]]:
    """Rebuild a trainer with the exact saved model / optimizer / rng state."""
    d = Path(out_dir)
    meta = json.loads((d / MANIFEST_FILE).read_text(encoding="utf-8"))
    cfg = ModelConfig(**meta["config"])
    trainer = Trainer(cfg, seed=meta["seed"])
    trainer.model.load_state_dict(torch.load(d / "model.pt", weights_only=True))
    trainer.opt.load_state_dict(torch.load(d / "optim.pt", weights_only=True))
    torch.set_rng_state(torch.load(d / "rng.pt", weights_only=True))
    return trainer, meta


def verify_checkpoint(out_dir: str) -> bool:
    """True iff the restored model's hash matches the checkpoint manifest."""
    trainer, meta = load_checkpoint(out_dir)
    return model_hash(trainer.model) == meta["model_hash"]
