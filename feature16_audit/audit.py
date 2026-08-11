"""Feature 16 — audit + evidence (epics 16.1-16.4).

The capstone cross-checks the whole run against its own artifacts -- nothing is
hardcoded, every number is read back from a file that an earlier stage wrote:

  * audit (16.1): shards re-verify, tokenizer re-verifies, checkpoint restores,
    the firewall partition is disjoint, and every learning-ledger batch id
    appears in the consumption ledger.
  * log completeness (16.4): every expected [PASS] event was actually emitted.
  * evidence.json (16.2) + evidence.md (16.3): a summary assembled from the real
    manifests, ledgers and reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature3_tokenizer.tokenizer_build import verify_frozen
from feature4_shards.shard_index import load_index, verify_shards
from feature5_firewall.firewall import eval_shards_blocked
from feature11_checkpoint.checkpoint import verify_checkpoint

__all__ = ["EXPECTED_PASS_EVENTS", "run_audit", "build_evidence", "write_evidence_md"]

EXPECTED_PASS_EVENTS = [
    "corpus_loaded", "corpus_cleaned", "tokenizer_frozen", "shards_written",
    "eval_shard_blocked", "mixture_compiled", "opus_selected", "sequences_packed",
    "batch_stream_ready", "trained", "contrastive_delta_s", "checkpoint_saved",
    "resume_next_batch_matched", "replay_hash_matched", "fork_lineage_recorded",
    "throughput_measured",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_audit(
    *,
    emitted_pass_events: list[str],
    tokenizer_dir: str,
    shard_root: str,
    checkpoint_dir: str,
    manifests: str,
) -> dict[str, Any]:
    """Cross-check the run against its artifacts. Returns {check: bool, ...}."""
    m = Path(manifests)
    index = load_index(shard_root)
    consumption = {r["batch_id"] for r in _read_jsonl(m / "consumption_ledger.jsonl")}
    learning = _read_jsonl(m / "learning_ledger.jsonl")

    checks = {
        "shards_verify": verify_shards(shard_root)["ok"],
        "tokenizer_verify": verify_frozen(tokenizer_dir),
        "checkpoint_verify": verify_checkpoint(checkpoint_dir),
        "firewall_disjoint": eval_shards_blocked(index)["ok"],
        "ledger_consistent": all(r["batch_id"] in consumption for r in learning) and bool(learning),
        "log_complete": all(e in emitted_pass_events for e in EXPECTED_PASS_EVENTS),
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def build_evidence(
    *, tokenizer_dir: str, shard_root: str, checkpoint_dir: str, manifests: str, audit: dict[str, Any],
) -> dict[str, Any]:
    """Assemble evidence.json from the real artifacts (no hardcoded values)."""
    m = Path(manifests)
    tok_manifest = _read_json(Path(tokenizer_dir) / "tokenizer_manifest.json")
    index = load_index(shard_root)
    mixture = _read_json(m / "mixture_plan.json")
    learning = _read_jsonl(m / "learning_ledger.jsonl")
    consumption = _read_jsonl(m / "consumption_ledger.jsonl")
    ckpt = _read_json(Path(checkpoint_dir) / "checkpoint_manifest.json")
    fork = _read_json(m / "fork_lineage.json")
    packing = _read_json(m / "packing_report.json")
    delta = _read_json(m / "contrastive_delta_s.json")

    return {
        "kind": "evidence",
        "audit": audit,
        "tokenizer": {"hash": tok_manifest.get("hash"), "vocab_size": tok_manifest.get("vocab_size"),
                      "n_merges": tok_manifest.get("n_merges")},
        "shards": {"n_shards": index["n_shards"], "total_tokens": index["total_tokens"],
                   "by_split": index["by_split"]},
        "mixture": {"all_floors_met": mixture.get("all_floors_met"),
                    "lane_totals": mixture.get("lane_totals")},
        "training": {
            "steps": len(learning),
            "batches_consumed": len(consumption),
            "first_loss": learning[0]["loss_nats"] if learning else None,
            "last_loss": learning[-1]["loss_nats"] if learning else None,
        },
        "checkpoint": {"model_hash": ckpt.get("model_hash"), "ledger_offset": ckpt.get("ledger_offset")},
        "fork": {"branch_id": fork.get("branch_id"), "diverged": fork.get("diverged")},
        "packing": {"efficiency": packing.get("packing_efficiency")},
        "contrastive": {"n_pairs": len(delta.get("pairs", []))},
    }


def write_evidence_md(evidence: dict[str, Any], path: str) -> None:
    """A short human-readable evidence summary (16.3)."""
    a = evidence["audit"]
    lines = [
        "# Evidence — Training Data Execution System",
        "",
        f"**Audit:** {'ALL CHECKS PASSED' if a['all_passed'] else 'FAILURES PRESENT'}",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, ok in a["checks"].items():
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    tok, sh, tr = evidence["tokenizer"], evidence["shards"], evidence["training"]
    lines += [
        "",
        "## Artifacts",
        f"- Tokenizer: vocab {tok['vocab_size']}, {tok['n_merges']} merges, hash `{str(tok['hash'])[:16]}…`",
        f"- Shards: {sh['n_shards']} shards, {sh['total_tokens']:,} tokens",
        f"- Training: {tr['steps']} steps, loss {tr['first_loss']} → {tr['last_loss']}",
        f"- Checkpoint offset: {evidence['checkpoint']['ledger_offset']}; "
        f"fork `{evidence['fork']['branch_id']}` diverged={evidence['fork']['diverged']}",
        f"- Packing efficiency: {evidence['packing']['efficiency']}; "
        f"contrastive pairs: {evidence['contrastive']['n_pairs']}",
        "",
        "_Every number above is read back from a written artifact; none is hardcoded._",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
