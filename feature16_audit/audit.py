"""Feature 16 — audit + evidence (epics 16.1-16.4).

The capstone cross-checks the whole run against its own artifacts -- nothing is
hardcoded, every number is read back from a file that an earlier stage wrote:

  * audit (16.1): shards re-verify, tokenizer re-verifies, checkpoint restores,
    the firewall partition is disjoint, and every learning-ledger batch id
    appears in the consumption ledger.
  * log completeness (16.4): every expected [PASS] event was actually emitted.
  * evidence.json (16.2): per-requirement PASS/FAIL with a pointer to the file
    or event that proves it.
  * evidence.md (16.3): the human-readable Requirement / Result / Evidence table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature3_tokenizer.tokenizer_build import verify_frozen
from feature4_shards.shard_index import load_index, verify_shards
from feature5_firewall.firewall import eval_shards_blocked
from feature11_checkpoint.checkpoint import verify_checkpoint

__all__ = ["EXPECTED_PASS_EVENTS", "REQUIREMENTS", "run_audit", "build_evidence", "write_evidence_md"]

EXPECTED_PASS_EVENTS = [
    "corpus_loaded", "corpus_cleaned", "tokenizer_hash_verified", "tokenizer_frozen",
    "manifests_validated", "shards_written", "eval_shard_blocked", "mixture_compiled",
    "opus_selected", "sequences_packed", "batch_stream_ready", "trained",
    "contrastive_delta_s", "checkpoint_saved", "resume_next_batch_matched",
    "replay_hash_matched", "fork_lineage_recorded", "throughput_measured",
]

# requirement label -> (evidence description, evidence location)
REQUIREMENTS = [
    ("Tokenizer integrity", "Manifest record", "manifests/tokenizer_manifest.json"),
    ("Evaluation firewall", "Blocked-shard event", "run.log ([PASS] eval_shard_blocked)"),
    ("Packing correctness", "Packed-batch report", "manifests/packing_report.json"),
    ("Mixture compliance", "Planned versus actual shares", "manifests/mixture_plan.json"),
    ("OPUS audit trail", "Candidate decision records", "ledgers/opus_decision_ledger.jsonl"),
    ("Crash recovery", "Expected and resumed batch ids", "run.log ([PASS] resume_next_batch_matched)"),
    ("Replay", "Original and replay hashes", "run.log ([PASS] replay_hash_matched)"),
    ("Learning trace", "Loss linked to source data", "ledgers/learning_ledger.jsonl"),
    ("Throughput", "Performance report", "performance.json"),
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_audit(
    *, emitted_pass_events: list[str], tokenizer_dir: str, shard_root: str,
    checkpoint_dir: str, manifests: str, ledgers: str,
) -> dict[str, Any]:
    """Cross-check the run against its artifacts. Returns {checks, all_passed, emitted}."""
    led = Path(ledgers)
    index = load_index(shard_root)
    consumption = {r["batch_id"] for r in _read_jsonl(led / "consumption_ledger.jsonl")}
    learning = _read_jsonl(led / "learning_ledger.jsonl")

    checks = {
        "shards_verify": verify_shards(shard_root)["ok"],
        "tokenizer_verify": verify_frozen(tokenizer_dir),
        "checkpoint_verify": verify_checkpoint(checkpoint_dir),
        "firewall_disjoint": eval_shards_blocked(index)["ok"],
        "ledger_consistent": bool(learning) and all(r["batch_id"] in consumption for r in learning),
        "log_complete": all(e in emitted_pass_events for e in EXPECTED_PASS_EVENTS),
    }
    return {"checks": checks, "all_passed": all(checks.values()), "emitted": list(emitted_pass_events)}


def _requirement_results(
    *, audit: dict[str, Any], manifests: Path, ledgers: Path, evidence_dir: Path,
) -> dict[str, bool]:
    checks, emitted = audit["checks"], audit["emitted"]
    packing = _read_json(manifests / "packing_report.json")
    mixture = _read_json(manifests / "mixture_plan.json")
    opus = _read_jsonl(ledgers / "opus_decision_ledger.jsonl")
    learning = _read_jsonl(ledgers / "learning_ledger.jsonl")
    performance = _read_json(evidence_dir / "performance.json")
    return {
        "Tokenizer integrity": checks["tokenizer_verify"],
        "Evaluation firewall": checks["firewall_disjoint"] and "eval_shard_blocked" in emitted,
        "Packing correctness": packing.get("packing_efficiency", 0) > 0,
        "Mixture compliance": bool(mixture.get("all_floors_met")),
        "OPUS audit trail": len(opus) > 0,
        "Crash recovery": checks["checkpoint_verify"] and "resume_next_batch_matched" in emitted,
        "Replay": "replay_hash_matched" in emitted,
        "Learning trace": checks["ledger_consistent"] and len(learning) > 0,
        "Throughput": bool(performance) and "throughput_measured" in emitted,
    }


def build_evidence(
    *, tokenizer_dir: str, shard_root: str, checkpoint_dir: str,
    manifests: str, ledgers: str, evidence_dir: str, audit: dict[str, Any],
) -> dict[str, Any]:
    """Assemble evidence.json from the real artifacts (no hardcoded values)."""
    man, led, ev = Path(manifests), Path(ledgers), Path(evidence_dir)
    results = _requirement_results(audit=audit, manifests=man, ledgers=led, evidence_dir=ev)
    requirements = {
        label: {"result": "PASS" if results[label] else "FAIL",
                "evidence": desc, "location": loc}
        for label, desc, loc in REQUIREMENTS
    }

    tok_manifest = _read_json(Path(tokenizer_dir) / "tokenizer_manifest.json")
    index = load_index(shard_root)
    mixture = _read_json(man / "mixture_plan.json")
    learning = _read_jsonl(led / "learning_ledger.jsonl")
    consumption = _read_jsonl(led / "consumption_ledger.jsonl")
    ckpt = _read_json(Path(checkpoint_dir) / "checkpoint_manifest.json")
    fork = _read_json(man / "fork_lineage.json")
    packing = _read_json(man / "packing_report.json")
    delta = _read_json(man / "contrastive_delta_s.json")

    return {
        "kind": "evidence",
        "all_passed": audit["all_passed"] and all(results.values()),
        "requirements": requirements,
        "audit_checks": audit["checks"],
        "tokenizer": {"hash": tok_manifest.get("hash"), "vocab_size": tok_manifest.get("vocab_size"),
                      "n_merges": tok_manifest.get("n_merges")},
        "shards": {"n_shards": index["n_shards"], "total_tokens": index["total_tokens"],
                   "by_split": index["by_split"]},
        "mixture": {"all_floors_met": mixture.get("all_floors_met"),
                    "lane_totals": mixture.get("lane_totals")},
        "training": {
            "steps": len(learning), "batches_consumed": len(consumption),
            "first_loss": learning[0]["loss_nats"] if learning else None,
            "last_loss": learning[-1]["loss_nats"] if learning else None,
        },
        "checkpoint": {"model_hash": ckpt.get("model_hash"), "ledger_offset": ckpt.get("ledger_offset")},
        "fork": {"branch_id": fork.get("branch_id"), "diverged": fork.get("diverged")},
        "packing": {"efficiency": packing.get("packing_efficiency")},
        "contrastive": {"n_pairs": len(delta.get("pairs", []))},
    }


def write_evidence_md(evidence: dict[str, Any], path: str) -> None:
    """The required Requirement / Result / Evidence summary table (16.3)."""
    lines = [
        "# Evidence — Training Data Execution System",
        "",
        f"**Overall: {'ALL REQUIREMENTS PASS' if evidence['all_passed'] else 'FAILURES PRESENT'}**",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
    ]
    for label, _desc, _loc in REQUIREMENTS:
        req = evidence["requirements"][label]
        lines.append(f"| {label} | {req['result']} | {req['evidence']} — `{req['location']}` |")
    tok, sh, tr = evidence["tokenizer"], evidence["shards"], evidence["training"]
    lines += [
        "",
        "## Run summary",
        f"- Tokenizer: vocab {tok['vocab_size']}, {tok['n_merges']} merges, hash `{str(tok['hash'])[:16]}…`",
        f"- Shards: {sh['n_shards']} shards, {sh['total_tokens']:,} tokens ({sh['by_split']})",
        f"- Training: {tr['steps']} steps, loss {tr['first_loss']} → {tr['last_loss']} "
        f"({tr['batches_consumed']} batches consumed)",
        f"- Checkpoint offset: {evidence['checkpoint']['ledger_offset']}; "
        f"fork `{evidence['fork']['branch_id']}` diverged={evidence['fork']['diverged']}",
        f"- Packing efficiency: {evidence['packing']['efficiency']}; "
        f"contrastive pairs: {evidence['contrastive']['n_pairs']}",
        "",
        "_Every value above is read back from a written artifact; none is hardcoded._",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
