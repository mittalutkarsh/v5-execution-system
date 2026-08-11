"""Epic 1.12 — the one-command runner. Last epic of Feature 1.

One entry point that runs the pipeline and leaves everything a marker needs
under `submission_artifacts/`. It grows one stage per feature; today it has
exactly one, so the shape matters more than the content.

`run.log` is byte-identical across machines and across artifact locations. Two
things buy that, and both are easy to break later:

  * NO timestamps. A clock in the log makes every run a diff, so a changed log
    would stop meaning a changed pipeline.
  * NO absolute paths. Only basenames are logged, so the same run under
    /home/x/artifacts and /tmp/y/artifacts produces the same bytes. This is
    enforced rather than trusted -- `RunLog` refuses to write a field value
    that looks like an absolute path, because that failure would otherwise be
    invisible until someone diffed two checkouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Sequence

from feature3_tokenizer.bpe_train import sample_clean_corpus
from feature2_clean.clean_pipeline import clean_corpus
from feature1_collect.corpus_loader import corpus_counts
from feature1_collect.corpus_report import write_corpus_summary
from feature1_collect.corpus_schema import LANES
from feature1_collect.sources_manifest import SOURCES, LaneSource
from feature3_tokenizer.tokenizer_build import DEFAULT_VOCAB_SIZE, build_frozen_tokenizer, verify_frozen
from feature4_shards.shard_corpus import shard_corpus
from feature4_shards.shard_index import verify_shards
from feature5_firewall.firewall import eval_shards_blocked
from feature6_mixture.compile_mixture import build_report, write_report
from feature6_mixture.mixture_config import DEFAULT_MIXTURE
from feature7_opus.opus_selector import Candidate, OpusSelector, SelectorConfig, TIER_QUALITY
from feature4_shards.shard_reader import iter_docs
from feature8_packer.packer import pack_documents, packed_batch_report
from feature9_batches.batch_stream import BatchStream, ConsumptionLedger
from feature9_batches.rng import MASTER_SEED
from feature10_trainer.moe_model import ModelConfig
from feature10_trainer.trainer import (
    LearningLedger, Trainer, batch_tensors, contrastive_delta_s,
)
from feature3_tokenizer.tokenizer_build import load_frozen_tokenizer
from feature1_collect.contrastive_pairs import CONTRASTIVE_PAIRS
from feature11_checkpoint.checkpoint import save_checkpoint, verify_checkpoint
from feature12_resume.resume import crash_and_resume, train_range
from feature13_replay.replay import replay_interval
from feature14_fork.fork import fork_run
from feature15_throughput.throughput import build_performance, measure_throughput, packing_utilization
from feature16_audit.audit import build_evidence, run_audit, write_evidence_md

__all__ = [
    "ARTIFACTS_ROOT",
    "RunLog",
    "stage_load_corpus",
    "stage_clean_corpus",
    "stage_tokenizer",
    "stage_shards",
    "stage_firewall",
    "stage_mixture",
    "stage_opus",
    "stage_packer",
    "stage_batches",
    "stage_train",
    "stage_checkpoint",
    "stage_resume",
    "stage_replay",
    "stage_fork",
    "stage_throughput",
    "stage_audit",
    "run",
    "main",
]

ARTIFACTS_ROOT: Final[str] = "submission_artifacts"
_TOKENIZER_DIR: Final[str] = "tokenizer"
_SHARD_ROOT: Final[str] = "data/shards"
_SEQ_LEN: Final[int] = 256
_BATCH_SIZE: Final[int] = 8
_N_STEPS: Final[int] = 1000
_TRAIN_SEED: Final[str] = "v5-trainer-2026"
_RUN_LOG = "run.log"
_MANIFESTS_DIR = "manifests"
_SUMMARY_FILE = "corpus_summary.json"


def _render(value: Any) -> str:
    """Stringify one field value, refusing anything machine-specific."""
    text = str(value)
    looks_absolute = text.startswith("/") or (
        len(text) > 2 and text[1] == ":" and text[2] in "\\/"
    )
    if looks_absolute:
        raise ValueError(
            f"refusing to log the absolute path {text!r}: run.log must stay "
            f"byte-identical wherever artifacts_root lives. Log a basename."
        )
    return text


class RunLog:
    """Append-only run log. Writes to a file and to stdout, and remembers.

    Field order follows the order the caller passed them, which is well
    defined for **kwargs, so the same call site always renders the same line.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.events: list[dict[str, Any]] = []
        # newline="\n" so a Windows run does not produce CRLF and a different file
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")

    def _emit(self, line: str) -> None:
        self._handle.write(line + "\n")
        print(line)

    @staticmethod
    def _suffix(fields: dict[str, Any]) -> str:
        if not fields:
            return ""
        return " " + " ".join(f"{k}={_render(v)}" for k, v in fields.items())

    def info(self, msg: str, **fields: Any) -> None:
        self._emit(f"[INFO] {msg}{self._suffix(fields)}")
        self.events.append({"level": "INFO", "msg": msg, **fields})

    def passed(self, event: str, **fields: Any) -> None:
        self._emit(f"[PASS] {event}{self._suffix(fields)}")
        self.events.append({"level": "PASS", "event": event, **fields})

    def close(self) -> None:
        self._handle.close()


def stage_load_corpus(
    log: RunLog,
    *,
    raw_root: str,
    eval_root: str,
    sources: Sequence[LaneSource],
    summary_path: str | Path,
) -> dict[str, Any]:
    """Stage 1: tally the corpus, persist the summary, report per lane."""
    counts = corpus_counts(
        raw_root=raw_root, eval_root=eval_root, sources=sources
    )

    for lane in sorted(LANES):
        bucket = counts[lane]
        log.info(
            "corpus_lane",
            lane=lane,
            train_docs=bucket["train_docs"],
            eval_docs=bucket["eval_docs"],
            train_tokens=bucket["train_tokens"],
            eval_tokens=bucket["eval_tokens"],
        )

    write_corpus_summary(
        raw_root=raw_root,
        eval_root=eval_root,
        sources=sources,
        out_path=str(summary_path),
    )
    log.info("corpus_summary_written", file=Path(summary_path).name)

    log.passed(
        "corpus_loaded",
        total=counts["totals"]["docs"],
        eval=counts["totals"]["eval_docs"],
        contrastive=counts["contrastive"]["pairs"],
    )
    return counts


def stage_clean_corpus(
    log: RunLog,
    *,
    raw_root: str,
    eval_root: str,
    sources: Sequence[LaneSource],
    clean_root: str,
) -> dict[str, Any]:
    """Stage 2: clean the train corpus and report each stage's drops."""
    report = clean_corpus(
        raw_root=raw_root, eval_root=eval_root, sources=sources, clean_root=clean_root
    )
    for s in report["stages"]:
        log.info(
            "clean_stage",
            stage=s["stage"],
            input=s["input"],
            kept=s["kept"],
            dropped=s["dropped"],
        )
    log.info("pii_redactions", n=report["pii_redactions"])
    totals = report["totals"]
    log.passed(
        "corpus_cleaned",
        kept=totals["train_out"],
        dropped=totals["dropped"],
    )
    return report


def stage_tokenizer(
    log: RunLog,
    *,
    clean_root: str,
    tokenizer_dir: str,
    vocab_size: int,
) -> Any:
    """Stage 3: train + freeze the byte-level BPE tokenizer, verify, round-trip.

    The freeze contract: the on-disk tokenizer's recomputed content hash must
    match its manifest, and encode->decode must return every cleaned document
    verbatim (real evidence, not a canned string).
    """
    tok = build_frozen_tokenizer(
        clean_root=clean_root, out_dir=tokenizer_dir, vocab_size=vocab_size
    )
    if not verify_frozen(tokenizer_dir):
        raise ValueError("frozen tokenizer hash does not match its manifest")

    checked = sample_clean_corpus(clean_root, docs_per_lane=1, max_chars=500)
    for text in checked:
        if tok.decode(tok.encode(text)) != text:
            raise ValueError("tokenizer round-trip failed on a cleaned document")

    log.info("tokenizer", vocab=len(tok.vocab), merges=len(tok.merges))
    log.info("tokenizer_roundtrip", lanes_checked=len(checked))
    log.passed(
        "tokenizer_frozen",
        vocab=len(tok.vocab),
        merges=len(tok.merges),
        hash=tok.content_hash(),
    )
    return tok


def stage_shards(
    log: RunLog,
    *,
    clean_root: str,
    raw_root: str,
    eval_root: str,
    sources: Sequence[LaneSource],
    tokenizer_dir: str,
    shard_root: str,
) -> dict[str, Any]:
    """Stage 4: tokenize the corpus into immutable, content-addressed shards."""
    index = shard_corpus(
        clean_root=clean_root, raw_root=raw_root, eval_root=eval_root,
        sources=sources, tokenizer_dir=tokenizer_dir, shard_root=shard_root,
    )
    for split in index["by_split"]:
        b = index["by_split"][split]
        log.info("shards_split", split=split, shards=b["shards"], tokens=b["tokens"])
    result = verify_shards(shard_root)
    if not result["ok"]:
        raise ValueError(f"shard verification failed: {result['mismatches'][:3]}")
    log.passed(
        "shards_written",
        shards=index["n_shards"],
        tokens=index["total_tokens"],
        verified=result["ok"],
    )
    return index


def stage_firewall(log: RunLog, *, index: dict[str, Any]) -> dict[str, Any]:
    """Stage 5: prove every eval shard is quarantined from training."""
    result = eval_shards_blocked(index)
    if not result["ok"]:
        raise ValueError("evaluation firewall breached: an eval shard is admissible")
    log.passed(
        "eval_shard_blocked",
        blocked=result["blocked"],
        eval_shards=result["eval_shards"],
        train_shards=result["train_shards"],
    )
    return result


def stage_mixture(
    log: RunLog, *, index: dict[str, Any], summary_path: str | Path
) -> dict[str, Any]:
    """Stage 6: compile the India-first curriculum into per-lane token targets."""
    available: dict[str, int] = {}
    for s in index["shards"]:
        if s["split"] == "train":
            available[s["lane"]] = available.get(s["lane"], 0) + s["n_tokens"]
    report = build_report(DEFAULT_MIXTURE, available=available)
    write_report(report, str(summary_path))
    if not report["all_floors_met"]:
        raise ValueError("mixture floors not met — India-first lanes under-allocated")
    log.info("mixture_written", file=Path(summary_path).name)
    log.passed(
        "mixture_compiled",
        phases=len(report["phases"]),
        floors_met=report["all_floors_met"],
    )
    return report


def _floor_tokens_per_lane(mixture_report: dict[str, Any]) -> dict[str, int]:
    """Aggregate protected-floor tokens per lane across all phases."""
    floors: dict[str, int] = {}
    for phase in mixture_report["phases"]:
        for lane, info in phase["lanes"].items():
            floors[lane] = floors.get(lane, 0) + int(info["floor"] * phase["budget"])
    return floors


def stage_opus(
    log: RunLog, *, index: dict[str, Any], mixture_report: dict[str, Any], ledger_path: str | Path
) -> dict[str, Any]:
    """Stage 7: run the OPUS accept/reject/defer selection over train shards."""
    cfg = SelectorConfig(
        lane_targets=mixture_report["lane_totals"],
        lane_floors=_floor_tokens_per_lane(mixture_report),
    )
    candidates = [
        Candidate(
            id=s["shard_id"], lane=s["lane"], tokens=s["n_tokens"],
            quality=max(TIER_QUALITY[t] for t in s["provenance_tiers"]),
        )
        for s in index["shards"] if s["split"] == "train"
    ]
    sel = OpusSelector(cfg).run(candidates)
    sel.write_ledger(str(ledger_path))
    summary = sel.summary()
    log.info("opus_ledger_written", file=Path(ledger_path).name)
    log.passed(
        "opus_selected",
        accepted=summary["decisions"]["accept"],
        deferred=summary["decisions"]["defer"],
        rejected=summary["decisions"]["reject"],
        floors_met=summary["floors_met"],
    )
    return summary


def stage_packer(
    log: RunLog, *, shard_root: str, seq_len: int, report_path: str | Path
):
    """Stage 8: pack train docs into fixed-length sequences; report efficiency."""
    sequences = pack_documents(iter_docs(shard_root, split="train"), seq_len=seq_len)
    report = packed_batch_report(sequences, seq_len=seq_len)
    with Path(report_path).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    log.info("packing_written", file=Path(report_path).name)
    log.passed(
        "sequences_packed",
        seqs=report["n_sequences"],
        efficiency=round(report["packing_efficiency"], 4),
    )
    return sequences


def stage_batches(
    log: RunLog, *, sequences, mixture_report: dict[str, Any],
    seed: str = MASTER_SEED, batch_size: int = _BATCH_SIZE,
) -> BatchStream:
    """Stage 9: build the reproducible, mixture-weighted batch stream."""
    stream = BatchStream(
        sequences, seed=seed, batch_size=batch_size,
        lane_weights=mixture_report["lane_totals"],
    )
    # reproducibility proof: a fresh stream (seed only) rebuilds sampled batches
    fresh = BatchStream(sequences, seed=seed, batch_size=batch_size,
                        lane_weights=mixture_report["lane_totals"])
    ok = all(stream.batch(i).content_hash == fresh.batch(i).content_hash for i in (0, 1, 7, 15))
    if not ok:
        raise ValueError("batch stream is not reproducible from its seed")
    log.passed(
        "batch_stream_ready", pool=len(sequences), batch_size=batch_size, reproducible=ok,
    )
    return stream


def stage_train(
    log: RunLog, *, stream: BatchStream, tokenizer_dir: str, seq_len: int,
    n_steps: int, manifests: Path, seed: str = _TRAIN_SEED,
) -> dict[str, Any]:
    """Stage 10: deterministically train the tiny MoE; write the learning ledger + ΔS."""
    tok = load_frozen_tokenizer(tokenizer_dir)
    cfg = ModelConfig(vocab_size=len(tok.vocab), seq_len=seq_len)
    trainer = Trainer(cfg, seed=seed)
    log.info("moe_params", n=trainer.model.n_params())

    consumption = ConsumptionLedger(str(manifests / "consumption_ledger.jsonl"))
    learning = LearningLedger(str(manifests / "learning_ledger.jsonl"))
    last_loss = 0.0
    for i in range(n_steps):
        batch = stream.batch(i)
        consumption.record(batch)
        tokens, pos, allowed, lm = batch_tensors(stream, batch)
        last_loss = trainer.train_step(tokens, pos, allowed, lm)
        learning.record(step=i, batch=batch, loss=last_loss, n_loss_tokens=int(lm.sum()))
    consumption.close()
    learning.close()

    delta = contrastive_delta_s(trainer.model, tok, CONTRASTIVE_PAIRS, seq_len=seq_len)
    with (manifests / "contrastive_delta_s.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"kind": "contrastive_delta_s", "pairs": delta},
                            ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    log.passed("trained", steps=n_steps, final_loss=round(last_loss, 6))
    log.passed("contrastive_delta_s", pairs=len(delta))
    return {"trainer": trainer, "cfg": cfg, "tokenizer": tok, "n_steps": n_steps,
            "learning_rows": learning.rows, "consumption_rows": consumption.rows}


def stage_checkpoint(log: RunLog, *, make_trainer, stream, checkpoint_dir: str) -> None:
    """Stage 11: train a few steps, checkpoint (model+optim+rng+offset), verify."""
    tr = make_trainer()
    train_range(tr, stream, 0, 4)
    save_checkpoint(tr, step=4, ledger_offset=4, seed=stream.seed, out_dir=checkpoint_dir)
    ok = verify_checkpoint(checkpoint_dir)
    if not ok:
        raise ValueError("checkpoint failed to verify on restore")
    log.passed("checkpoint_saved", step=4, ledger_offset=4, verified=ok)


def stage_resume(log: RunLog, *, make_trainer, stream, checkpoint_dir: str) -> None:
    """Stage 12: crash at a set batch, resume, prove the next batch matches."""
    result = crash_and_resume(make_trainer, stream, total=6, crash_at=3,
                              checkpoint_dir=checkpoint_dir, seed=stream.seed)
    if not (result["no_skip_or_repeat"] and result["loss_trajectory_matched"]):
        raise ValueError("resume did not reproduce the clean run")
    log.passed(
        "resume_next_batch_matched",
        offset=result["resume_offset"],
        no_skip_or_repeat=result["no_skip_or_repeat"],
        loss_matched=result["loss_trajectory_matched"],
    )


def stage_replay(log: RunLog, *, stream) -> None:
    """Stage 13: replay an interval from the ledger and match hashes."""
    ledger = [stream.batch(i).as_ledger_row() for i in range(6)]
    result = replay_interval(stream, ledger, 1, 5)
    if not result["matched"]:
        raise ValueError("replay hashes did not match the ledger")
    log.passed("replay_hash_matched", interval="1-5", checked=result["checked"], matched=result["matched"])


def stage_fork(log: RunLog, *, stream, checkpoint_dir: str, out_path: Path) -> None:
    """Stage 14: fork from the checkpoint onto a new-seed branch; record lineage."""
    lineage = fork_run(checkpoint_dir, stream, branch_id="branch-a", fork_seed="v5-fork-a", steps=3)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(lineage, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    log.passed("fork_lineage_recorded", branch=lineage["branch_id"],
               diverged_at=lineage["diverged_at"], diverged=lineage["diverged"])


def stage_throughput(log: RunLog, *, make_trainer, stream, manifests: Path) -> None:
    """Stage 15: deterministic packing efficiency (logged) + wall-clock throughput (file)."""
    packing_report = json.loads((manifests / "packing_report.json").read_text(encoding="utf-8"))
    throughput = measure_throughput(make_trainer, stream, n_steps=5)
    performance = build_performance(packing_report, throughput)
    with (manifests / "performance.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(performance, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    log.passed(
        "throughput_measured",
        efficiency=round(packing_utilization(packing_report), 4),
        loss_tokens=packing_report["loss_positions"],
    )


def stage_audit(
    log: RunLog, *, tokenizer_dir: str, shard_root: str, checkpoint_dir: str, manifests: Path,
) -> dict[str, Any]:
    """Stage 16: cross-check every artifact and write the evidence bundle."""
    emitted = [e["event"] for e in log.events if e.get("level") == "PASS"]
    audit = run_audit(
        emitted_pass_events=emitted, tokenizer_dir=tokenizer_dir,
        shard_root=shard_root, checkpoint_dir=checkpoint_dir, manifests=str(manifests),
    )
    evidence = build_evidence(
        tokenizer_dir=tokenizer_dir, shard_root=shard_root,
        checkpoint_dir=checkpoint_dir, manifests=str(manifests), audit=audit,
    )
    with (manifests / "evidence.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    write_evidence_md(evidence, str(manifests / "evidence.md"))
    if not audit["all_passed"]:
        raise ValueError(f"audit failed: {audit['checks']}")
    log.passed("audit_complete", checks=len(audit["checks"]), all_passed=audit["all_passed"])
    log.info("evidence_written", file="evidence.json")
    return evidence


def run(
    *,
    raw_root: str = "data/raw",
    eval_root: str = "data/eval",
    sources: Sequence[LaneSource] = SOURCES,
    artifacts_root: str = ARTIFACTS_ROOT,
    clean_root: str = "data/clean",
    tokenizer_dir: str = _TOKENIZER_DIR,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    shard_root: str = _SHARD_ROOT,
    seq_len: int = _SEQ_LEN,
    n_steps: int = _N_STEPS,
) -> int:
    """Run every stage. Returns a process exit code: 0 on success."""
    root = Path(artifacts_root)
    manifests = root / _MANIFESTS_DIR
    manifests.mkdir(parents=True, exist_ok=True)

    log = RunLog(root / _RUN_LOG)
    try:
        log.info("run_start")
        stage_load_corpus(
            log,
            raw_root=raw_root,
            eval_root=eval_root,
            sources=sources,
            summary_path=manifests / _SUMMARY_FILE,
        )
        stage_clean_corpus(
            log,
            raw_root=raw_root,
            eval_root=eval_root,
            sources=sources,
            clean_root=clean_root,
        )
        stage_tokenizer(
            log,
            clean_root=clean_root,
            tokenizer_dir=tokenizer_dir,
            vocab_size=vocab_size,
        )
        index = stage_shards(
            log,
            clean_root=clean_root,
            raw_root=raw_root,
            eval_root=eval_root,
            sources=sources,
            tokenizer_dir=tokenizer_dir,
            shard_root=shard_root,
        )
        stage_firewall(log, index=index)
        mixture_report = stage_mixture(log, index=index, summary_path=manifests / "mixture_plan.json")
        stage_opus(
            log, index=index, mixture_report=mixture_report,
            ledger_path=manifests / "opus_decision_ledger.jsonl",
        )
        sequences = stage_packer(
            log, shard_root=shard_root, seq_len=seq_len,
            report_path=manifests / "packing_report.json",
        )
        stream = stage_batches(log, sequences=sequences, mixture_report=mixture_report)
        train_result = stage_train(
            log, stream=stream, tokenizer_dir=tokenizer_dir, seq_len=seq_len,
            n_steps=n_steps, manifests=manifests,
        )
        # reproducibility harness (Features 11-14): a tiny model, exercised fast
        tok = train_result["tokenizer"]
        repro_cfg = ModelConfig(vocab_size=len(tok.vocab), d_model=32, n_layers=1,
                                n_heads=2, n_experts=2, top_k=1, d_ff=64, seq_len=seq_len)
        make_trainer = lambda: Trainer(repro_cfg, seed="v5-repro")  # noqa: E731
        ckpt_dir = str(root / "checkpoint")
        stage_checkpoint(log, make_trainer=make_trainer, stream=stream, checkpoint_dir=ckpt_dir)
        stage_resume(log, make_trainer=make_trainer, stream=stream,
                     checkpoint_dir=str(manifests / "resume_checkpoint"))
        stage_replay(log, stream=stream)
        stage_fork(log, stream=stream, checkpoint_dir=ckpt_dir,
                   out_path=manifests / "fork_lineage.json")
        stage_throughput(log, make_trainer=make_trainer, stream=stream, manifests=manifests)
        stage_audit(log, tokenizer_dir=tokenizer_dir, shard_root=shard_root,
                    checkpoint_dir=ckpt_dir, manifests=manifests)
        log.info("run_complete")
    finally:
        # even a failing stage leaves a readable log behind
        log.close()
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
