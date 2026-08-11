# Training Data Execution System

A reproducible, auditable pipeline that turns documents into training batches.
Given a seed and a ledger offset, any batch can be reconstructed exactly. That is
what makes crash-resume, replay, and fork correct by construction rather than by
bookkeeping.

Scope is deliberately small: ~10M tokens, a byte-level BPE tokenizer, a tiny
mixture-of-experts model. The goal is correctness and reproducibility, not scale.
The corpus leans toward India-first lanes (Hindi, Bengali, Tamil alongside web,
code, and math), so a few design choices are aimed at not starving them.

## The reproducibility invariant

The system is organized around one property:

> A seed plus a ledger offset reconstructs any batch, byte for byte.

Batch `i` is a pure function of `(seed, i)`. Each batch draws from its own
generator, seeded from a hash of the master seed and the batch index, so batch
900 does not depend on having produced batches 0 through 899. It can be rebuilt
directly.

This is what keeps the recovery features small:

- Resume continues from the checkpoint's ledger offset. "The next batch" is
  recomputed, not remembered, so no batch is skipped or repeated.
- Replay recomputes any interval and checks the batch ids, sequence indices, and
  content hashes against the recorded ledger.
- Fork restores an earlier checkpoint and continues under a different seed, with
  the lineage recorded back to the parent.

The data plane (shards, sampling, ledgers, hashes) is byte-identical on any
machine, because it is integers and hashes. The model's loss values are
deterministic only on the same machine: floating-point addition is not
associative, and different CPUs sum reductions in a different order. Those two
claims are kept separate throughout.

## Pipeline

`python run_demo.py` runs the full path:

```
documents
  -> clean & filter          normalize, dedup, quality, PII, decontaminate
  -> frozen BPE tokenizer     byte-level, content-hashed
  -> immutable shards         content-addressed, with manifests
  -> evaluation firewall      eval shards quarantined from loss
  -> mixture / curriculum     lane weights, protected floors
  -> OPUS selector            accept / reject / defer / floor override
  -> packer                   loss masks, attention masks, position ids
  -> batch stream + ledger    deterministic; one record per batch
  -> tiny MoE trainer         learning ledger: per-token surprisal, delta-S
  -> checkpoint               model + optimizer + RNG + ledger offset
  -> crash -> resume -> replay -> fork
  -> audit + evidence bundle
```

Each stage emits a `[PASS]` line to `run.log` and writes its artifacts. The final
audit cross-checks the run against those artifacts and produces the evidence
bundle.

## Design decisions

**Byte-level tokenizer.** The base alphabet is the 256 byte values, so every
script in the corpus is representable with no unknown token. This matters for
Indian scripts specifically: a word-level tokenizer that splits on Unicode word
characters drops combining marks (Bengali vowel signs, the virama), which
fragments a syllable into single consonants. Byte-level makes that failure
impossible. The tokenizer is validated by a round-trip check on every lane:
decode(encode(text)) must equal the original text exactly.

**Protected floors in the mixture.** Lane proportions are the model's capability
budget, and the India-first lanes are small and easily starved. The curriculum
compiler reserves each lane's floor first and distributes the remainder by
weight, so a lane can never fall below its floor regardless of the weights.
Meeting the floors is arithmetic, verified in the mixture plan.

**OPUS with a floor override.** The selector accepts, rejects, or defers each
candidate by score. A score trained on English and coding benchmarks will
under-value Indian-language data, so OPUS accepts unconditionally while a lane is
below its floor. It also exposes a delta-S surprisal hook for a perspective
signal. Every decision is written to a ledger.

**Immutable, content-addressed artifacts.** Shards are written once and named by
the sha256 of their bytes. The tokenizer is frozen and hashed, and that hash is
referenced by every downstream stage. The audit re-hashes every shard, reloads
the tokenizer, restores the checkpoint, and confirms every batch id in the
learning ledger appears in the consumption ledger. A changed byte breaks a hash
and fails the audit.

**Determinism discipline.** No timestamps or absolute paths reach `run.log`; the
logger rejects any value that looks like an absolute path. Sampling is seeded,
the tokenizer trainer breaks ties deterministically, and JSON is written with
stable key ordering. Two runs on one machine produce a byte-identical log.

## Running it

The corpus is committed, so there is no fetch step and no network access:

```bash
python run_demo.py
```

`n_steps` is a cumulative target. A later `run(n_steps=N)` resumes from the saved
training checkpoint and trains only the remaining steps rather than restarting;
that an incremental run matches a single long run is covered by a test.

To re-fetch the corpus from source instead of using the committed copy:

```bash
pip install -e ".[dev]"            # installs datasets + huggingface_hub
python -m feature1_collect.fetch   # writes data/raw/ (cached after the first run)
```

## Tests

```bash
pip install pytest
pytest        # 295 tests, fully offline; datasets / huggingface_hub not required
```

The tests are the specification of what "correct" means here: round-trip
tokenization, shard immutability, mask and position-id correctness, floor
enforcement, batch reconstruction, resume equivalence, replay hash matching.

## Output

One command regenerates the full evidence tree:

```
submission_artifacts/
  run.log             complete [PASS]/[INFO] event sequence
  evidence.json       per-requirement PASS/FAIL, each with a pointer to its proof
  evidence.md         Requirement / Result / Evidence table
  performance.json    packing efficiency and loss-bearing tokens per second
  manifests/          tokenizer, shard index, mixture plan, packing report,
                      contrastive delta-S, fork lineage, corpus summary
  ledgers/            consumption, learning, and OPUS decision ledgers
  checkpoints/        main / train / resume (offset + model hash committed;
                      the weight binaries are regenerated by the run)
```

Every value in the bundle is read back from a file an earlier stage wrote;
nothing is hardcoded.

## Code layout

Source is grouped by feature, one package per stage; tests are under `tests/`.
Imports are package-qualified.

```
feature1_collect/     schema, sources, fetch, contrastive pairs, eval split, loader, report
feature2_clean/       normalize, content hash + dedup, quality, near-dup, PII, decontaminate
feature3_tokenizer/   byte level, BPE trainer, freeze + manifest, encode, decode, integrity
feature4_shards/      content-addressed uint16 shards, manifests, index, reader
feature5_firewall/    eval shards can never enter a train batch
feature6_mixture/     phases, lane weights, protected floors -> per-lane targets
feature7_opus/        accept / reject / defer + floor override + delta-S hook + ledger
feature8_packer/      position ids, segment + attention masks, loss masks
feature9_batches/     deterministic batch stream (seed + offset) + consumption ledger
feature10_trainer/    tiny PyTorch MoE + learning ledger (F1 surprisal) + delta-S (F2)
feature11_checkpoint/ model + optimizer + RNG + offset checkpoint, with a model hash
feature12_resume/     deliberate crash + resume (no skip / repeat; loss matches)
feature13_replay/     replay an interval from seed + ledger; ids and hashes match
feature14_fork/       fork onto a new-seed branch; lineage recorded
feature15_throughput/ packing efficiency (logged) + wall-clock throughput (file only)
feature16_audit/      cross-check artifacts + build the evidence bundle
run_demo.py           one command; runs all sixteen stages
tokenizer/            the frozen artifact: vocab.json, merges.txt, manifest
```

## Notes and limitations

The crash, resume, replay, and fork demonstrations use a small model against the
real batch stream, not the full training model. The property being checked (the
reconstructed batches and the resumed loss trajectory match exactly) is a
property of the stream and the checkpoint, both of which are the real ones;
retraining the full model twice would prove nothing further.

The committed demonstration trains for a few hundred steps. That is enough to
show the loss falling below the uniform baseline and to link every loss value to
the batch, and through the batch to the source documents, that produced it. The
same curve keeps descending with more steps; the committed run is sized to be
fast to reproduce and easy to check.
