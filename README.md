# The data plane that can prove what it fed the model

Everyone who trains a model obsesses over the model. Layers, attention variants,
the newest optimizer. I've argued elsewhere that this is the wrong obsession: the
thing that decides whether your model is brilliant or useless is the data, and
specifically what data, in what proportions, in what order.

But there's a step before any of that, and almost nobody builds it properly. If
the data decides everything, then you had better be able to say, with certainty,
exactly what the model consumed, why it consumed it, and how to rebuild any
moment of the run. Not roughly. Exactly. Byte for byte.

That is what this repository is. It is a small but complete training-data
execution system: the machine that turns a pile of documents into training
batches and keeps a receipt for every decision along the way. It is built for a
V5 India-first model, so the design leans toward the lanes that usually get
starved, but the machinery is general.

The goal here was never scale. The corpus is about ten million tokens, the
tokenizer is a toy, the model is a tiny mixture-of-experts. The goal was to prove
that the data system is correct, reproducible and auditable, and to make those
claims checkable rather than asserted.

## The one invariant everything hangs on

Here is the promise the whole system is organized around:

> A seed plus a ledger offset reconstructs any batch, byte for byte.

Read that again, because it is the load-bearing wall. If it holds, everything
else follows almost for free. You can crash the run and resume it without
skipping or repeating a single batch, because "the next batch" is a fact you can
recompute, not a position you have to remember. You can replay any slice of
history and get identical batch ids and hashes. You can fork an experiment from
an old checkpoint and know precisely where the two timelines split.

The trick that buys all of this is simple: batch number `i` is a pure function
of `(seed, i)`. Each batch gets its own generator, seeded from a hash of the
master seed and the batch index. So batch 900 does not depend on having played
batches 0 through 899. You can rebuild it directly. That one decision is why
resume, replay and fork are three lines each instead of three subsystems each.

## The path, end to end

Run one command and the system walks the full path:

```
documents
  -> clean & filter          (normalize, dedup, quality, PII, decontaminate)
  -> frozen BPE tokenizer     (byte-level, content-hashed)
  -> immutable shards         (content-addressed, with manifests)
  -> evaluation firewall      (eval shards quarantined from loss)
  -> mixture / curriculum      (lane weights, protected floors)
  -> OPUS selector            (accept / reject / defer / floor override)
  -> packer                   (loss masks, attention masks, position ids)
  -> batch stream + ledger    (deterministic; a receipt per batch)
  -> tiny MoE trainer         (learning ledger: per-token surprisal, ΔS)
  -> checkpoint               (model + optimizer + RNG + ledger offset)
  -> crash -> resume -> replay -> fork
  -> audit + evidence bundle
```

Every stage writes down what it did. By the end there is a `run.log` with a
`[PASS]` line for each stage and an evidence bundle that a grader, or a future
version of me, can check against the artifacts without trusting a word I say.

## The design decisions I would defend

A few choices here are opinions, not defaults, and they are the interesting part.

**Byte-level tokenizer, because Indian scripts break the usual one.** The base
alphabet is the 256 byte values, so every script the corpus contains, Devanagari,
Bengali, Tamil, and anything else, is representable with no unknown token and
nothing to fragment. This is not cosmetic. A word-level tokenizer that splits on
Unicode word characters quietly shreds Bengali, because the vowel signs and the
virama are combining marks that the usual `\w` throws away. I found this the hard
way when a Bengali word came back as a string of single-consonant fragments. The
byte-level choice makes that failure impossible, and it is the difference between
a tokenizer that respects the India-first lanes and one that silently corrupts
them. The whole tokenizer turns on one property, checked on every lane: encode
then decode returns the original text, exactly.

**Protected floors, because the mixture is the model's personality.** The
mixture is a diet. Every slice of the token budget you hand to one skill is a
slice you take from another, and that single set of proportions is who the model
becomes. The India-first lanes are small and easy to starve, so the curriculum
reserves a floor for them first and only then distributes what's left by weight.
A lane can never fall below its floor no matter what the weights say. The
compiler proves this: the floors are allocated before anything else, so meeting
them is arithmetic, not hope.

**OPUS, because your own selector will betray you.** During selection you decide
what data to accept, reject or defer, usually by some score. The failure mode is
famous and quiet: a myopic selector, scoring against mostly-English,
mostly-coding benchmarks, glances at the first few hundred tokens of your
precious Indian-language data, sees nothing that moves its score, and throws it
away. So OPUS carries a protected-floor override. While a lane sits below its
floor, the selector accepts regardless of score. It also carries a ΔS surprisal
hook, the honest place to plug in a perspective signal later, so the selector can
be taught to care about the right things rather than bribed into it. Every
decision, accept, reject, defer or override, lands in a ledger, so the selection
is a record you can audit, not a black box.

**Immutability and hashes everywhere, because "trust me" is not evidence.**
Shards are written once and content-addressed by the sha256 of their bytes. The
tokenizer is frozen and hashed, and that hash is the identity every downstream
stage references. Re-running re-verifies: the audit re-hashes every shard,
reloads the tokenizer, restores the checkpoint, and checks that every batch id in
the learning ledger appears in the consumption ledger. If a single byte changed,
a hash would not match, and the audit would say so.

**Determinism is a discipline, not an accident.** No timestamps and no absolute
paths reach `run.log`, which is enforced rather than hoped for: the logger
refuses to write a value that looks like an absolute path. Two runs on the same
machine produce a byte-identical log. Sampling is seeded, the tokenizer trainer
breaks ties deterministically, and the JSON is written with stable ordering. The
one honest exception is the model's loss values, which are deterministic on a
given machine but not portable across hardware, because floating-point addition
is not associative and different CPUs sum in different orders. So the data plane,
the part that is all integers and hashes, is byte-identical anywhere. The
model's floats are byte-identical only on the same machine. I keep those two
claims separate on purpose.

## Run it

The corpus is committed, so there is no fetch step and no network. Clone and run:

```bash
python run_demo.py
```

It executes all sixteen stages on the real corpus and writes the full evidence
tree with no manual intervention. The learning trace descends from the start;
`n_steps` is a cumulative target, so a later `run(n_steps=N)` resumes from the
saved training checkpoint and trains only the remaining steps rather than
starting over. That the incremental run matches a single long run is itself a
test.

Run the tests, which are the actual specification of what "correct" means here:

```bash
pip install pytest
pytest        # 295 tests, fully offline; datasets / huggingface_hub not required
```

## What it produces

One command regenerates the whole bundle:

```
submission_artifacts/
  run.log             complete [PASS]/[INFO] event sequence for the run
  evidence.json       per-requirement PASS/FAIL, each with a pointer to its proof
  evidence.md         the Requirement / Result / Evidence summary table
  performance.json    packing efficiency and loss-bearing tokens per second
  manifests/          tokenizer, shard index, mixture plan, packing report,
                      contrastive ΔS, fork lineage, corpus summary
  ledgers/            consumption, learning, and OPUS decision ledgers
  checkpoints/        main / train / resume (offset + model hash committed;
                      the weight binaries are regenerated by the run)
```

Nothing in that bundle is hardcoded. Every number is read back from a file an
earlier stage wrote. If a claim in `evidence.md` cannot be reconstructed from the
artifacts, it is not a claim I want credit for.

## How the code is laid out

Source is grouped by feature, one package per pipeline stage, and tests live
under `tests/`. Imports are package-qualified.

```
feature1_collect/     schema, sources, fetch, contrastive pairs, eval split, loader, report
feature2_clean/       normalize, content hash + dedup, quality, near-dup, PII, decontaminate
feature3_tokenizer/   byte level, BPE trainer, freeze + manifest, encode, decode, integrity
feature4_shards/      content-addressed uint16 shards, manifests, index, reader
feature5_firewall/    eval firewall: eval shards can never enter a train batch
feature6_mixture/     curriculum: phases, lane weights, protected floors -> targets
feature7_opus/        accept / reject / defer + floor override + ΔS hook + ledger
feature8_packer/      position ids, segment + attention masks, loss masks, contrastive policy
feature9_batches/     deterministic batch stream (seed + offset) + consumption ledger
feature10_trainer/    tiny PyTorch MoE + learning ledger (F1 surprisal) + ΔS (F2)
feature11_checkpoint/ model + optimizer + RNG + offset checkpoint, with a model hash
feature12_resume/     deliberate crash + resume (no skip / repeat; loss matches a clean run)
feature13_replay/     replay an interval from seed + ledger; ids and hashes match
feature14_fork/       fork onto a new-seed branch; lineage recorded back to the parent
feature15_throughput/ packing efficiency (logged) + wall-clock throughput (file only)
feature16_audit/      cross-check every artifact + build the evidence bundle
run_demo.py           the one command; runs all sixteen stages
tokenizer/            the frozen artifact: vocab.json, merges.txt, manifest
```

## What I would tell a reviewer up front

Two things are worth saying plainly rather than hiding.

The crash, resume, replay and fork demonstrations run on a small model exercised
against the real batch stream, not on the full training model, because retraining
the full model twice to prove resume would be slow and would prove nothing extra.
The property being proved, that the reconstructed batches and the resumed loss
trajectory match exactly, is about the stream and the checkpoint, and both are
the real ones.

The committed demonstration trains for a few hundred steps, which is enough to
show the loss falling well below the uniform baseline and to link every loss
value to the batch, and through the batch to the source documents, that produced
it. Trained longer, the same curve keeps descending; the point of the committed
run is that it is fast to reproduce and easy to check, not that it produces a
finished model.

## The bottom line

The architecture gets the headlines. The data decides the outcome. And a data
decision you cannot reconstruct is not a decision, it is a guess you got away
with once. This system exists so that every batch the model ever sees can be
named, explained, and rebuilt from a seed and an offset. Get that right, and the
rest of training is the part you can actually reason about.
