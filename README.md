# v5-execution-system

A small but complete, reproducible and auditable **Training Data Execution System**
for LLM pretraining (Session 6 assignment). Built epic by epic; the delivery plan lives
in the companion tracker (`session6_plan.md` / the Assignment page).

> Complete: all 16 features. `python run_demo.py` runs the whole path
> (documents → shards → manifests → mixture → packing → batches → training →
> ledgers → checkpoint → crash → resume → replay → fork → audit) on the
> committed corpus, **offline, no network, no manual steps**, and writes the
> full `submission_artifacts/` evidence bundle. The governing invariant: a seed
> plus a ledger offset reconstructs any batch byte-for-byte.

## Status

| Feature | Epic | State |
|---|---|---|
| 1 · Collecting data | 1.1 corpus data model | ✅ done |
| 1 · Collecting data | 1.2 sources manifest | ✅ done |
| 1 · Collecting data | 1.3 fetch one lane (web/FineWeb) | ✅ done |
| 1 · Collecting data | 1.4–1.7 fetch code / math / indic / multilingual | ✅ done |
| 1 · Collecting data | 1.8 author contrastive pairs (36) | ✅ done |
| 1 · Collecting data | 1.9 carve eval split | ✅ done |
| 1 · Collecting data | 1.10 corpus loader | ✅ done |
| 1 · Collecting data | 1.11 corpus summary report | ✅ done |
| 1 · Collecting data | 1.12 wire load_corpus into run_demo | ✅ done |
| **1 · Collecting data — COMPLETE** | | ✅ |
| 2 · Clean & filter | 2.1–2.7 normalize · hash+dedup · quality · near-dup · PII · decontaminate · pipeline | ✅ done |
| **2 · Clean & filter — COMPLETE** | | ✅ |
| 3 · Frozen BPE tokenizer | 3.1–3.8 byte level · BPE trainer · freeze · encode · decode · specials · manifest+hash · run_demo | ✅ done |
| **3 · Frozen BPE tokenizer — COMPLETE** | | ✅ |
| 4 · Immutable shards + manifests | 4.1–4.4 | ✅ |
| 5 · Evaluation firewall | 5.1–5.3 | ✅ |
| 6 · Mixture / curriculum | 6.1–6.4 | ✅ |
| 7 · OPUS selector | 7.1–7.5 | ✅ |
| 8 · Packer (masks, position ids) | 8.1–8.6 | ✅ |
| 9 · Batch stream + consumption ledger | 9.1–9.5 | ✅ |
| 10 · Trainer (MoE) + learning ledger | 10.1–10.6 | ✅ |
| 11 · Checkpoints | 11.1–11.4 | ✅ |
| 12 · Crash + resume | 12.1–12.4 | ✅ |
| 13 · Replay | 13.1–13.3 | ✅ |
| 14 · Fork | 14.1–14.3 | ✅ |
| 15 · Throughput | 15.1–15.3 | ✅ |
| 16 · Audit + evidence | 16.1–16.6 | ✅ |
| **ALL 16 FEATURES — COMPLETE** | | ✅ |

The full ~10M-token pool is fetched and hash-verified across all five lanes
(web 4.0M, code 2.0M, math 1.2M, indic 2.2M, multilingual 0.6M; 13,087 docs).
The code lane uses `Nan-Do/code-search-net-python` — chosen at fetch because the
originally-declared script-based/gated code sources would not load ungated.

## Layout (so far)

Source is grouped by feature; tests live under `tests/`. Imports are
package-qualified (e.g. `from feature2_clean.pii_scrub import scrub_pii`).

```
feature1_collect/          # Feature 1 — collecting data
  corpus_schema.py         # 1.1 — Document / ContrastivePair records + validators
  sources_manifest.py      # 1.2 — LaneSource, SOURCES (10M pool), validate_sources
  fetch.py                 # 1.3–1.7 — fetch_source / fetch_all: stream -> documents.jsonl + fetch_log + sha256
  contrastive_pairs.py     # 1.8 — 36 hand-authored Indian-vantage vs Western-default pairs
  eval_split.py            # 1.9 — select_eval / carve_eval: deterministic held-out eval (T1 only)
  corpus_loader.py         # 1.10 — iter_documents / corpus_counts / load_corpus
  corpus_report.py         # 1.11 — build/write corpus_summary.json (deterministic)
feature2_clean/            # Feature 2 — clean & filter
  text_tokens.py           # shared — script-aware word tokenizer (keeps Indic combining marks)
  text_normalize.py        # 2.1 — normalize_text/normalize_document (NFC, idempotent)
  content_hash.py          # 2.2 — content_hash + dedup_exact
  quality_filter.py        # 2.3 — quality_ok / filter_quality (length, symbol, repetition)
  near_dedup.py            # 2.4 — MinHash + LSH near-duplicate removal (pure Python)
  pii_scrub.py             # 2.5 — scrub_pii (email/phone -> placeholders, idempotent)
  decontaminate.py         # 2.6 — n-gram decontamination of train vs eval + contrastive
  clean_pipeline.py        # 2.7 — clean_corpus: compose 2.1-2.6 -> data/clean + report
feature3_tokenizer/        # Feature 3 — frozen byte-level BPE tokenizer
  byte_level.py            # 3.1 — byte<->symbol + pre-tokenize (256-byte base; lossless every lane)
  bpe_train.py             # 3.2 — deterministic byte-level BPE trainer (lazy heap) + corpus sampler
  bpe_tokenizer.py         # 3.3-3.6 — Tokenizer: save/load, encode, decode, specials, integrity
  tokenizer_build.py       # 3.7 — build/freeze + manifest + content hash + verify
feature4_shards/           # 4 — content-addressed uint16 shards + manifests + index + reader
feature5_firewall/         # 5 — eval firewall (eval shards never enter a train batch)
feature6_mixture/          # 6 — curriculum: phases, lane weights, protected floors -> targets
feature7_opus/             # 7 — OPUS accept/reject/defer + floor override + ΔS hook + ledger
feature8_packer/           # 8 — pack to seq_len: position ids, segment/attention masks, loss mask
feature9_batches/          # 9 — deterministic batch stream (seed+offset) + consumption ledger
feature10_trainer/         # 10 — tiny PyTorch MoE + learning ledger (F1 surprisal) + ΔS (F2)
feature11_checkpoint/      # 11 — model+optim+rng+offset checkpoint (+ hash) & restore
feature12_resume/          # 12 — deliberate crash + resume (no skip/repeat; loss matches)
feature13_replay/          # 13 — replay [a,b) from seed+ledger; hashes match
feature14_fork/            # 14 — fork onto a new-seed branch; lineage recorded
feature15_throughput/      # 15 — packing efficiency (logged) + wall-clock throughput (file)
feature16_audit/           # 16 — cross-check artifacts + evidence.json/.md (nothing hardcoded)
run_demo.py                # one-command runner: all 16 stages -> submission_artifacts/ + evidence
tokenizer/                 # the FROZEN artifact: vocab.json, merges.txt, manifest (committed)
tests/                     # test_*.py — invariant tests (295 passing, fully offline)
pyproject.toml             # deps + pytest config (pythonpath=".", testpaths=["tests"])
```

## Run the tests (offline, no network)

```bash
pip install pytest
pytest                 # 295 tests; datasets / huggingface_hub not required
```

## Do a real fetch (network; one-time)

```bash
pip install -e ".[dev]"            # pulls datasets + huggingface_hub
python -m feature1_collect.fetch   # fetches ALL lanes into data/raw/ (cached after first run)
```

## Run the whole pipeline (one command)

The corpus is committed, so this needs **no network and no setup** — clone and run:

```bash
python run_demo.py    # all 16 stages -> submission_artifacts/ (run.log + evidence bundle)
```

It regenerates the full evidence tree:

```
submission_artifacts/
  run.log             # complete [PASS]/[INFO] event sequence
  evidence.json       # per-requirement PASS/FAIL + pointer to the proving artifact
  evidence.md         # human-readable Requirement / Result / Evidence table
  performance.json    # packing efficiency + loss-bearing tokens/sec
  manifests/          # tokenizer_manifest, shard_index, mixture_plan, packing_report,
                      #   contrastive_delta_s, fork_lineage, corpus_summary
  ledgers/            # consumption_ledger, learning_ledger, opus_decision_ledger
  checkpoints/        # main / train / resume (model+optim+rng+offset; weights regenerated)
```

`n_steps` is a cumulative target; a later `run(n_steps=N)` resumes from the saved
`checkpoints/train` and trains only the delta (proven: incremental == monolithic).

To re-fetch the corpus from scratch instead of using the committed copy:
`python -m feature1_collect.fetch` (needs `datasets` + `huggingface_hub`).

Reproducibility: each source is pinned to an exact upstream revision (recorded in
`data/raw/fetch_log.jsonl` with a sha256), and `documents.jsonl` is written
deterministically, so a re-fetch is a cached no-op and re-runs are byte-identical.
