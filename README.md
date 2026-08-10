# v5-execution-system

A small but complete, reproducible and auditable **Training Data Execution System**
for LLM pretraining (Session 6 assignment). Built epic by epic; the delivery plan lives
in the companion tracker (`session6_plan.md` / the Assignment page).

> This README grows as features land. The full README, the one-command `run_demo.py`,
> and the evidence bundle are built at the final feature.

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
| 3 · Frozen BPE tokenizer | 3.1 BPE trainer | ⏳ next |

The full ~10M-token pool is fetched and hash-verified across all five lanes
(web 4.0M, code 2.0M, math 1.2M, indic 2.2M, multilingual 0.6M; 13,087 docs).
The code lane uses `Nan-Do/code-search-net-python` — chosen at fetch because the
originally-declared script-based/gated code sources would not load ungated.

## Layout (so far)

```
corpus_schema.py        # 1.1 — Document / ContrastivePair records + validators
sources_manifest.py     # 1.2 — LaneSource, SOURCES (10M pool), validate_sources
fetch.py                # 1.3–1.7 — fetch_source / fetch_all: stream -> documents.jsonl + fetch_log + sha256
contrastive_pairs.py    # 1.8 — 36 hand-authored Indian-vantage vs Western-default pairs
eval_split.py           # 1.9 — select_eval / carve_eval: deterministic held-out eval (T1 only)
corpus_loader.py        # 1.10 — iter_documents / corpus_counts / load_corpus (eval routed by manifest)
corpus_report.py        # 1.11 — build/write corpus_summary.json (deterministic, regenerable)
run_demo.py             # 1.12 + 2.7 — one-command runner: load-corpus + clean-corpus stages
text_normalize.py       # 2.1 — normalize_text/normalize_document (NFC, deterministic, idempotent)
content_hash.py         # 2.2 — content_hash + dedup_exact
quality_filter.py       # 2.3 — quality_ok / filter_quality (length, symbol, repetition)
near_dedup.py           # 2.4 — MinHash + LSH near-duplicate removal (pure Python, deterministic)
pii_scrub.py            # 2.5 — scrub_pii (email/phone -> placeholders, idempotent)
decontaminate.py        # 2.6 — n-gram decontamination of train vs eval + contrastive
clean_pipeline.py       # 2.7 — clean_corpus: compose 2.1-2.6 -> data/clean + cleaning_report
test_*.py               # invariant tests (162 passing, fully offline)
pyproject.toml          # deps + pytest config (pythonpath=".")
```

## Run the tests (offline, no network)

```bash
pip install pytest
pytest                 # 162 tests; datasets / huggingface_hub not required
```

## Do a real fetch (network; one-time)

```bash
pip install -e ".[dev]"   # pulls datasets + huggingface_hub
python fetch.py           # fetches ALL lanes into data/raw/ (cached after first run)
```

Reproducibility: each source is pinned to an exact upstream revision (recorded in
`data/raw/fetch_log.jsonl` with a sha256), and `documents.jsonl` is written
deterministically, so a re-fetch is a cached no-op and re-runs are byte-identical.
