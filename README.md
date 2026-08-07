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
| 1 · Collecting data | 1.8 author contrastive pairs | ⏳ next |

The full ~10M-token pool is fetched and hash-verified across all five lanes
(web 4.0M, code 2.0M, math 1.2M, indic 2.2M, multilingual 0.6M; 13,087 docs).
The code lane uses `Nan-Do/code-search-net-python` — chosen at fetch because the
originally-declared script-based/gated code sources would not load ungated.

## Layout (so far)

```
corpus_schema.py        # 1.1 — Document / ContrastivePair records + validators
sources_manifest.py     # 1.2 — LaneSource, SOURCES (10M pool), validate_sources
fetch.py                # 1.3–1.7 — fetch_source / fetch_all: stream -> documents.jsonl + fetch_log + sha256
test_*.py               # invariant tests (59 passing, fully offline)
pyproject.toml          # deps + pytest config (pythonpath=".")
```

## Run the tests (offline, no network)

```bash
pip install pytest
pytest                 # 59 tests; datasets / huggingface_hub not required
```

## Do a real fetch (network; one-time)

```bash
pip install -e ".[dev]"   # pulls datasets + huggingface_hub
python fetch.py           # fetches ALL lanes into data/raw/ (cached after first run)
```

Reproducibility: each source is pinned to an exact upstream revision (recorded in
`data/raw/fetch_log.jsonl` with a sha256), and `documents.jsonl` is written
deterministically, so a re-fetch is a cached no-op and re-runs are byte-identical.
