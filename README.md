# v5-execution-system

A small but complete, reproducible and auditable **Training Data Execution System**
for LLM pretraining (Session 6 assignment). Built epic by epic; the delivery plan lives
in the companion tracker (`session6_plan.md` / the Assignment page).

> This README is a placeholder that grows as features land. The full README, the
> one-command `run_demo.py`, and the evidence bundle are built at the final feature.

## Status

| Feature | Epic | State |
|---|---|---|
| 1 · Collecting data | 1.1 corpus data model | ✅ done |
| 1 · Collecting data | 1.2 sources manifest | ⏳ next |

## Layout (so far)

```
corpus_schema.py        # Epic 1.1 — Document / ContrastivePair records + validators
test_corpus_schema.py   # Epic 1.1 — invariant tests
```

## Run the tests

```bash
pip install -r requirements.txt
pytest -q
```
