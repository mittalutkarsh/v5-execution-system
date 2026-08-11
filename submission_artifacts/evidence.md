# Evidence — Training Data Execution System

**Overall: ALL REQUIREMENTS PASS**

| Requirement | Result | Evidence |
|---|---|---|
| Tokenizer integrity | PASS | Manifest record — `manifests/tokenizer_manifest.json` |
| Evaluation firewall | PASS | Blocked-shard event — `run.log ([PASS] eval_shard_blocked)` |
| Packing correctness | PASS | Packed-batch report — `manifests/packing_report.json` |
| Mixture compliance | PASS | Planned versus actual shares — `manifests/mixture_plan.json` |
| OPUS audit trail | PASS | Candidate decision records — `ledgers/opus_decision_ledger.jsonl` |
| Crash recovery | PASS | Expected and resumed batch ids — `run.log ([PASS] resume_next_batch_matched)` |
| Replay | PASS | Original and replay hashes — `run.log ([PASS] replay_hash_matched)` |
| Learning trace | PASS | Loss linked to source data — `ledgers/learning_ledger.jsonl` |
| Throughput | PASS | Performance report — `performance.json` |

## Run summary
- Tokenizer: vocab 12000, 11740 merges, hash `3eb8d6c50b13dc27…`
- Shards: 171 shards, 10,423,262 tokens ({'eval': {'docs': 29, 'shards': 3, 'tokens': 113811}, 'train': {'docs': 13026, 'shards': 168, 'tokens': 10309451}})
- Training: 300 steps, loss 9.563157 → 8.242889 (300 batches consumed)
- Checkpoint offset: 4; fork `branch-a` diverged=True
- Packing efficiency: 0.8891546753379762; contrastive pairs: 36

_Every value above is read back from a written artifact; none is hardcoded._
