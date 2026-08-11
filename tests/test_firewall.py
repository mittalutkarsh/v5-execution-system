"""Feature 5 tests — the evaluation firewall. Fully offline."""

from __future__ import annotations

import pytest

from feature5_firewall.firewall import EvalFirewall, FirewallViolation, eval_shards_blocked

INDEX = {
    "kind": "shard_index",
    "shards": [
        {"shard_id": "train-web-0000", "split": "train", "lane": "web"},
        {"shard_id": "train-indic-0000", "split": "train", "lane": "indic"},
        {"shard_id": "eval-indic-0000", "split": "eval", "lane": "indic"},
        {"shard_id": "eval-mling-0000", "split": "eval", "lane": "multilingual"},
    ],
}


def test_partition_is_by_split() -> None:
    fw = EvalFirewall(INDEX)
    assert set(fw.train_pool()) == {"train-web-0000", "train-indic-0000"}
    assert fw.eval_ids == {"eval-indic-0000", "eval-mling-0000"}


def test_admit_passes_train_and_blocks_eval() -> None:
    fw = EvalFirewall(INDEX)
    assert fw.admit("train-web-0000") == "train-web-0000"
    with pytest.raises(FirewallViolation, match="may not enter a training batch"):
        fw.admit("eval-indic-0000")


def test_no_eval_shard_is_in_the_train_pool() -> None:
    fw = EvalFirewall(INDEX)
    assert not (set(fw.train_pool()) & fw.eval_ids)


def test_audit_reports_disjoint_partition() -> None:
    audit = EvalFirewall(INDEX).audit()
    assert audit == {
        "train_shards": 2, "eval_shards": 2, "disjoint": True, "overlap": [],
        "eval_ids": ["eval-indic-0000", "eval-mling-0000"],
    }


def test_eval_shards_blocked_summary() -> None:
    result = eval_shards_blocked(INDEX)
    assert result["ok"] is True
    assert result["blocked"] == 2 and result["eval_shards"] == 2
