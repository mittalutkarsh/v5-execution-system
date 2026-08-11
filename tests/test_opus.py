"""Feature 7 tests — the OPUS selector. Fully offline."""

from __future__ import annotations

from feature7_opus.opus_selector import Candidate, OpusSelector, SelectorConfig, default_score


def test_accept_reject_defer_by_score() -> None:
    cfg = SelectorConfig(lane_targets={"web": 10_000})
    sel = OpusSelector(cfg)
    assert sel.offer(Candidate("a", "web", 100, quality=0.9)) == "accept"
    assert sel.offer(Candidate("b", "web", 100, quality=0.35)) == "defer"
    assert sel.offer(Candidate("c", "web", 100, quality=0.10)) == "reject"


def test_lane_target_stops_accepting() -> None:
    cfg = SelectorConfig(lane_targets={"web": 150})
    sel = OpusSelector(cfg)
    assert sel.offer(Candidate("a", "web", 100, quality=0.9)) == "accept"
    # lane now has 100/150; a second 100-token high-quality doc still fits (under target)...
    assert sel.offer(Candidate("b", "web", 100, quality=0.9)) == "accept"
    # ...now at 200 >= 150, further accepts are rejected as target reached
    d = sel.offer(Candidate("c", "web", 100, quality=0.9))
    assert d == "reject" and sel.ledger[-1]["reason"] == "lane_target_reached"


def test_floor_override_accepts_low_quality() -> None:
    cfg = SelectorConfig(lane_targets={"indic": 10_000}, lane_floors={"indic": 500})
    sel = OpusSelector(cfg)
    d = sel.offer(Candidate("x", "indic", 100, quality=0.01))  # terrible score...
    assert d == "accept" and sel.ledger[-1]["reason"] == "floor_override"


def test_delta_s_boosts_the_score() -> None:
    cfg = SelectorConfig(lane_targets={"web": 10_000}, delta_s_weight=0.5)
    marginal = Candidate("m", "web", 100, quality=0.35, delta_s=0.0)
    boosted = Candidate("b", "web", 100, quality=0.35, delta_s=0.6)  # +0.30 -> 0.65
    assert default_score(marginal, cfg) < cfg.accept_threshold
    assert default_score(boosted, cfg) >= cfg.accept_threshold


def test_ledger_records_every_decision_in_order() -> None:
    cfg = SelectorConfig(lane_targets={"web": 10_000})
    cands = [Candidate(f"d{i}", "web", 100, quality=0.9) for i in range(5)]
    sel = OpusSelector(cfg).run(cands)
    assert [r["candidate_id"] for r in sel.ledger] == ["d0", "d1", "d2", "d3", "d4"]
    assert sel.ledger[-1]["lane_total_after"] == 500


def test_selection_is_deterministic() -> None:
    cfg = SelectorConfig(lane_targets={"web": 300}, lane_floors={"web": 100})
    cands = [Candidate(f"d{i}", "web", 100, quality=(i % 3) / 3) for i in range(6)]
    a = OpusSelector(cfg).run(cands).ledger
    b = OpusSelector(cfg).run(cands).ledger
    assert a == b


def test_summary_reports_floor_satisfaction() -> None:
    cfg = SelectorConfig(lane_targets={"indic": 1000}, lane_floors={"indic": 200})
    sel = OpusSelector(cfg).run([Candidate("x", "indic", 250, quality=0.9)])
    s = sel.summary()
    assert s["floors_met"] is True and s["accepted_tokens"]["indic"] == 250
