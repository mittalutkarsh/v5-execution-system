"""Feature 7 — the OPUS selector (epics 7.1-7.5).

OPUS decides, per candidate unit of data, whether to accept / reject / defer it
into the training mixture:

  * scoring interface (7.1): a pluggable score(candidate, cfg) -> [0, 1].
  * accept/reject/defer rule (7.2): above the accept threshold -> accept while
    the lane target has room; between the thresholds -> defer; below -> reject.
  * protected-floor override (7.3): while a lane sits below its protected floor,
    accept regardless of score, so the India-first floors are always filled.
  * ΔS surprisal hook (7.4): a candidate carries delta_s = S(y-) - S(y+); the
    default scorer lets it boost the score. Until the trainer exists (Feature
    10) delta_s is neutral (0.0), but the wiring is here.
  * decision ledger (7.5): every decision is recorded, append-only, so the
    selection is fully auditable and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = [
    "TIER_QUALITY", "Candidate", "SelectorConfig", "default_score", "OpusSelector",
]

# higher provenance -> higher base quality (deterministic, documented)
TIER_QUALITY = {"T0": 1.0, "T1": 0.9, "T2": 0.6, "T3": 0.4}


@dataclass(frozen=True)
class Candidate:
    id: str
    lane: str
    tokens: int
    quality: float          # base quality signal in [0, 1]
    delta_s: float = 0.0    # ΔS = S(y-) - S(y+); neutral until the trainer runs


@dataclass(frozen=True)
class SelectorConfig:
    lane_targets: Mapping[str, int]         # planned tokens per lane (from mixture)
    lane_floors: Mapping[str, int] = field(default_factory=dict)  # protected floor tokens
    accept_threshold: float = 0.5
    reject_threshold: float = 0.2
    delta_s_weight: float = 0.3


def default_score(c: Candidate, cfg: SelectorConfig) -> float:
    """Base quality nudged by the ΔS surprisal signal, clamped to [0, 1]."""
    return max(0.0, min(1.0, c.quality + cfg.delta_s_weight * c.delta_s))


class OpusSelector:
    """Runs the accept/reject/defer rule over a stream of candidates."""

    def __init__(
        self,
        cfg: SelectorConfig,
        score: Callable[[Candidate, SelectorConfig], float] = default_score,
    ) -> None:
        self.cfg = cfg
        self.score_fn = score
        self.lane_total: dict[str, int] = {lane: 0 for lane in cfg.lane_targets}
        self.ledger: list[dict[str, Any]] = []

    def offer(self, c: Candidate) -> str:
        """Decide on one candidate, updating the running tally and the ledger."""
        score = self.score_fn(c, self.cfg)
        used = self.lane_total.get(c.lane, 0)
        target = self.cfg.lane_targets.get(c.lane, 0)
        floor = self.cfg.lane_floors.get(c.lane, 0)

        if used < floor:
            decision, reason = "accept", "floor_override"
        elif used >= target:
            decision, reason = "reject", "lane_target_reached"
        elif score >= self.cfg.accept_threshold:
            decision, reason = "accept", "score>=accept"
        elif score >= self.cfg.reject_threshold:
            decision, reason = "defer", "marginal_score"
        else:
            decision, reason = "reject", "score<reject"

        if decision == "accept":
            self.lane_total[c.lane] = used + c.tokens

        self.ledger.append({
            "candidate_id": c.id, "lane": c.lane, "tokens": c.tokens,
            "score": round(score, 4), "delta_s": round(c.delta_s, 4),
            "decision": decision, "reason": reason,
            "lane_total_after": self.lane_total.get(c.lane, 0),
        })
        return decision

    def run(self, candidates) -> "OpusSelector":
        for c in candidates:
            self.offer(c)
        return self

    def summary(self) -> dict[str, Any]:
        counts = {"accept": 0, "reject": 0, "defer": 0}
        for row in self.ledger:
            counts[row["decision"]] += 1
        floors_met = all(
            self.lane_total.get(lane, 0) >= floor
            for lane, floor in self.cfg.lane_floors.items()
        )
        return {
            "decisions": counts,
            "accepted_tokens": {k: self.lane_total[k] for k in sorted(self.lane_total)},
            "floors_met": floors_met,
        }

    def write_ledger(self, out_path: str) -> None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for row in self.ledger:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
