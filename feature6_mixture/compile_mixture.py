"""Epics 6.2-6.4 — compile a mixture config into per-lane token targets.

For each phase: reserve the protected floors first, then distribute the
remaining budget by the lane weights. Because floors are reserved up front,
every lane provably meets its floor (Epic 6.3). Integer token counts are
apportioned by the largest-remainder method so per-phase totals are exact and
deterministic. build_report() emits the planned shares (Epic 6.4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from feature6_mixture.mixture_config import MixtureConfig

__all__ = ["compile_mixture", "build_report", "write_report", "REPORT_KIND"]

REPORT_KIND = "mixture_plan"
_JSON = {"ensure_ascii": False, "sort_keys": True, "indent": 2}


def _apportion(budget: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Split `budget` into integers proportional to `weights`, summing exactly."""
    total = sum(weights.values())
    if total <= 0:
        return {k: 0 for k in weights}
    exact = {k: budget * w / total for k, w in weights.items()}
    floor = {k: int(v) for k, v in exact.items()}
    remainder = budget - sum(floor.values())
    # hand the leftover tokens to the largest fractional parts (ties: lane name)
    order = sorted(exact, key=lambda k: (-(exact[k] - floor[k]), k))
    for k in order[:remainder]:
        floor[k] += 1
    return floor


def compile_mixture(config: MixtureConfig) -> list[dict[str, Any]]:
    """Return a per-phase plan: [{name, budget, lanes:{lane:{planned_tokens,...}}}]."""
    config.validate()
    phase_budgets = _apportion(
        config.total_tokens, {p.name: p.fraction for p in config.phases}
    )
    plan: list[dict[str, Any]] = []
    for ph in config.phases:
        budget = phase_budgets[ph.name]
        floor_tokens = {lane: int(ph.floors.get(lane, 0.0) * budget) for lane in ph.weights}
        remainder = budget - sum(floor_tokens.values())
        weight_alloc = _apportion(remainder, dict(ph.weights))
        planned = {lane: floor_tokens[lane] + weight_alloc.get(lane, 0) for lane in ph.weights}
        lanes = {
            lane: {
                "planned_tokens": planned[lane],
                "planned_share": (planned[lane] / budget) if budget else 0.0,
                "floor": ph.floors.get(lane, 0.0),
                "meets_floor": planned[lane] >= floor_tokens[lane],
            }
            for lane in sorted(ph.weights)
        }
        plan.append({"name": ph.name, "budget": budget, "lanes": lanes})
    return plan


def build_report(config: MixtureConfig, available: Mapping[str, int] | None = None) -> dict[str, Any]:
    """A full mixture report: the plan, per-lane totals, and (optional) coverage."""
    plan = compile_mixture(config)
    lane_totals: dict[str, int] = {}
    for phase in plan:
        for lane, info in phase["lanes"].items():
            lane_totals[lane] = lane_totals.get(lane, 0) + info["planned_tokens"]
    report: dict[str, Any] = {
        "kind": REPORT_KIND,
        "total_tokens": config.total_tokens,
        "phases": plan,
        "lane_totals": {k: lane_totals[k] for k in sorted(lane_totals)},
        "all_floors_met": all(
            info["meets_floor"] for phase in plan for info in phase["lanes"].values()
        ),
    }
    if available is not None:
        report["coverage"] = {
            lane: {
                "planned": lane_totals.get(lane, 0),
                "available": available.get(lane, 0),
                "needs_repetition": lane_totals.get(lane, 0) > available.get(lane, 0),
            }
            for lane in sorted(set(lane_totals) | set(available))
        }
    return report


def write_report(report: dict[str, Any], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, **_JSON) + "\n")
