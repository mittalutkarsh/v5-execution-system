"""Feature 6 tests — mixture config + compiler + floors. Fully offline."""

from __future__ import annotations

import pytest

from feature6_mixture.compile_mixture import build_report, compile_mixture
from feature6_mixture.mixture_config import DEFAULT_MIXTURE, MixtureConfig, Phase


def test_default_mixture_validates() -> None:
    assert DEFAULT_MIXTURE.validate() is DEFAULT_MIXTURE
    assert DEFAULT_MIXTURE.lanes() == ["code", "indic", "math", "multilingual", "web"]


def test_phase_budgets_sum_to_total_exactly() -> None:
    plan = compile_mixture(DEFAULT_MIXTURE)
    assert sum(p["budget"] for p in plan) == DEFAULT_MIXTURE.total_tokens


def test_lane_tokens_sum_to_phase_budget_exactly() -> None:
    for phase in compile_mixture(DEFAULT_MIXTURE):
        assert sum(l["planned_tokens"] for l in phase["lanes"].values()) == phase["budget"]


def test_protected_floors_are_always_met() -> None:
    report = build_report(DEFAULT_MIXTURE)
    assert report["all_floors_met"] is True
    for phase in report["phases"]:
        for lane, info in phase["lanes"].items():
            assert info["planned_share"] >= info["floor"] - 1e-9


def test_floor_lifts_a_lane_above_its_weight() -> None:
    # multilingual has a tiny weight but a real floor -> the floor must win
    cfg = MixtureConfig(
        total_tokens=100_000,
        phases=(Phase("p", 1.0,
                      weights={"web": 0.95, "multilingual": 0.05},
                      floors={"multilingual": 0.20}),),
    )
    lanes = compile_mixture(cfg)[0]["lanes"]
    assert lanes["multilingual"]["planned_share"] >= 0.20 - 1e-9


def test_compile_is_deterministic() -> None:
    assert compile_mixture(DEFAULT_MIXTURE) == compile_mixture(DEFAULT_MIXTURE)


def test_coverage_flags_repetition_when_target_exceeds_available() -> None:
    report = build_report(DEFAULT_MIXTURE, available={"indic": 100, "web": 10**9})
    assert report["coverage"]["indic"]["needs_repetition"] is True
    assert report["coverage"]["web"]["needs_repetition"] is False


def test_bad_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        MixtureConfig(total_tokens=1000, phases=(
            Phase("a", 0.3, {"web": 1.0}), Phase("b", 0.3, {"web": 1.0}),
        )).validate()
