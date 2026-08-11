"""Epic 6.1 — the mixture / curriculum config.

A curriculum is a sequence of phases. Each phase takes a fraction of the total
training budget and mixes the lanes by relative weights, subject to PROTECTED
FLOORS -- minimum shares that guarantee the India-first lanes (indic,
multilingual) are never starved, no matter what the weights say. This is data
only; the compiler (6.2) turns it into concrete per-lane token targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = ["Phase", "MixtureConfig", "DEFAULT_MIXTURE", "TRAINING_BUDGET_TOKENS"]

TRAINING_BUDGET_TOKENS = 3_000_000  # ~0.3 keep-fraction of the ~10M pool


@dataclass(frozen=True)
class Phase:
    name: str
    fraction: float                       # share of the total budget in this phase
    weights: Mapping[str, float]          # relative lane weights
    floors: Mapping[str, float] = field(default_factory=dict)  # min share per lane


@dataclass(frozen=True)
class MixtureConfig:
    total_tokens: int
    phases: tuple[Phase, ...]

    def lanes(self) -> list[str]:
        seen: dict[str, None] = {}
        for ph in self.phases:
            for lane in ph.weights:
                seen.setdefault(lane, None)
        return sorted(seen)

    def validate(self) -> "MixtureConfig":
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if not self.phases:
            raise ValueError("a mixture needs at least one phase")
        if abs(sum(p.fraction for p in self.phases) - 1.0) > 1e-6:
            raise ValueError("phase fractions must sum to 1.0")
        for ph in self.phases:
            if ph.fraction <= 0:
                raise ValueError(f"phase {ph.name!r} fraction must be positive")
            if not ph.weights or sum(ph.weights.values()) <= 0:
                raise ValueError(f"phase {ph.name!r} needs positive lane weights")
            if any(w < 0 for w in ph.weights.values()):
                raise ValueError(f"phase {ph.name!r} has a negative weight")
            if any(not (0 <= f <= 1) for f in ph.floors.values()):
                raise ValueError(f"phase {ph.name!r} floors must be in [0, 1]")
            if sum(ph.floors.values()) > 1 + 1e-9:
                raise ValueError(f"phase {ph.name!r} floors sum to more than the budget")
            for lane in ph.floors:
                if lane not in ph.weights:
                    raise ValueError(f"phase {ph.name!r} floors an unweighted lane {lane!r}")
        return self


# India-first curriculum: broad warmup, then an India-emphasis phase. The floors
# guarantee indic/multilingual never fall below a protected share.
DEFAULT_MIXTURE = MixtureConfig(
    total_tokens=TRAINING_BUDGET_TOKENS,
    phases=(
        Phase(
            name="warmup",
            fraction=0.3,
            weights={"web": 0.40, "code": 0.20, "math": 0.15, "indic": 0.20, "multilingual": 0.05},
            floors={"indic": 0.15, "multilingual": 0.03},
        ),
        Phase(
            name="india_emphasis",
            fraction=0.7,
            weights={"web": 0.25, "code": 0.15, "math": 0.10, "indic": 0.40, "multilingual": 0.10},
            floors={"indic": 0.35, "multilingual": 0.08},
        ),
    ),
).validate()
