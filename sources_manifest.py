"""Epic 1.2 — the sources manifest. A declaration of intent, not a downloader.

Says which dataset is meant to feed each lane, at what provenance tier, for
roughly how many tokens. Nothing here fetches, opens, or resolves anything:
no network, no file I/O, no `datasets`. Acquisition is Epic 1.3.

Two absences are deliberate, not oversights:
  * contrastive pairs (Epic 1.8) are not a corpus lane and do not appear here
  * the eval split (Epic 1.9) is carved later out of T0/T1 sources, which is
    what `eval_eligible` exists to identify -- it does not carve anything

And two fields are deliberately empty for now:
  * revision="" everywhere. The exact snapshot is pinned at fetch time in 1.3
    and written back, so a run can be reproduced. An empty revision means
    "not yet pinned", never "latest".
  * target_tokens are PRE-tokenizer estimates in tokens, used only to shape
    the mixture. Real counts arrive after tokenization and will differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Sequence

from corpus_schema import LANES, PROVENANCE_TIERS

__all__ = [
    "LaneSource",
    "SOURCES",
    "POOL_TARGET_TOKENS",
    "POOL_TOLERANCE",
    "EVAL_TIERS",
    "lane_totals",
    "eval_eligible",
    "validate_sources",
]

# Size of the whole pool this manifest is meant to add up to, and how far the
# sum may drift from it before that is a bug rather than rounding.
POOL_TARGET_TOKENS: Final[int] = 10_000_000
POOL_TOLERANCE: Final[float] = 0.05

# Rule 3 of the tier convention in corpus_schema: eval requires T0 or T1.
EVAL_TIERS: Final[frozenset[str]] = frozenset({"T0", "T1"})


@dataclass(frozen=True, slots=True, kw_only=True)
class LaneSource:
    """One dataset declared as a feed for one lane.

    `dataset` and `config` name the source in the upstream catalogue; nothing
    in this module resolves them. `text_field` names the column the text
    actually lives in -- FineWeb and Wikipedia use "text", but
    codeparrot/github-code-clean uses "content", and guessing wrong fails at
    fetch time rather than here. It has no default on purpose: a new source
    must state where its text is. `gated` records whether the upstream
    requires accepting terms or authenticating -- 1.3 fetches ungated sources
    only, so a gated entry is rejected here rather than failing mid-download.
    """

    source_id: str          # short unique key, stable across the project
    lane: str               # one of LANES
    dataset: str            # upstream dataset name, never resolved here
    config: str             # upstream config; carries the language. May be ""
    text_field: str         # upstream COLUMN holding the text. Not always "text"
    revision: str           # pinned at fetch in 1.3. "" = not yet pinned
    license: str            # as declared upstream
    provenance_tier: str    # one of PROVENANCE_TIERS
    target_tokens: int      # pre-tokenizer estimate, > 0
    gated: bool             # must be False for this manifest
    notes: str              # free text. May be ""


SOURCES: Final[tuple[LaneSource, ...]] = (
    LaneSource(
        source_id="web-fineweb",
        lane="web",
        dataset="HuggingFaceFW/fineweb",
        config="sample-10BT",
        text_field="text",
        revision="",
        license="ODC-BY-1.0",
        provenance_tier="T2",
        target_tokens=4_000_000,
        gated=False,
        notes="Broad English web crawl. Largest single lane by design.",
    ),
    LaneSource(
        source_id="code-github",
        lane="code",
        dataset="codeparrot/github-code-clean",
        config="",
        text_field="content",
        revision="",
        license="permissive-only (MIT/Apache/BSD filtered)",
        provenance_tier="T2",
        target_tokens=2_000_000,
        gated=False,
        notes=(
            "Exact permissive+ungated source confirmed at fetch (1.3). The "
            "licence string here describes the intended filter, not a "
            "verified per-file result."
        ),
    ),
    LaneSource(
        source_id="math-openwebmath",
        lane="math",
        dataset="open-web-math/open-web-math",
        config="",
        text_field="text",
        revision="",
        license="ODC-BY-1.0",
        provenance_tier="T2",
        target_tokens=1_200_000,
        gated=False,
        notes="Mathematical web text, already filtered upstream.",
    ),
    LaneSource(
        source_id="indic-wiki-hi",
        lane="indic",
        dataset="wikimedia/wikipedia",
        config="20231101.hi",
        text_field="text",
        revision="",
        license="CC-BY-SA-3.0",
        provenance_tier="T1",
        target_tokens=1_000_000,
        gated=False,
        notes="Hindi. Largest of the three Indic feeds.",
    ),
    LaneSource(
        source_id="indic-wiki-bn",
        lane="indic",
        dataset="wikimedia/wikipedia",
        config="20231101.bn",
        text_field="text",
        revision="",
        license="CC-BY-SA-3.0",
        provenance_tier="T1",
        target_tokens=700_000,
        gated=False,
        notes="Bengali.",
    ),
    LaneSource(
        source_id="indic-wiki-ta",
        lane="indic",
        dataset="wikimedia/wikipedia",
        config="20231101.ta",
        text_field="text",
        revision="",
        license="CC-BY-SA-3.0",
        provenance_tier="T1",
        target_tokens=500_000,
        gated=False,
        notes="Tamil.",
    ),
    LaneSource(
        source_id="mling-wiki-es",
        lane="multilingual",
        dataset="wikimedia/wikipedia",
        config="20231101.es",
        text_field="text",
        revision="",
        license="CC-BY-SA-3.0",
        provenance_tier="T1",
        target_tokens=300_000,
        gated=False,
        notes="Spanish. Non-Indic ballast for cross-lingual transfer.",
    ),
    LaneSource(
        source_id="mling-wiki-fr",
        lane="multilingual",
        dataset="wikimedia/wikipedia",
        config="20231101.fr",
        text_field="text",
        revision="",
        license="CC-BY-SA-3.0",
        provenance_tier="T1",
        target_tokens=300_000,
        gated=False,
        notes="French.",
    ),
)


def lane_totals(sources: Iterable[LaneSource]) -> dict[str, int]:
    """Target tokens summed per lane.

    Every lane in LANES appears, including any that no source feeds, so a
    caller can spot a gap by reading a zero rather than a missing key. Keys
    are inserted in sorted order, so the dict is deterministic.
    """
    totals: dict[str, int] = {lane: 0 for lane in sorted(LANES)}
    for source in sources:
        totals[source.lane] = totals.get(source.lane, 0) + source.target_tokens
    return totals


def eval_eligible(sources: Iterable[LaneSource]) -> tuple[LaneSource, ...]:
    """Sources an eval split may be carved from: tier T0 or T1 only.

    This identifies candidates. It does not carve anything -- that is 1.9.
    """
    return tuple(s for s in sources if s.provenance_tier in EVAL_TIERS)


def validate_sources(sources: Sequence[LaneSource]) -> tuple[LaneSource, ...]:
    """Raise ValueError on the first problem found, else return the sources."""
    items = tuple(sources)
    if not items:
        raise ValueError("manifest is empty: every lane in LANES needs a source")

    seen: set[str] = set()
    for index, source in enumerate(items):
        where = f"source[{index}]"

        if not isinstance(source.source_id, str) or not source.source_id.strip():
            raise ValueError(f"{where}: source_id must be a non-empty string")
        where = f"source {source.source_id!r}"

        if source.source_id in seen:
            raise ValueError(f"{where}: duplicate source_id")
        seen.add(source.source_id)

        if source.lane not in LANES:
            raise ValueError(
                f"{where}: lane must be one of {sorted(LANES)}, got {source.lane!r}"
            )
        if source.provenance_tier not in PROVENANCE_TIERS:
            raise ValueError(
                f"{where}: provenance_tier must be one of "
                f"{sorted(PROVENANCE_TIERS)}, got {source.provenance_tier!r}"
            )
        if not isinstance(source.dataset, str) or not source.dataset.strip():
            raise ValueError(f"{where}: dataset must be a non-empty string")
        if not isinstance(source.text_field, str) or not source.text_field.strip():
            raise ValueError(
                f"{where}: text_field must be a non-empty string -- it names "
                f"the upstream column the text is read from"
            )
        if not isinstance(source.license, str) or not source.license.strip():
            raise ValueError(
                f"{where}: license must be a non-empty string -- an unlicensed "
                f"source cannot be tiered"
            )
        # bool is a subclass of int, so True would otherwise slip through as 1
        if isinstance(source.target_tokens, bool) or not isinstance(
            source.target_tokens, int
        ):
            raise ValueError(
                f"{where}: target_tokens must be an int, got "
                f"{type(source.target_tokens).__name__}"
            )
        if source.target_tokens <= 0:
            raise ValueError(
                f"{where}: target_tokens must be > 0, got {source.target_tokens}"
            )
        if source.gated:
            raise ValueError(
                f"{where}: gated is True, but 1.3 fetches ungated sources only"
            )

    missing = sorted(LANES - {s.lane for s in items})
    if missing:
        raise ValueError(f"lanes with no source: {missing}")

    total = sum(s.target_tokens for s in items)
    drift = abs(total - POOL_TARGET_TOKENS) / POOL_TARGET_TOKENS
    if drift > POOL_TOLERANCE:
        raise ValueError(
            f"grand total {total:,} is {drift:.1%} from POOL_TARGET_TOKENS "
            f"{POOL_TARGET_TOKENS:,}, outside the {POOL_TOLERANCE:.0%} tolerance"
        )
    return items
