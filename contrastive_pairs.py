"""Epic 1.8 — the contrastive perspective corpus. Hand-written, not fetched.

Minimal pairs on contested, globally-framed topics. One shared prefix, two
continuations that differ only in a localized framing span: `y_plus` takes the
Indian vantage, `y_minus` the Western default. No downloads, no tokenizer, no
file I/O, no network — every string below was authored by hand.

The design guard, restated because it is the whole point:

    `y_plus` is a FACTUAL, historically defensible Indian framing. It is never
    a chauvinistic one. `y_minus` is a genuine Western-default framing, not a
    strawman — each is something a reasonable Western source actually writes.
    The contrast is about VANTAGE POINT, not about disparaging anyone, and
    `chauvinism="none"` is the assertion that this held for every pair.

Two consequences of that guard worth stating:

  * Topics were chosen only where the Indian framing is defensible on the
    record. Claims that would need pseudo-history to support are excluded on
    purpose — a chauvinistic or false `y_plus` would poison the signal this
    corpus exists to carry.
  * Where the divergence is about institutional standing rather than fact
    (Ayurveda, yoga), `y_plus` describes the standing and makes no efficacy
    claim. Neither continuation asserts anything the other denies.

Pairs are built from a specification table rather than 36 hand-typed
constructor calls: the factory mints ids, vantage and chauvinism uniformly, so
those three cannot drift across entries. The prose — the part that is actually
authored — stays readable as four fields per row.
"""

from __future__ import annotations

from typing import Final, Iterable, Sequence

from corpus_schema import (
    CHAUVINISM,
    VANTAGE,
    ContrastivePair,
    validate_contrastive,
)

__all__ = ["CONTRASTIVE_PAIRS", "validate_all"]


# (topic, prefix, y_plus, y_minus)
_SPEC: Final[tuple[tuple[str, str, str, str], ...]] = (
    # ---------------------------------------------------------------- history
    (
        "colonial economic impact",
        "The economic impact of British colonial rule on India was",
        "a sustained transfer of wealth that deindustrialised Bengal's textile "
        "economy and held per-capita income roughly flat for close to two centuries.",
        "a mixed legacy that introduced railways, a professional civil service, "
        "and modern administrative institutions.",
    ),
    (
        "1857",
        "The events of 1857 across northern India are best described as",
        "the first large-scale war of independence, in which soldiers, peasants "
        "and displaced nobility fought together against Company rule.",
        "the Sepoy Mutiny, an uprising among Indian troops in the Bengal Army "
        "that was suppressed within a year.",
    ),
    (
        "Partition",
        "The Partition of British India in 1947 is remembered as",
        "a hurried administrative division drawn in about five weeks by a "
        "boundary commission, displacing some fifteen million people.",
        "the transfer of power that created two independent dominions and ended "
        "British rule in South Asia.",
    ),
    (
        "Bengal famine 1943",
        "The Bengal famine of 1943 was",
        "a policy failure in which wartime procurement, export priorities and "
        "the denial of boats deepened a shortage into a famine that killed two "
        "to three million people.",
        "a wartime supply crisis in the region, worsened by a cyclone and the "
        "loss of rice imports from Burma.",
    ),
    (
        "East India Company",
        "The English East India Company is best understood as",
        "a chartered corporation with its own standing army, which conquered "
        "territory and collected land revenue for a century before the Crown "
        "took over.",
        "a pioneering joint-stock trading company whose commercial success "
        "gradually drew it into Indian politics.",
    ),
    (
        "Indian soldiers in the world wars",
        "The Indian Army's role in the two world wars was",
        "immense and under-commemorated: about 1.3 million served in the First "
        "World War and around 2.5 million volunteered in the Second, the largest "
        "volunteer force ever raised.",
        "a significant colonial contribution of troops and materiel to the "
        "Allied war effort.",
    ),
    (
        "Jallianwala Bagh",
        "The killings at Jallianwala Bagh in April 1919 are described as",
        "a massacre, in which troops fired without warning on an enclosed and "
        "unarmed gathering, and which became a turning point in the independence "
        "movement.",
        "a controversial episode during a period of civil unrest in Punjab, "
        "later criticised in Parliament.",
    ),
    # ------------------------------------------------------------ trade, money
    (
        "spice trade",
        "The spice trade between Europe and India is usually framed as",
        "European entry into commercial networks that had already connected the "
        "Malabar coast to Arabia, East Africa and Southeast Asia for over a "
        "thousand years.",
        "an achievement of the Age of Discovery, when European navigators opened "
        "a sea route to the East.",
    ),
    (
        "cotton textiles",
        "Before industrialisation, Indian cotton textiles were",
        "the world's dominant manufactured export, out-competing British cloth "
        "until tariffs and duties reversed the direction of the trade.",
        "a traditional craft industry that was eventually displaced by "
        "mechanised mills in Lancashire.",
    ),
    (
        "financial year",
        "For a company registered in Mumbai, the financial year begins in",
        "April and closes on 31 March, the period the Companies Act, 2013 fixes "
        "for Indian companies.",
        "January and closes on 31 December, following the calendar year.",
    ),
    (
        "lakh and crore",
        "Writing a figure as 4,50,00,000 is",
        "the standard Indian numbering convention, where lakh and crore mark the "
        "natural places and appear in statutes, budgets and price lists.",
        "an unusual digit grouping; the conventional form would be 45,000,000.",
    ),
    (
        "share of world output",
        "India's place in the world economy around 1700 was",
        "that of one of its largest producers, accounting on Maddison's estimates "
        "for roughly a quarter of global output.",
        "that of a pre-industrial agrarian economy that had not yet begun to "
        "modernise.",
    ),
    # ------------------------------------------------------- science, industry
    (
        "space programme",
        "India's space programme is best characterised by",
        "cost-disciplined engineering: a Mars orbiter that succeeded on the first "
        "attempt in 2014, and the first soft landing near the lunar south pole "
        "in 2023.",
        "the rapid progress of an emerging space nation catching up with "
        "established programmes.",
    ),
    (
        "zero and place value",
        "The decimal place-value system with zero originated",
        "in Indian mathematics, where Brahmagupta set out rules for zero as a "
        "number in 628 CE, reaching Europe centuries later through Arabic "
        "transmission.",
        "in the Arab world, which is why the digits are called Arabic numerals "
        "in English.",
    ),
    (
        "pharmaceutical industry",
        "India's pharmaceutical industry is known globally for",
        "producing a large share of the world's vaccine doses and supplying "
        "affordable medicines across Africa, Asia and Latin America.",
        "manufacturing low-cost generic versions of drugs developed elsewhere.",
    ),
    (
        "IT services",
        "The Indian IT services industry is usually described as",
        "a distributed engineering capability built out of the Y2K remediation "
        "effort, now running critical systems for much of the Fortune 500.",
        "an offshore outsourcing sector built on lower labour costs.",
    ),
    # ---------------------------------------------------------------- language
    (
        "languages and vernaculars",
        "The languages spoken across India's states are",
        "full languages with their own scripts, literatures and classical "
        "traditions, several with more speakers than most European national "
        "languages.",
        "regional vernaculars and dialects, spoken alongside Hindi and English.",
    ),
    (
        "English in India",
        "English in India today is",
        "one language within a multilingual repertoire, with Indian English "
        "recognised as an established variety having its own conventions.",
        "a colonial inheritance that happens to serve as the country's link "
        "language.",
    ),
    (
        "multilingualism",
        "An Indian who speaks three languages fluently is",
        "unremarkable; multilingualism is the ordinary condition, and most people "
        "move between a home language, a state language and a link language daily.",
        "notably accomplished, given how few people anywhere are trilingual.",
    ),
    (
        "Panini's grammar",
        "Panini's grammar of Sanskrit is",
        "a formal generative description of a language, written around the fourth "
        "century BCE, whose rule notation anticipates ideas in modern linguistics "
        "and computer science.",
        "an early treatise on the grammar of an ancient liturgical language.",
    ),
    # -------------------------------------------------------------------- food
    (
        "curry",
        "The word curry, as used on English menus, is",
        "a colonial-era catch-all that flattens dozens of distinct regional "
        "cuisines and techniques into a single imagined dish.",
        "a convenient category for spiced South Asian dishes served with rice "
        "or bread.",
    ),
    (
        "spice",
        "The use of spices in Indian cooking is",
        "a matter of layering aroma in sequence, where most spices contribute no "
        "heat at all and are bloomed in a particular order.",
        "what makes the food hot, though dishes can usually be ordered at a "
        "milder spice level.",
    ),
    (
        "vegetarian cooking",
        "Vegetarian cooking in India is",
        "a developed culinary tradition spanning regional cuisines, with whole "
        "categories of dish that never treat meat as an absence.",
        "a widely available option, reflecting religious dietary restrictions.",
    ),
    (
        "chai",
        "In an Indian household, the drink called chai is",
        "tea brewed together with milk, sugar and spices as one preparation, "
        "using leaf grown in Assam and the Nilgiris; the word simply means tea.",
        "a spiced tea latte, usually made from a syrup or a concentrate.",
    ),
    # ------------------------------------------------------------------- sport
    (
        "cricket",
        "Within the world of the sport, cricket in India is",
        "the centre of the sport's global economy, supplying most of its revenue "
        "and its largest playing and viewing base.",
        "a legacy of British rule that became unusually popular in the "
        "subcontinent.",
    ),
    (
        "field hockey",
        "India's Olympic record in field hockey is",
        "the strongest of any nation, with eight gold medals including six in "
        "succession between 1928 and 1956.",
        "a historical footnote from an era before the country turned its "
        "attention to cricket.",
    ),
    # ------------------------------------------------------ politics, policy
    (
        "non-alignment",
        "India's foreign policy during the Cold War is best read as",
        "a deliberate doctrine of strategic autonomy, refusing bloc membership "
        "while dealing with both superpowers on its own terms.",
        "a position of neutrality between the two blocs, sometimes read as "
        "indecision.",
    ),
    (
        "universal franchise",
        "India adopting universal adult franchise at independence was",
        "an immediate constitutional commitment, extended to every adult at the "
        "first general election in 1951-52 with no property or literacy "
        "qualification.",
        "an ambitious experiment, given that most of the newly enfranchised were "
        "poor and could not read.",
    ),
    (
        "Green Revolution",
        "The Green Revolution in Indian agriculture is remembered as",
        "the period when the country moved from importing food to cereal "
        "self-sufficiency, at a lasting cost in groundwater depletion and "
        "regional imbalance.",
        "a successful transfer of Western agricultural technology and "
        "high-yielding seed varieties.",
    ),
    (
        "monsoon",
        "For most of rural India, the southwest monsoon is",
        "the axis the agricultural year turns on, with sowing calendars, credit "
        "cycles and budget forecasts all keyed to the date of onset over Kerala.",
        "the rainy season, a period of heavy rainfall and occasional flooding.",
    ),
    # ------------------------------------------------- time, date, convention
    (
        "date format",
        "The date 03/04/2024 written in India refers to",
        "3 April, following the day-month-year order used in Indian official and "
        "everyday writing.",
        "March 4, following the month-day-year order.",
    ),
    (
        "Indian Standard Time",
        "Compared with other national time zones, Indian Standard Time is",
        "a single national zone at UTC+05:30, a deliberate half-hour offset that "
        "keeps one clock across a span wide enough for two.",
        "an unusual time zone, offset by half an hour from the standard hourly "
        "divisions.",
    ),
    (
        "festival dates",
        "Indian festival dates falling on different days each year happens because",
        "they follow lunisolar calendars, in which lunar months are corrected "
        "against the solar year so that each festival stays in its season.",
        "they are set by traditional calendars and do not have fixed dates.",
    ),
    (
        "Diwali",
        "Explained to someone outside India, the festival of Diwali is",
        "a multi-day festival observed differently across regions, marking "
        "distinct events and, in several states, opening the accounting year.",
        "the festival of lights, sometimes described as the Indian equivalent "
        "of Christmas.",
    ),
    # ------------------------------------------------------ knowledge systems
    (
        "yoga",
        "Yoga, in its original formulation, is",
        "a philosophical system of eight limbs, of which physical posture is one; "
        "the textual tradition is concerned mainly with ethics, breath and "
        "attention.",
        "a physical practice of postures and stretching, widely taught as "
        "low-impact exercise.",
    ),
    (
        "Ayurveda",
        "Within the Indian health system, Ayurveda is",
        "a codified medical system with five-and-a-half-year degree programmes, "
        "state regulation under the Ministry of Ayush, and its own pharmacopoeia.",
        "a traditional wellness practice, usually grouped with complementary and "
        "alternative medicine.",
    ),
)


def _build(spec: Sequence[tuple[str, str, str, str]]) -> tuple[ContrastivePair, ...]:
    """Mint ids, vantage and chauvinism uniformly so they cannot drift."""
    return tuple(
        ContrastivePair(
            id=f"pair-{index:04d}",
            topic=topic,
            prefix=prefix,
            y_plus=y_plus,
            y_minus=y_minus,
            vantage=VANTAGE,
            chauvinism=CHAUVINISM,
        )
        for index, (topic, prefix, y_plus, y_minus) in enumerate(spec, start=1)
    )


CONTRASTIVE_PAIRS: Final[tuple[ContrastivePair, ...]] = _build(_SPEC)


def validate_all(
    pairs: Iterable[ContrastivePair] = CONTRASTIVE_PAIRS,
) -> tuple[ContrastivePair, ...]:
    """Validate every pair and reject duplicate ids. Returns the pairs."""
    items = tuple(pairs)
    seen: set[str] = set()
    for pair in items:
        validate_contrastive(pair)
        if pair.id in seen:
            raise ValueError(f"duplicate id: {pair.id!r}")
        seen.add(pair.id)
    return items
