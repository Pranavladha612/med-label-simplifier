"""Step 2: pull out the "critical facts" from a piece of label text.

We look for two kinds of facts:

1. Quantities: a number plus a unit, e.g. "200 mg", "1 or 2 tablets", "every 4 to 6 hours",
   "under 12 years". These are the facts where a mistake could hurt someone, so they must
   survive simplification exactly.
2. Warning concepts: important safety topics, e.g. "allergic reaction", "stomach bleeding",
   "alcohol", "pregnancy". The simplified text may use plainer words, so each concept has
   a list of acceptable plain-language matches.

This is deliberately rule-based (regular expressions) so every decision is easy to inspect.
A natural upgrade is to replace or combine it with a medical NER model (scispaCy / med7).
"""

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

WORD_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "half": "0.5", "½": "0.5",
}

# Spelled-out fractions used on labels: "one-half", "a half", "one and one-half", "two and one-half".
# Listed first so "one-half" isn't read as the range "one to half".
_FRACTION = r"(?:(?:\d+|one|two|three|four|five)\s+and\s+)?(?:one|a)[- ]half"
# A single number: 200, 0.5, 1/2, a fraction, or a word like "two".
_NUM = (rf"(?:{_FRACTION}|\d+(?:\.\d+)?(?:/\d+)?|½|"
        + "|".join(w for w in WORD_NUMBERS if w != "½") + r")")
# A number or a range: "4 to 6", "4-6", "1 or 2".
_RANGE = rf"(?P<low>{_NUM})(?:\s*(?:to|-|–|or)\s*(?P<high>{_NUM}))?"

# ---------------------------------------------------------------------------
# Units. Each canonical unit maps to the spellings we accept for it.
# ---------------------------------------------------------------------------

UNITS = {
    "mg": ["mg", "milligrams?"],
    "mcg": ["mcg", "micrograms?", "µg"],
    "g": ["g", "grams?"],
    "ml": ["ml", "mL", "milliliters?", "millilitres?"],
    "%": ["%", "percent"],
    "unit": ["units?"],
    "tablet": ["tablets?", "caplets?", "pills?"],
    "capsule": ["capsules?", "softgels?", "gelcaps?"],
    "dose": ["doses?"],
    "teaspoon": ["teaspoons?", "tsp"],
    "tablespoon": ["tablespoons?", "tbsp"],
    "drop": ["drops?"],
    "puff": ["puffs?", "inhalations?"],
    "spray": ["sprays?"],
    "lozenge": ["lozenges?"],
    "patch": ["patch(?:es)?"],
    "drink": ["drinks?"],       # "3 or more alcoholic drinks"
    "time": ["times"],          # "3 times a day"
    "minute": ["minutes?", "mins?"],
    "hour": ["hours?", "hrs?"],
    "day": ["days?"],
    "week": ["weeks?"],
    "month": ["months?"],
    "year": ["years?"],
}

_UNIT_LOOKUP = []  # (compiled pattern, canonical unit)
for canonical, spellings in UNITS.items():
    for spelling in spellings:
        _UNIT_LOOKUP.append((re.compile(rf"^{spelling}$", re.IGNORECASE), canonical))

_ALL_UNITS = "|".join(s for spellings in UNITS.values() for s in spellings)
QUANTITY_RE = re.compile(
    rf"(?<![\w.]){_RANGE}"
    r"(?:\s+or\s+more)?"          # "3 or more drinks"
    # one describing word: "3 alcoholic drinks", "2 regular tablets". It must not itself be a unit,
    # or "500 mg tablet" would be read as "500 tablets".
    rf"(?:\s+(?!(?:{_ALL_UNITS})(?![A-Za-z]))[a-z-]+)?"
    rf"\s*(?P<unit>{_ALL_UNITS})(?![A-Za-z])",
    re.IGNORECASE,
)
# Ages written without "years": "age 60 or older", "aged 65".
AGE_RE = re.compile(rf"\bage[sd]?\s+(?P<age>{_NUM})\b", re.IGNORECASE)
# "once a day" / "twice daily" have no digit, so handle them separately.
ONCE_TWICE_RE = re.compile(r"\b(once|twice)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Warning concepts: (name, pattern to find it in the ORIGINAL, pattern accepted in the SIMPLIFIED)
# ---------------------------------------------------------------------------

# Patterns use \b (word boundary) wherever a short stem could hide inside another word:
# "liver" is inside "delivery", "renal" inside "adrenaline", "fit" inside "benefit", "stop" inside "nonstop".
CONCEPTS = [
    ("allergic reaction", r"allerg", r"allerg"),
    ("bleeding", r"bleed", r"bleed|blood"),
    ("heart attack", r"heart attack", r"heart attack"),
    ("stroke", r"\bstroke", r"\bstroke"),
    ("heart disease", r"heart (?:disease|problem|failure)", r"\bheart\b"),
    ("high blood pressure", r"high blood pressure|hypertension", r"blood pressure"),
    ("liver", r"\bliver\b|hepat", r"\bliver\b"),
    ("kidney", r"kidney|\brenal\b", r"kidney"),
    ("alcohol", r"alcohol|\bdrinks\b", r"alcohol|drink"),
    ("pregnancy", r"pregnan", r"pregnan"),
    ("breast-feeding", r"breast[- ]?feed|nursing", r"breast|nursing"),
    ("overdose", r"overdose", r"overdose|too much"),
    # "sleepiness"/"sleepy" count, but not "sleep" alone: "trouble sleeping" means the opposite.
    ("drowsiness", r"drows|sleepy|sedat", r"drows|sleep(?:y|iness)|tired"),
    ("driving / machinery", r"driving|operating machinery|machinery", r"driv|machine"),
    ("asthma", r"asthma", r"asthma"),
    ("diabetes", r"diabet", r"diabet|blood sugar"),
    ("seizures", r"seizure|convulsion", r"seizure|\bfits?\b"),
    ("suicidal thoughts", r"suicid", r"suicid|hurt(?:ing)? yourself|\bkill"),
    ("surgery", r"surgery", r"surgery|operation"),
    ("MAOI interaction", r"\bMAOI|monoamine oxidase", r"MAOI|monoamine"),
    ("blood thinners", r"anticoagul|blood thinn|warfarin", r"blood thinn|warfarin|anticoagul"),
    ("stop use", r"stop (?:use|using|taking)", r"\bstop"),
    ("keep away from children", r"out of (?:the )?reach of children", r"away from children|reach of children|where children"),
]


@dataclass(frozen=True)
class Fact:
    """One critical fact found in label text."""
    kind: str   # "quantity" or "concept"
    key: str    # canonical form used for matching, e.g. "4-6 hour" or "allergic reaction"
    text: str   # the exact words found in the text, for display


def _normalise_number(raw: str) -> str:
    """"two" -> "2", "one-half" -> "0.5", "two and one-half" -> "2.5", "200" -> "200"."""
    raw = raw.lower()
    if raw.endswith("half") and raw != "half":
        whole = raw.split(" and ")[0] if " and " in raw else "0"
        value = float(WORD_NUMBERS.get(whole, whole)) + 0.5
        return f"{value:g}"
    return WORD_NUMBERS.get(raw, raw)


def _canonical_unit(raw: str) -> str:
    """Map a unit spelling to its standard name, e.g. "caplets" -> "tablet"."""
    for pattern, canonical in _UNIT_LOOKUP:
        if pattern.match(raw):
            return canonical
    return raw.lower()


def extract_quantities(text: str) -> list[Fact]:
    """Every number-plus-unit in the text (doses, times, ages, frequencies) as normalised facts."""
    facts = []
    for m in QUANTITY_RE.finditer(text):
        low = _normalise_number(m.group("low"))
        high = m.group("high")
        value = f"{low}-{_normalise_number(high)}" if high else low
        unit = _canonical_unit(m.group("unit"))
        facts.append(Fact("quantity", f"{value} {unit}", m.group(0).strip()))
    for m in AGE_RE.finditer(text):
        facts.append(Fact("quantity", f"{_normalise_number(m.group('age'))} year", m.group(0)))
    for m in ONCE_TWICE_RE.finditer(text):
        value = "1" if m.group(1).lower() == "once" else "2"
        facts.append(Fact("quantity", f"{value} time", m.group(0)))
    return facts


def extract_concepts(text: str) -> list[Fact]:
    """Every warning concept (allergy, bleeding, pregnancy...) mentioned in ORIGINAL label text."""
    facts = []
    for name, original_pattern, _ in CONCEPTS:
        m = re.search(original_pattern, text, re.IGNORECASE)
        if m:
            facts.append(Fact("concept", name, m.group(0)))
    return facts


def extract_facts(text: str) -> list[Fact]:
    """All critical facts in a piece of ORIGINAL label text (duplicates removed)."""
    seen, unique = set(), []
    for fact in extract_quantities(text) + extract_concepts(text):
        if (fact.kind, fact.key) not in seen:
            seen.add((fact.kind, fact.key))
            unique.append(fact)
    return unique


def concept_present(concept_name: str, simplified_text: str) -> bool:
    """Is this warning concept mentioned in the simplified text (plain words allowed)?"""
    for name, _, simplified_pattern in CONCEPTS:
        if name == concept_name:
            return re.search(simplified_pattern, simplified_text, re.IGNORECASE) is not None
    return False


@dataclass
class FactCheck:
    """Which facts from the original survived the rewrite, which were lost, and which numbers were invented."""
    kept: list[Fact]            # facts from the original that survived
    missing: list[Fact]         # facts from the original that were lost  <- dangerous
    invented: list[Fact]        # quantities in the simplified text that weren't in the original  <- dangerous

    @property
    def recall(self) -> float:
        """Share of the original's facts kept in the rewrite (1.0 if there were none)."""
        total = len(self.kept) + len(self.missing)
        return 1.0 if total == 0 else len(self.kept) / total


def compare_facts(original: str, simplified: str) -> FactCheck:
    """Check a rewrite against its original: kept, missing and invented facts."""
    original_facts = extract_facts(original)
    simplified_quantity_keys = {f.key for f in extract_quantities(simplified)}
    original_quantity_keys = {f.key for f in original_facts if f.kind == "quantity"}

    kept, missing = [], []
    for fact in original_facts:
        if fact.kind == "quantity":
            present = fact.key in simplified_quantity_keys
        else:
            present = concept_present(fact.key, simplified)
        (kept if present else missing).append(fact)

    invented, seen = [], set()
    for fact in extract_quantities(simplified):
        if fact.key not in original_quantity_keys and fact.key not in seen:
            seen.add(fact.key)
            invented.append(fact)

    return FactCheck(kept=kept, missing=missing, invented=invented)
