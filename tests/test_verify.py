"""Tests for the NLI meaning check. The first run downloads the NLI model (~500 MB)."""

from medsimp.verify import _most_similar, check_meaning, split_units

ORIGINAL = (
    "It is especially important not to use ibuprofen at 20 weeks or later in pregnancy unless definitely "
    "directed to do so by a doctor because it may cause problems in the unborn child or complications "
    "during delivery. Keep out of reach of children. In case of overdose, get medical help or contact a "
    "Poison Control Center right away."
)


def test_split_units_skips_tiny_headings_and_splits_long_runs():
    units = split_units("Warnings\n" + " ".join(["word"] * 70))
    assert "Warnings" not in units
    assert all(len(u.split()) <= 30 for u in units)


def test_retrieval_finds_the_matching_sentence():
    candidates = split_units(ORIGINAL)
    best = _most_similar("If you take too much, call Poison Control right away.", candidates, k=1)[0]
    assert "Poison Control" in best


def test_faithful_rewrite_is_supported():
    simplified = (
        "Do not use ibuprofen at 20 weeks or later in pregnancy unless a doctor tells you to. "
        "In case of overdose, get medical help or contact a Poison Control Center right away."
    )
    result = check_meaning(ORIGINAL, simplified)
    assert result.faithfulness == 1.0


def test_invented_warning_is_flagged():
    # A real hallucination the LLM produced during development: the original never mentions the liver here.
    simplified = "Do not use ibuprofen at 20 weeks or later in pregnancy. It may cause liver problems."
    result = check_meaning(ORIGINAL, simplified)
    flagged = [f.sentence for f in result.flags if f.direction == "unsupported"]
    assert "It may cause liver problems." in flagged


def test_contradiction_is_caught():
    simplified = "It is safe to use ibuprofen at 20 weeks or later in pregnancy."
    result = check_meaning(ORIGINAL, simplified)
    assert result.hard_contradictions
