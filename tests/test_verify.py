"""Tests for the NLI meaning check. The first run downloads the NLI model (~500 MB)."""

from medsimp.verify import _most_similar, check_meaning, split_units, split_units_marked

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
    candidates = split_units_marked(ORIGINAL)
    best_text, _ = _most_similar("If you take too much, call Poison Control right away.", candidates, k=1)[0]
    assert "Poison Control" in best_text


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


def test_only_rewrite_side_contradictions_on_clean_evidence_are_hard():
    from medsimp.verify import Flag
    assert Flag("unsupported", "contradiction", "x", 0.99, reliable=True).hard
    assert not Flag("unsupported", "contradiction", "x", 0.99, reliable=False).hard   # run-on evidence
    assert not Flag("lost", "contradiction", "x", 0.99, reliable=True).hard           # original-side
    assert not Flag("unsupported", "missing", "x", 0.1).hard


def test_run_on_pieces_are_not_clean_sentences():
    marked = dict(split_units_marked(
        "Keep out of reach of children. "
        "feel faint have bloody or black stools vomit blood you have symptoms of heart problems or stroke: "
        "Stop use and ask a doctor if you experience any of the following signs of stomach bleeding:"
    ))
    assert marked["Keep out of reach of children."] is True
    assert marked["feel faint have bloody or black stools vomit blood you have symptoms of heart problems or stroke:"] is False
    assert marked["Stop use and ask a doctor if you experience any of the following signs of stomach bleeding:"] is False


# The real ibuprofen "Stop use" list as openFDA delivers it: no punctuation between items.
RUN_ON = (
    "When using this product take with food or milk if stomach upset occurs Stop use and ask a doctor if you "
    "experience any of the following signs of stomach bleeding: feel faint have bloody or black stools vomit "
    "blood have stomach pain that does not get better you have symptoms of heart problems or stroke: chest pain "
    "slurred speech trouble breathing leg swelling weakness in one part or side of body pain gets worse or lasts "
    "more than 10 days fever gets worse or lasts more than 3 days"
)


def test_correct_sentences_are_not_hard_contradictions_against_run_on_text():
    # These correct rewrites were flagged as contradictions during development and triggered useless retries.
    simplified = (
        "Stop use and ask a doctor if you have chest pain.\n"
        "Stop use and ask a doctor if you have leg swelling.\n"
        "Stop use and ask a doctor if you have trouble breathing."
    )
    assert check_meaning(RUN_ON, simplified).hard_contradictions == []
