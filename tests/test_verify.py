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
    assert all(len(u.split()) <= 45 for u in units)


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


def test_colon_does_not_split_off_a_lowercase_fragment():
    # Real false alarm on loperamide: "do not use." alone was compared with "Do not use any other dosing device."
    units = split_units("- If you are under 2 years old (up to 33 lbs): do not use.\n- Stop use: Call a doctor.")
    assert "If you are under 2 years old (up to 33 lbs): do not use." in units
    assert "Call a doctor." in units


def test_correct_age_rule_is_not_a_contradiction():
    original = ("Do not use any other dosing device. children 2-5 years (34 to 47 lbs) ask a doctor "
                "children under 2 years (up to 33 lbs) do not use")
    simplified = "- If you are under 2 years old (up to 33 lbs): do not use."
    assert check_meaning(original, simplified).hard_contradictions == []


def test_only_rewrite_side_contradictions_on_clean_evidence_are_hard():
    from medsimp.verify import Flag
    assert Flag("unsupported", "contradiction", "x", 0.99, reliable=True).hard
    assert not Flag("unsupported", "contradiction", "x", 0.99, reliable=False).hard   # run-on evidence
    assert not Flag("lost", "contradiction", "x", 0.99, reliable=True).hard           # original-side
    assert not Flag("unsupported", "missing", "x", 0.1).hard


def test_slices_of_a_long_run_on_are_not_clean():
    # Real false contradiction on dextromethorphan: a 50+ word run-on starts with a capital and ends
    # with a full stop, but its slices are not sentences.
    run_on = ("Stop use and ask a doctor if nervousness, dizziness, or sleeplessness occur pain, cough, or nasal "
              "congestion gets worse or lasts more than 7 days fever gets worse or lasts more than 3 days redness "
              "or swelling is present new symptoms occur cough comes back or occurs with a rash or headache that lasts.")
    assert all(clean is False for _, clean in split_units_marked(run_on, split_all_colons=True))


def test_run_on_pieces_are_not_clean_sentences():
    marked = dict(split_units_marked(split_all_colons=True, text=
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
