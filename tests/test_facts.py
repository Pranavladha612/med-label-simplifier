"""Tests for the fact extractor. Run with:  pytest"""

from medsimp.facts import compare_facts, extract_facts, extract_quantities

IBUPROFEN_DIRECTIONS = (
    "adults and children 12 years and over: take 1 tablet every 4 to 6 hours while symptoms persist "
    "if pain or fever does not respond to 1 tablet, 2 tablets may be used "
    "do not exceed 6 tablets in 24 hours, unless directed by a doctor children under 12 years: ask a doctor"
)


def keys(facts):
    return {f.key for f in facts}


def test_finds_doses_times_and_ages():
    found = keys(extract_quantities(IBUPROFEN_DIRECTIONS))
    assert {"12 year", "1 tablet", "4-6 hour", "2 tablet", "6 tablet", "24 hour"} <= found


def test_ranges_and_word_numbers_normalise_the_same_way():
    assert keys(extract_quantities("take 1 or 2 tablets")) == {"1-2 tablet"}
    assert keys(extract_quantities("take one to two tablets")) == {"1-2 tablet"}
    assert keys(extract_quantities("every 4-6 hours")) == {"4-6 hour"}


def test_once_and_twice_daily():
    assert keys(extract_quantities("use twice daily")) == {"2 time"}
    assert keys(extract_quantities("take 2 times a day")) == {"2 time"}


def test_units_attached_to_numbers():
    assert keys(extract_quantities("contains 200mg ibuprofen")) == {"200 mg"}
    assert keys(extract_quantities("5 mL every 4 hrs")) == {"5 ml", "4 hour"}


def test_real_label_phrasings():
    # from the real ibuprofen warnings section
    text = "have 3 or more alcoholic drinks every day while using this product are age 60 or older"
    assert keys(extract_quantities(text)) == {"3 drink", "60 year"}
    # simplified versions must match the same keys
    assert keys(extract_quantities("You drink 3 or more alcoholic drinks a day. You are 60 years or older.")) \
        == {"3 drink", "60 year"}


def test_unit_is_not_taken_as_a_describing_word():
    # Real bug on naproxen: "500 mg tablet" was read as "500 tablets".
    assert keys(extract_quantities("take one 500 mg tablet")) == {"500 mg"}
    assert keys(extract_quantities("naproxen tablets 500 mg twice daily")) == {"500 mg", "2 time"}


def test_spelled_out_fractions():
    # Real false alarm on naproxen: the label says "one-half tablet", the rewrite "half tablet".
    assert keys(extract_quantities("250 mg (one-half tablet)")) == {"250 mg", "0.5 tablet"}
    assert keys(extract_quantities("take half a tablet")) == {"0.5 tablet"}
    assert keys(extract_quantities("take a half tablet")) == {"0.5 tablet"}
    assert keys(extract_quantities("one half tablet")) == {"0.5 tablet"}
    assert keys(extract_quantities("one and one-half tablets")) == {"1.5 tablet"}
    assert keys(extract_quantities("1375 mg (two and one-half tablets)")) == {"1375 mg", "2.5 tablet"}


def test_no_false_units_inside_words():
    # "gel" should not be read as "g", and "days" words without numbers are ignored
    assert extract_quantities("apply the gel for several days") == []


def test_warning_concepts():
    found = keys(extract_facts("Allergy alert: may cause a severe allergic reaction. Stomach bleeding warning"))
    assert {"allergic reaction", "bleeding"} <= found


def test_liver_is_not_found_inside_delivery():
    # Real bug: "delivery" matched the liver pattern, the checker asked for a liver warning,
    # and the LLM invented "Do not use if you have liver problems."
    assert "liver" not in keys(extract_facts("may cause complications during delivery"))
    assert "liver" in keys(extract_facts("ask a doctor if you have liver disease"))
    assert "liver" in keys(extract_facts("hepatic impairment"))


def test_concept_words_are_matched_as_whole_words():
    assert "kidney" not in keys(extract_facts("contains adrenaline"))
    assert "kidney" in keys(extract_facts("renal impairment"))


def test_simplified_side_ignores_look_alike_words():
    original = "ask a doctor if you have heart disease, seizures or liver problems, or suicidal thoughts. stop use if"
    simplified = "It may help with heartburn. The benefit is nonstop. Use a painkiller. Take it after delivery."
    missing = keys(compare_facts(original, simplified).missing)
    assert {"heart disease", "seizures", "liver", "suicidal thoughts", "stop use"} <= missing


def test_good_simplification_keeps_everything():
    simplified = (
        "For adults and children 12 years and over: take 1 tablet every 4 to 6 hours. "
        "If 1 tablet does not help, you can take 2 tablets. "
        "Do not take more than 6 tablets in 24 hours. Children under 12 years: ask a doctor."
    )
    check = compare_facts(IBUPROFEN_DIRECTIONS, simplified)
    assert check.missing == []
    assert check.invented == []
    assert check.recall == 1.0


def test_dropped_limit_and_invented_dose_are_caught():
    simplified = "Take 1 tablet every 4 to 6 hours. You can take up to 8 tablets a day."
    check = compare_facts(IBUPROFEN_DIRECTIONS, simplified)
    assert "24 hour" in keys(check.missing)
    assert "6 tablet" in keys(check.missing)
    assert "8 tablet" in keys(check.invented)
    assert check.recall < 1.0
