"""Step 5: measure how easy the text is to read.

Flesch-Kincaid grade uses sentence length AND word length. Careful: openFDA often removes the
punctuation from bulleted lists, so the original text looks like one enormous sentence and its
grade comes out unrealistically high. That would exaggerate how much we improved things.

So we also report word-level measures that don't depend on sentences:
- hard_words_pct: % of words that are "difficult" (not on the Dale-Chall list of familiar words)
- syllables_per_word: average word length in syllables
"""

import re

import textstat


def _end_lines_as_sentences(text: str) -> str:
    """Bullet lines like "- rash" have no full stop, so textstat would glue a whole list into one
    huge sentence. Treat every line as its own sentence, which is how a reader experiences it."""
    lines = [re.sub(r"^\s*[-•*]\s*", "", line).strip() for line in text.splitlines()]
    return " ".join(line if line.endswith((".", "!", "?", ":")) else line + "." for line in lines if line)


def readability(text: str) -> dict[str, float]:
    text = _end_lines_as_sentences(text)
    words = textstat.lexicon_count(text) or 1
    return {
        "fk_grade": round(textstat.flesch_kincaid_grade(text), 1),
        "hard_words_pct": round(100 * textstat.difficult_words(text) / words, 1),
        "syllables_per_word": round(textstat.syllable_count(text) / words, 2),
        "words": words,
    }
