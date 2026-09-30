"""Step 3: ask the LLM to rewrite label text in plain language.

The first prompt sets the rules. If verification finds problems, we send a "fix-it" prompt
that lists exactly what went wrong, and try again.
"""

import re

from . import config, llm

SIMPLIFY_PROMPT = """You rewrite medicine label text so that people with low reading skills can understand it.

Rules:
1. Write at about a grade {grade} reading level. Use short sentences with one idea each.
2. Use everyday words. Speak to the reader as "you".
3. When a medical word must stay (like "NSAID"), explain it in simple words the first time.
   Don't explain words a child already knows.
4. Keep EVERY number, dose, amount, time, and age. Write them as digits, with the same units (for example "4 to 6 hours", "12 years", "200 mg").
5. Keep EVERY warning and every "ask a doctor" or "stop using" instruction.
6. Do NOT add any facts, doses, or advice that are not in the original text.
7. Use a short bulleted list (lines starting with "- ") when the original is a list.
8. Reply with ONLY the rewritten text. No introduction and no notes.

Label section: {section}

Original text:
\"\"\"
{text}
\"\"\"
"""

FIX_PROMPT = """You rewrote a medicine label section in plain language, but a safety check found problems.

Original text:
\"\"\"
{original}
\"\"\"

Your rewrite:
\"\"\"
{previous}
\"\"\"

Problems to fix:
{problems}

Write a corrected version. Fix only these problems and keep the rest of your rewrite as it is.
Stay at a grade {grade} reading level: put the missing facts in simple words, and do NOT copy
difficult wording from the original. Keep every number and warning exactly, and add nothing new.
Reply with ONLY the corrected text."""


def simplify(text: str, section: str) -> str:
    prompt = SIMPLIFY_PROMPT.format(grade=config.TARGET_GRADE, section=section, text=text)
    return clean_reply(llm.ask(prompt))


def fix(original: str, previous: str, problems: list[str]) -> str:
    prompt = FIX_PROMPT.format(
        original=original,
        previous=previous,
        problems="\n".join(f"- {p}" for p in problems),
        grade=config.TARGET_GRADE,
    )
    return clean_reply(llm.ask(prompt))


def clean_reply(reply: str) -> str:
    """Remove things models sometimes add anyway, like code fences or 'Here is the rewrite:'."""
    reply = reply.strip().strip('"').strip()
    reply = re.sub(r"^```\w*\n?|\n?```$", "", reply).strip()
    reply = re.sub(r"^(here is|here's)[^\n]*:\s*\n", "", reply, flags=re.IGNORECASE).strip()
    return reply


def split_into_chunks(text: str, max_words: int = config.MAX_CHUNK_WORDS) -> list[str]:
    """Split a long section into pieces of at most `max_words`, preferring sentence boundaries."""
    if len(text.split()) <= max_words:
        return [text]
    sentences = re.split(r"(?<=[.!?:])\s+", text)
    chunks, current = [], []
    for sentence in sentences:
        words = sentence.split()
        # A single giant "sentence" (common in openFDA's flattened lists): hard-split it.
        while len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if len(current) + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks
