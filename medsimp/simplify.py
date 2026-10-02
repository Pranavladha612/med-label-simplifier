"""Step 3: ask the LLM to rewrite label text in plain language.

The prompts themselves live in prompts/prompts.yaml. The system prompt sets the rules; the
"simplify" prompt carries the label text. If verification finds problems, the "fix" prompt lists
exactly what went wrong, and the model tries again.
"""

import re
from functools import lru_cache

import yaml

from . import config, llm


@lru_cache(maxsize=1)
def load_prompts() -> dict:
    """Read prompts/prompts.yaml once."""
    with open(config.PROMPTS_FILE, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    for key in ("system", "simplify", "fix"):
        if key not in prompts:
            raise KeyError(f"{config.PROMPTS_FILE} is missing the '{key}' prompt")
    return prompts


def _system_prompt() -> str:
    """The system prompt from prompts.yaml with the target reading grade filled in."""
    return load_prompts()["system"].format(grade=config.TARGET_GRADE).strip()


def simplify(text: str, section: str) -> str:
    """First rewrite of one chunk of a label section, at the target reading level."""
    user = load_prompts()["simplify"].format(section=section, text=text).strip()
    return clean_reply(llm.ask(user, system=_system_prompt()))


def fix(original: str, previous: str, problems: list[str]) -> str:
    """Rewrite again, telling the model exactly which problems the checks found."""
    user = load_prompts()["fix"].format(
        original=original,
        previous=previous,
        problems="\n".join(f"- {p}" for p in problems),
    ).strip()
    return clean_reply(llm.ask(user, system=_system_prompt()))


def clean_reply(reply: str) -> str:
    """Remove things models sometimes add anyway, like code fences or 'Here is the rewrite:'."""
    reply = reply.strip().strip('"').strip()
    reply = re.sub(r"^```\w*\n?|\n?```$", "", reply).strip()
    reply = re.sub(r"^(here is|here's)[^\n]*:\s*\n", "", reply, flags=re.IGNORECASE).strip()
    reply = re.sub(r"^<rewrite>|</rewrite>$", "", reply).strip()   # in case the model echoes the tags
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
