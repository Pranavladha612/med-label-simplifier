"""The whole pipeline: fetch -> simplify -> verify -> (fix and retry) -> report.

Retry policy: we regenerate when the checks find a HARD problem:
  - a critical fact (dose, time, age, warning) is missing, or
  - a number appears that wasn't in the original (invented), or
  - the NLI model finds a contradiction.
"Possibly lost" / "possibly unsupported" NLI flags are softer (the model is imperfect), so they
are shown for human review but don't trigger a retry on their own.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from . import config, fetch, llm
from .facts import FactCheck, compare_facts
from .readability import readability
from .simplify import fix, simplify, split_into_chunks
from .verify import NLIResult, check_meaning, start_warm_up


@dataclass
class Attempt:
    text: str
    fact_check: FactCheck
    nli: NLIResult | None
    problems: list[str]
    chunk: int = 0              # which chunk of the section this attempt rewrote


@dataclass
class SectionResult:
    name: str
    original: str
    simplified: str
    attempts: list[Attempt]     # every LLM attempt, for every chunk, in order
    fact_check: FactCheck       # checks on the final, whole-section text
    nli: NLIResult | None
    problems: list[str]         # hard problems remaining in the final text
    readability_before: dict
    readability_after: dict

    @property
    def retries(self) -> int:
        return len(self.attempts) - len(split_into_chunks(self.original))

    @property
    def first_try_text(self) -> str:
        """The whole first draft: each chunk's first attempt, before any fix-and-retry."""
        firsts = {}
        for attempt in self.attempts:
            firsts.setdefault(attempt.chunk, attempt.text)
        return "\n".join(firsts[i] for i in sorted(firsts))

    @property
    def passed(self) -> bool:
        """True when the final version has no hard problems left."""
        return not self.problems

    @property
    def needs_review(self) -> bool:
        return not self.passed or bool(self.nli and self.nli.flags)


@dataclass
class DrugResult:
    drug: str
    display_name: str
    model: str
    sections: list[SectionResult] = field(default_factory=list)


def find_problems(fact_check: FactCheck, nli_result: NLIResult | None) -> list[str]:
    """Turn the check results into plain instructions the LLM can act on."""
    problems = []
    for f in fact_check.missing:
        what = "quantity" if f.kind == "quantity" else "warning about"
        problems.append(f'Missing {what} "{f.text}". Include it exactly.')
    for f in fact_check.invented:
        problems.append(f'"{f.text}" is not in the original. Remove it or correct it to match the original.')
    if nli_result:
        for flag in nli_result.hard_contradictions:
            problems.append(f'This sentence contradicts the original and must be fixed: "{flag.sentence}"')
    return problems


def _verify(original: str, simplified: str, use_nli: bool) -> tuple[FactCheck, NLIResult | None, list[str]]:
    fact_check = compare_facts(original, simplified)
    nli_result = check_meaning(original, simplified) if use_nli else None
    return fact_check, nli_result, find_problems(fact_check, nli_result)


def simplify_section(name: str, original: str, use_nli: bool = True) -> SectionResult:
    """Simplify one label section, splitting long sections into chunks and retrying on problems."""
    section_label = fetch.pretty_section_name(name)
    final_pieces, final_nli, attempts = [], [], []

    for i, chunk in enumerate(split_into_chunks(original)):
        text = simplify(chunk, section_label)
        fact_check, nli_result, problems = _verify(chunk, text, use_nli)
        attempts.append(Attempt(text, fact_check, nli_result, problems, chunk=i))

        for _ in range(config.MAX_RETRIES):
            if not problems:
                break
            text = fix(chunk, text, problems)
            fact_check, nli_result, problems = _verify(chunk, text, use_nli)
            attempts.append(Attempt(text, fact_check, nli_result, problems, chunk=i))

        final_pieces.append(text)
        if nli_result:
            final_nli.append(nli_result)

    simplified = "\n".join(final_pieces)
    if len(final_pieces) > 1:
        # Report on the whole section: facts are re-checked on the joined text (cheap),
        # NLI results are added up from the chunks (re-running NLI on the whole would be slow).
        fact_check = compare_facts(original, simplified)
        nli_result = NLIResult.combine(final_nli) if use_nli else None
        problems = find_problems(fact_check, nli_result)

    return SectionResult(
        name=name,
        original=original,
        simplified=simplified,
        attempts=attempts,
        fact_check=fact_check,
        nli=nli_result,
        problems=problems,
        readability_before=readability(original),
        readability_after=readability(simplified),
    )


def simplify_drug(
    drug: str,
    use_nli: bool = True,
    sections: list[str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> DrugResult:
    if use_nli:
        start_warm_up()   # load the NLI model in the background while the label downloads and the LLM writes
    label = fetch.fetch_label(drug)
    available = fetch.get_sections(label)
    if sections:
        available = {k: v for k, v in available.items() if k in sections}

    result = DrugResult(drug=drug, display_name=fetch.drug_display_name(label), model=config.OPENROUTER_MODEL)
    llm.models_used.clear()
    names = [fetch.pretty_section_name(name) for name in available]
    if on_progress and names:
        on_progress(f"Simplifying {len(names)} sections in parallel: {', '.join(names)}...")

    # Sections are independent, so they run in parallel threads: while one waits for the LLM, another
    # can use the NLI model. Progress is reported from this (main) thread, which is what Streamlit needs.
    workers = max(1, min(config.PARALLEL_SECTIONS, len(available)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(simplify_section, name, text, use_nli): name for name, text in available.items()}
        done = {}
        try:
            for future in as_completed(futures):
                name = futures[future]
                done[name] = future.result()   # re-raises errors such as DailyLimitReached
                if on_progress:
                    on_progress(f"Finished {fetch.pretty_section_name(name)} ({len(done)}/{len(futures)})")
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    result.sections = [done[name] for name in available]   # keep the label's section order
    result.model = ", ".join(sorted(llm.models_used)) or config.OPENROUTER_MODEL
    return result
