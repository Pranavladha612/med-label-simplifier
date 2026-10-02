"""Turn results into HTML with highlighted facts. Used by the Streamlit app and the Colab notebook."""

import html
import re

from .facts import extract_quantities
from .fetch import pretty_section_name

CSS = """
<style>
.fact-kept    { background: rgba(34,160,90,.22);  border-radius: 3px; padding: 0 2px; }
.fact-missing { background: rgba(220,50,50,.28);  border-radius: 3px; padding: 0 2px; font-weight: 600; }
.fact-invented{ background: rgba(230,140,20,.30); border-radius: 3px; padding: 0 2px; font-weight: 600; }
.label-box    { line-height: 1.65; font-size: 0.97rem; white-space: pre-wrap; }
.ms-grid      { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.ms-section   { border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.ms-flag      { margin: 4px 0; padding: 4px 8px; border-radius: 4px; font-size: 0.9rem; }
.ms-warn      { background: rgba(230,140,20,.15); }
.ms-bad       { background: rgba(220,50,50,.18); }
.ms-metrics   { display: flex; gap: 32px; flex-wrap: wrap; margin: 8px 0 4px; }
.ms-metrics b { font-size: 1.6rem; display: block; }
@media (max-width: 700px) { .ms-grid { grid-template-columns: 1fr; } }
</style>
"""

LEGEND = ('<span class="fact-kept">kept fact</span> <span class="fact-missing">missing fact</span> '
          '<span class="fact-invented">invented number</span>')


def highlight(text: str, spans: list[tuple[str, str]]) -> str:
    """HTML-escape text and wrap each (phrase, css_class) occurrence in a <span>."""
    out = html.escape(text)
    for phrase, css in sorted(spans, key=lambda s: -len(s[0])):  # longest first
        pattern = re.compile(rf"(?<![\w>]){re.escape(html.escape(phrase))}(?!\w)", re.IGNORECASE)
        out = pattern.sub(lambda m: f'<span class="{css}">{m.group(0)}</span>', out, count=1)
    return f'<div class="label-box">{out}</div>'


def fact_spans(section) -> tuple[list, list]:
    """Which phrases to highlight in the original and in the simplified text."""
    fc = section.fact_check
    original = [(f.text, "fact-kept") for f in fc.kept] + [(f.text, "fact-missing") for f in fc.missing]
    kept_keys = {f.key for f in fc.kept}
    simplified = [(f.text, "fact-kept") for f in extract_quantities(section.simplified) if f.key in kept_keys] + \
                 [(f.text, "fact-invented") for f in fc.invented]
    return original, simplified


def flag_label(flag) -> tuple[str, str]:
    """(text, css class) describing an NLI flag for people."""
    if flag.hard:
        return f"Contradiction ({flag.score:.0%}): {flag.sentence}", "ms-bad"
    if flag.kind == "contradiction" and flag.direction == "lost":
        return f"Possibly changed meaning (original sentence vs. the rewrite): {flag.sentence}", "ms-warn"
    if flag.kind == "contradiction":
        return f"Possible contradiction (low confidence: compared with a run-on fragment): {flag.sentence}", "ms-warn"
    if flag.direction == "lost":
        return f"Possibly lost (original idea not clearly found): {flag.sentence}", "ms-warn"
    return f"Possibly unsupported (not clearly in the original): {flag.sentence}", "ms-warn"


def section_html(s) -> str:
    """HTML for one section: original and rewrite side by side, metrics and flags."""
    original_spans, simplified_spans = fact_spans(s)
    b, a = s.readability_before, s.readability_after
    total = len(s.fact_check.kept) + len(s.fact_check.missing)
    info = f"Facts kept: {len(s.fact_check.kept)}/{total} · Retries: {s.retries}"
    if s.nli:
        info += f" · Meaning coverage: {s.nli.coverage:.0%} · Faithfulness: {s.nli.faithfulness:.0%}"
    flags = "".join(f'<div class="ms-flag ms-bad">{html.escape(p)}</div>' for p in s.problems)
    if s.nli:
        for flag in s.nli.flags:
            text, css = flag_label(flag)
            flags += f'<div class="ms-flag {css}">{html.escape(text)}</div>'
    icon = "✅" if s.passed and not (s.nli and s.nli.flags) else ("⚠️" if s.passed else "❌")
    return f"""
<div class="ms-section">
  <h3>{icon} {pretty_section_name(s.name)}</h3>
  <div class="ms-grid">
    <div><b>Original</b> · grade {b['fk_grade']} · {b['hard_words_pct']}% hard words
         {highlight(s.original, original_spans)}</div>
    <div><b>Simplified</b> · grade {a['fk_grade']} · {a['hard_words_pct']}% hard words
         {highlight(s.simplified, simplified_spans)}</div>
  </div>
  <p>{info}</p>
  {flags}
</div>"""


def drug_html(result) -> str:
    """A full report for one drug: summary numbers plus every section."""
    sections = result.sections
    if not sections:
        return CSS + "<p>This label has none of the sections we simplify.</p>"
    kept = sum(len(s.fact_check.kept) for s in sections)
    total = sum(len(s.fact_check.kept) + len(s.fact_check.missing) for s in sections)
    invented = sum(len(s.fact_check.invented) for s in sections)
    avg = lambda key, when: sum(getattr(s, when)[key] for s in sections) / len(sections)
    metrics = [
        ("Reading grade", f"{avg('fk_grade', 'readability_after'):.1f}",
         f"was {avg('fk_grade', 'readability_before'):.1f}"),
        ("Hard words", f"{avg('hard_words_pct', 'readability_after'):.0f}%",
         f"was {avg('hard_words_pct', 'readability_before'):.0f}%"),
        ("Critical facts kept", f"{kept}/{total}", f"{(kept / total if total else 1):.0%} recall"),
        ("Invented numbers", str(invented), ""),
    ]
    metric_html = "".join(f"<div>{name}<b>{value}</b><small>{note}</small></div>" for name, value, note in metrics)
    return (CSS + f"<h2>{html.escape(result.display_name)}</h2>"
            f"<p><small>Model: {html.escape(result.model)}</small><br>{LEGEND}</p>"
            f'<div class="ms-metrics">{metric_html}</div>'
            + "".join(section_html(s) for s in sections))
