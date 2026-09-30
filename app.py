"""Streamlit web app.   Run with:  streamlit run app.py"""

import html
import re

import streamlit as st

from medsimp import config
from medsimp.facts import extract_quantities
from medsimp.fetch import pretty_section_name
from medsimp.pipeline import simplify_drug

st.set_page_config(page_title="Plain-Language Drug Labels", page_icon="💊", layout="wide")

st.markdown(
    """
    <style>
    .fact-kept    { background: rgba(34,160,90,.22);  border-radius: 3px; padding: 0 2px; }
    .fact-missing { background: rgba(220,50,50,.28);  border-radius: 3px; padding: 0 2px; font-weight: 600; }
    .fact-invented{ background: rgba(230,140,20,.30); border-radius: 3px; padding: 0 2px; font-weight: 600; }
    .label-box    { line-height: 1.65; font-size: 0.97rem; white-space: pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💊 Plain-Language Drug Labels")
st.caption("Rewrites drug labels at a ~5th-grade reading level, then checks that no dose, time, or warning was lost.")
st.error(
    "**Not medical advice.** This is a research prototype. The simplified text may contain errors "
    "and must be reviewed by a pharmacist before anyone relies on it.",
    icon="⚠️",
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Choose a medicine")
    examples = ["ibuprofen", "acetaminophen", "loratadine", "diphenhydramine", "omeprazole", "loperamide"]
    pick = st.selectbox("Common examples", ["(type your own)"] + examples, index=1)
    typed = st.text_input("…or type a generic / brand name", "")
    drug = typed.strip() or (pick if pick != "(type your own)" else "")

    use_nli = st.toggle("Meaning check (NLI)", value=True,
                        help="Uses a local AI model to check each sentence. Slower, but catches changed meanings.")
    all_sections = st.checkbox("All label sections", value=False,
                               help="Off = only directions, warnings and 'stop use' (fewer API calls).")
    run = st.button("Simplify label", type="primary", disabled=not drug, use_container_width=True)
    st.divider()
    st.caption(f"Model: `{config.OPENROUTER_MODEL}`  \nChange it in the `.env` file.")

    st.markdown(
        '**Legend**  \n<span class="fact-kept">kept fact</span> '
        '<span class="fact-missing">missing fact</span> '
        '<span class="fact-invented">invented number</span>',
        unsafe_allow_html=True,
    )

CORE_SECTIONS = ["dosage_and_administration", "warnings", "stop_use", "boxed_warning", "contraindications"]


def highlight(text: str, spans: list[tuple[str, str]]) -> str:
    """HTML-escape text and wrap each (phrase, css_class) occurrence in a <span>."""
    out = html.escape(text)
    for phrase, css in sorted(spans, key=lambda s: -len(s[0])):  # longest first
        pattern = re.compile(rf"(?<![\w>]){re.escape(html.escape(phrase))}(?!\w)", re.IGNORECASE)
        out = pattern.sub(lambda m: f'<span class="{css}">{m.group(0)}</span>', out, count=1)
    return f'<div class="label-box">{out}</div>'


if run:
    status = st.status(f"Working on **{drug}**…", expanded=True)
    try:
        result = simplify_drug(
            drug,
            use_nli=use_nli,
            sections=None if all_sections else CORE_SECTIONS,
            on_progress=status.write,
        )
        st.session_state["result"] = result
        status.update(label="Done", state="complete", expanded=False)
    except Exception as error:
        status.update(label="Something went wrong", state="error")
        st.error(f"**{type(error).__name__}:** {error}\n\nFinished sections are cached, so trying again is quick.")
        with st.expander("Technical details"):
            st.exception(error)

result = st.session_state.get("result")
if not result:
    st.info("Pick a medicine in the sidebar and press **Simplify label**.")
    st.stop()
if not result.sections:
    st.warning("This label has none of the sections we simplify.")
    st.stop()

# ---------------------------------------------------------------- summary
st.header(result.display_name)
sections = result.sections
kept = sum(len(s.fact_check.kept) for s in sections)
total = sum(len(s.fact_check.kept) + len(s.fact_check.missing) for s in sections)
invented = sum(len(s.fact_check.invented) for s in sections)
avg = lambda key, when: sum(getattr(s, when)[key] for s in sections) / len(sections)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reading grade", f"{avg('fk_grade', 'readability_after'):.1f}",
          f"{avg('fk_grade', 'readability_after') - avg('fk_grade', 'readability_before'):.1f} vs original",
          delta_color="inverse")
c2.metric("Hard words", f"{avg('hard_words_pct', 'readability_after'):.0f}%",
          f"{avg('hard_words_pct', 'readability_after') - avg('hard_words_pct', 'readability_before'):.0f} pts",
          delta_color="inverse")
c3.metric("Critical facts kept", f"{kept}/{total}", f"{(kept / total if total else 1):.0%} recall", delta_color="off")
c4.metric("Invented numbers", invented, delta_color="off")

# ---------------------------------------------------------------- per section
for s in sections:
    icon = "✅" if s.passed and not (s.nli and s.nli.flags) else ("⚠️" if s.passed else "❌")
    with st.expander(f"{icon} {pretty_section_name(s.name)}", expanded=True):
        left, right = st.columns(2)
        orig_spans = [(f.text, "fact-kept") for f in s.fact_check.kept] + \
                     [(f.text, "fact-missing") for f in s.fact_check.missing]
        kept_keys = {f.key for f in s.fact_check.kept}
        simp_spans = [(f.text, "fact-kept") for f in extract_quantities(s.simplified) if f.key in kept_keys] + \
                     [(f.text, "fact-invented") for f in s.fact_check.invented]
        with left:
            b = s.readability_before
            st.markdown(f"**Original** · grade {b['fk_grade']} · {b['hard_words_pct']}% hard words")
            st.markdown(highlight(s.original, orig_spans), unsafe_allow_html=True)
        with right:
            a = s.readability_after
            st.markdown(f"**Simplified** · grade {a['fk_grade']} · {a['hard_words_pct']}% hard words")
            st.markdown(highlight(s.simplified, simp_spans), unsafe_allow_html=True)

        st.markdown("---")
        info = [f"**Facts kept:** {len(s.fact_check.kept)}/{len(s.fact_check.kept) + len(s.fact_check.missing)}",
                f"**Retries:** {s.retries}"]
        if s.nli:
            info += [f"**Meaning coverage:** {s.nli.coverage:.0%}", f"**Faithfulness:** {s.nli.faithfulness:.0%}"]
        st.markdown(" · ".join(info))

        for problem in s.problems:
            st.error(problem)
        if s.nli:
            for flag in s.nli.flags:
                if flag.kind == "contradiction":
                    st.error(f"**Contradiction** ({flag.score:.0%}): {flag.sentence}")
                elif flag.direction == "lost":
                    st.warning(f"**Possibly lost** (original idea not clearly found): {flag.sentence}")
                else:
                    st.warning(f"**Possibly unsupported** (not clearly in the original): {flag.sentence}")

        if s.retries:
            with st.popover(f"See all {len(s.attempts)} attempts"):
                for i, attempt in enumerate(s.attempts, 1):
                    st.markdown(f"**Attempt {i}**" + (" (problems: " + "; ".join(attempt.problems) + ")"
                                                     if attempt.problems else " ✅"))
                    st.text(attempt.text)
