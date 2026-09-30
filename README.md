# 💊 Plain-Language Drug Labels

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pranavladha612/med-label-simplifier/blob/main/colab_demo.ipynb)

**Simplify medication labels for low-literacy readers, then *prove* nothing safety-critical was lost.**

▶ **Try it in your browser:** click the badge above. The Colab demo runs the fact checker and the
hallucination check with no setup; add a free OpenRouter key to run the full pipeline on any medicine.

> ⚠️ **Not medical advice.** This is a research prototype. Simplified text may contain errors
> and must be reviewed by a pharmacist before anyone relies on it.

Rewriting text in simpler words is easy for an LLM. The hard part, and the point of this project,
is **verification**: making sure every dose, time limit, age limit and warning survives the
rewrite, and that the model didn't invent anything.

## How it works

```
openFDA label ──► split into sections ──► LLM simplifies (grade 5)
                                                │
                  ┌─────────────────────────────┘
                  ▼
           ┌─ Fact check ────────────────────────────────┐
           │  • extract doses / times / ages / warnings   │
           │  • every original fact still present?        │
           │  • any numbers that weren't in the original? │
           ├─ Meaning check (NLI, both directions) ───────┤
           │  • original idea supported by simplified?    │
           │  • simplified sentence supported by original?│
           │  • any contradictions?                       │
           └──────────────────────────────────────────────┘
                  │ problems found?
          yes ◄───┴───► no ──► report readability + metrics
           │
           ▼
   "fix-it" prompt listing the exact problems ──► retry (max 2)
```

| Step | File | What it does |
|---|---|---|
| 1. Fetch | `medsimp/fetch.py` | Downloads labels from the free [openFDA API](https://open.fda.gov/apis/drug/label/) |
| 2. Extract facts | `medsimp/facts.py` | Rule-based extraction of quantities ("1 or 2 tablets", "every 4 to 6 hours", "age 60") and warning concepts |
| 3. Simplify | `medsimp/simplify.py` + `prompts/prompts.yaml` | Prompts the LLM (via [OpenRouter](https://openrouter.ai)) |
| 4. Verify | `medsimp/facts.py`, `medsimp/verify.py` | Fact recall + invented numbers; "retrieve-then-verify" NLI with `cross-encoder/nli-deberta-v3-small` running locally |
| 5. Measure | `medsimp/readability.py` | Flesch-Kincaid grade, % hard words |
| Glue | `medsimp/pipeline.py` | Runs the loop and retries |

## Project structure

```
med-label-simplifier/
├── config.yaml              ← configuration file: model, fallbacks, temperature, retries, thresholds, sections
├── prompts/prompts.yaml     ← prompt file: system, simplify and fix prompts, with design notes
├── .env.example             ← template for the API key (the real .env is git-ignored)
├── requirements.txt
├── app.py                   ← Streamlit web app
├── cli.py                   ← command-line tool
├── evaluate.py              ← runs the pipeline on many drugs and saves metrics
├── colab_demo.ipynb         ← browser demo (Open in Colab badge above)
├── medsimp/                 ← the pipeline
│   ├── config.py            ← reads config.yaml and .env
│   ├── llm.py               ← OpenRouter client: fallbacks, retries, caching, error handling
│   ├── fetch.py  facts.py  simplify.py  verify.py  readability.py  pipeline.py  render.py
├── notebooks/evaluation.ipynb
└── tests/                   ← 22 tests (pytest)
```

## How the LLM is used

The LLM is called through the **OpenRouter API** (`medsimp/llm.py`), which uses the OpenAI API format and
gives access to many models with one key.

- **Two jobs:** it *simplifies* each label section, and when the checks find a problem it *repairs* its own
  rewrite from a precise list of what went wrong. That makes a generate → verify → repair loop.
- **Reliability:** automatic fallback to other models when the main one is busy, retries with backoff,
  a timeout, clear errors for daily-limit and empty replies, and hidden "thinking" text from reasoning models.
- **Efficiency:** `temperature: 0` for repeatable results; every reply is cached on disk (keyed on
  model + prompts), so re-runs and the evaluation never pay twice; long sections are split into ≤250-word chunks.

## Prompt design (`prompts/prompts.yaml`)

| Technique | Why |
|---|---|
| System / user split | The rules are the same for every call, so they live in the system prompt. Only the label text changes. |
| Role + audience | "Your readers may read at about a grade 5 level. Their safety depends on…" steers word choice better than "make it simpler". |
| Numbered, testable rules | Each rule matches something the verifier checks (numbers as digits, keep every warning, add nothing), so failures are measurable. |
| One-shot example | Shows the output style, including turning an unpunctuated openFDA list into bullets and keeping every number. It uses a different drug from the demos. |
| Delimiters | Label text goes inside `<label>` tags, so the label's own instructions ("ask a doctor") aren't mistaken for instructions to the model. |
| Output contract | "Reply with ONLY the rewritten text", plus a cleanup step in code, so replies can be verified directly. |
| Targeted repair prompt | Lists the exact problems and says to change only those. It forbids copying jargon from the original, which earlier versions did. |

**Effect of the prompt file (v3) vs. the earlier single-message prompt**, same free models, same labels
(Directions, Warnings and Stop-use sections):

| | Earlier prompt | `prompts.yaml` v3 |
|---|---|---|
| Ibuprofen: reading grade / hard words | 5.2 / 11% | **4.4 / 9%** |
| Loratadine: reading grade / hard words | 6.3 / 20% | **5.1 / 15%** |
| Critical facts kept | 34/34 and 16/16 | 34/34 and 16/16 |
| Invented numbers | 0 | 0 |

Simpler text with no loss of safety-critical facts. This is two drugs; `python evaluate.py` runs the
full 12-drug comparison.

## Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env
```

Then open `.env` and paste your OpenRouter key (from https://openrouter.ai/keys).
To change the model or any other setting, edit `config.yaml`. Any model from
https://openrouter.ai/models works; ids ending in `:free` cost nothing.

The first run downloads the NLI model (about 500 MB), once.

## Usage

**Web app** (for demos):
```bash
streamlit run app.py
```

**Command line** (one drug):
```bash
python cli.py ibuprofen
python cli.py loratadine --sections warnings stop_use
python cli.py acetaminophen --no-nli
```

**Evaluation** (many drugs, then open the notebook):
```bash
python evaluate.py
jupyter notebook notebooks/evaluation.ipynb
```

**Tests:**
```bash
pytest
```

> **Free-tier limits:** free OpenRouter models allow a limited number of requests per day.
> Every LLM reply is cached in `cache/llm/`, so if you hit the limit, re-run the same command
> later and it will continue where it stopped without repeating paid or limited calls.

## Metrics

| Metric | Meaning |
|---|---|
| **Critical-fact recall** (headline) | % of original doses/times/ages/warnings still present |
| **Invented numbers** | Quantities in the output that weren't in the original |
| **NLI coverage** | % of original ideas entailed by the simplified text |
| **NLI faithfulness** | % of simplified sentences entailed by the original |
| **Contradictions** | Sentences the NLI model says contradict the other text |
| **Flesch-Kincaid grade** | Reading level (see caveat below) |
| **Hard words %** | Words not on the Dale-Chall familiar-word list |
| **First try vs. final** | How much the verify-and-retry loop helps |

**Caveat:** openFDA removes punctuation from bulleted lists, so original sections often look like
one huge sentence, which inflates their Flesch-Kincaid grade. Hard-words % doesn't depend on
sentences and is the fairer before/after comparison.

## Design notes and early findings

These came up while building the project and are worth discussing in a write-up:

1. **Two checks catch different errors.** On ibuprofen, the LLM invented *"It may cause liver problems"*
   in the pregnancy warning. The fact checker missed it because "liver" appears elsewhere in the label
   ("liver cirrhosis"), but the NLI check flagged the sentence as unsupported.
2. **Small NLI models can't read paragraphs.** Given a 4-sentence premise that contains the exact
   hypothesis sentence, `nli-deberta-v3-small` answers "neutral". So each sentence is compared with the
   3 most word-similar sentences on the other side (retrieve, then verify). This is also ~10× faster
   than comparing every pair.
3. **An overly strict verifier makes output worse.** Early on, false alarms triggered retries, and
   the model "fixed" them by copying jargon back from the original. Now only reliable signals
   (missing or invented facts, contradictions in simplified sentences) trigger retries; the rest are
   shown for human review.
4. **openFDA text has no punctuation in lists**, which breaks sentence splitting and inflates
   readability scores. Every line is now treated as a sentence when measuring readability.

## Known limitations

- The fact extractor is regex-based. It is transparent and easy to test, but will miss unusual phrasings.
  Unit conversions ("24 hours" → "1 day") are counted as missing, which is deliberately strict.
- The NLI model is small and imperfect, so its "possibly lost/unsupported" flags are for human review
  and don't trigger retries on their own. Only contradictions do.
- There are no human-written reference simplifications, so SARI isn't computed yet (see notebook, *Next steps*).

## Ideas to extend it

- Swap or combine the regex extractor with a medical NER model (scispaCy, med7) and compare.
- Hand-label ~50 outputs to measure the precision and recall of the *verifier itself*.
- Compare several LLMs on first-try fact recall.
- Add Hindi or another language: simplify *and* translate, then verify across languages.
