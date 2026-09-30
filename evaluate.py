"""Run the pipeline on a set of drugs and save the results for the evaluation notebook.

    python evaluate.py                 # 12 drugs, every label section (~130 LLM requests)
    python evaluate.py --quick         # 8 drugs, key sections only (fits in one day's 50 free requests)
    python evaluate.py --drugs ibuprofen naproxen --sections warnings stop_use
    python evaluate.py --no-nli        # faster, skips the meaning check

Output:
    results/eval.csv            one row per (drug, section) with all the metrics
    results/eval_details.json   the original and simplified text plus every flag

Free OpenRouter models have daily request limits. Every LLM reply is cached in cache/llm/,
so if you hit the limit, just run the same command again later and it continues where it stopped.
"""

import argparse
import json
from dataclasses import asdict

import pandas as pd

from medsimp import config
from medsimp.llm import DailyLimitReached
from medsimp.pipeline import simplify_drug

DEFAULT_DRUGS = [
    "ibuprofen", "acetaminophen", "naproxen", "aspirin", "diphenhydramine", "loratadine",
    "cetirizine", "loperamide", "omeprazole", "famotidine", "dextromethorphan", "guaifenesin",
]

# --quick: the first 8 drugs, and only the sections that hold most doses and warnings.
# boxed_warning is where prescription labels (e.g. naproxen) put their most serious warnings.
QUICK_DRUGS = DEFAULT_DRUGS[:8]
QUICK_SECTIONS = ["dosage_and_administration", "warnings", "stop_use", "boxed_warning"]


def section_row(drug_result, s) -> dict:
    total_facts = len(s.fact_check.kept) + len(s.fact_check.missing)
    return {
        "drug": drug_result.drug,
        "model": drug_result.model,
        "section": s.name,
        "words_before": s.readability_before["words"],
        "words_after": s.readability_after["words"],
        "fk_before": s.readability_before["fk_grade"],
        "fk_after": s.readability_after["fk_grade"],
        "hard_words_before": s.readability_before["hard_words_pct"],
        "hard_words_after": s.readability_after["hard_words_pct"],
        "facts_total": total_facts,
        "facts_kept": len(s.fact_check.kept),
        "fact_recall": s.fact_check.recall,
        "invented_facts": len(s.fact_check.invented),
        "nli_coverage": s.nli.coverage if s.nli else None,
        "nli_faithfulness": s.nli.faithfulness if s.nli else None,
        "contradictions": len(s.nli.contradictions) if s.nli else None,
        "hard_contradictions": len(s.nli.hard_contradictions) if s.nli else None,
        "first_try_recall": s.attempts[0].fact_check.recall,
        "first_try_invented": len(s.attempts[0].fact_check.invented),
        "retries": s.retries,
        "passed": s.passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drugs", nargs="*", help="drug names (default: 12 common drugs)")
    parser.add_argument("--sections", nargs="*", help="only these openFDA sections (default: all)")
    parser.add_argument("--quick", action="store_true", help="8 drugs, key sections only")
    parser.add_argument("--no-nli", action="store_true")
    args = parser.parse_args()
    drugs = args.drugs or (QUICK_DRUGS if args.quick else DEFAULT_DRUGS)
    sections = args.sections or (QUICK_SECTIONS if args.quick else None)

    rows, details, done = [], [], []
    for drug in drugs:
        print(f"\n### {drug}")
        try:
            result = simplify_drug(drug, use_nli=not args.no_nli, sections=sections,
                                   on_progress=lambda m: print("  " + m))
        except DailyLimitReached as error:
            # No point trying the other drugs today. Replies so far are cached, so re-running continues here.
            print(f"  STOPPED: {error}")
            break
        except Exception as error:  # keep going so one bad drug doesn't stop the whole run
            print(f"  FAILED: {error}")
            continue
        done.append(drug)
        for s in result.sections:
            rows.append(section_row(result, s))
            details.append({
                "drug": drug,
                "section": s.name,
                "original": s.original,
                "simplified": s.simplified,
                "missing": [asdict(f) for f in s.fact_check.missing],
                "invented": [asdict(f) for f in s.fact_check.invented],
                "nli_flags": [asdict(f) for f in s.nli.flags] if s.nli else [],
                "attempts": [a.text for a in s.attempts],
            })
        save(rows, details)   # after every drug, so an interrupted run still leaves usable results

    missing = [d for d in drugs if d not in done]
    print(f"\nSaved {len(rows)} sections from {len(done)}/{len(drugs)} drugs to {config.RESULTS_DIR}")
    if missing:
        print(f"Not finished: {', '.join(missing)}. Run the same command again later to continue.")


def save(rows: list[dict], details: list[dict]) -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "eval.csv", index=False)
    (config.RESULTS_DIR / "eval_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
