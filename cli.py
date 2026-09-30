"""Simplify one drug's label from the command line.

Examples:
    python cli.py ibuprofen
    python cli.py "loratadine" --sections warnings stop_use
    python cli.py acetaminophen --no-nli        (faster: skip the NLI meaning check)
"""

import argparse
import sys

from medsimp.fetch import pretty_section_name
from medsimp.pipeline import simplify_drug
from medsimp.render import flag_label

DISCLAIMER = "NOT MEDICAL ADVICE. Research prototype. Simplified text must be reviewed by a pharmacist."


def main():
    parser = argparse.ArgumentParser(description="Simplify a drug label and verify nothing important was lost.")
    parser.add_argument("drug", help="generic or brand name, e.g. ibuprofen")
    parser.add_argument("--sections", nargs="*", help="only these openFDA sections, e.g. warnings stop_use")
    parser.add_argument("--no-nli", action="store_true", help="skip the NLI meaning check (faster)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # so Windows terminals don't choke on symbols
    print(f"\n{DISCLAIMER}\n")
    result = simplify_drug(args.drug, use_nli=not args.no_nli, sections=args.sections, on_progress=print)

    print(f"\n=== {result.display_name}  (model: {result.model}) ===")
    for s in result.sections:
        before, after = s.readability_before, s.readability_after
        status = "PASSED" if s.passed else "NEEDS REVIEW"
        print(f"\n--- {pretty_section_name(s.name)} [{status}] ---")
        print(f"\nORIGINAL (grade {before['fk_grade']}, {before['hard_words_pct']}% hard words):\n{s.original}")
        print(f"\nSIMPLIFIED (grade {after['fk_grade']}, {after['hard_words_pct']}% hard words):\n{s.simplified}")
        print(f"\nCritical facts kept: {len(s.fact_check.kept)}/{len(s.fact_check.kept) + len(s.fact_check.missing)}"
              f"   retries: {s.retries}")
        for f in s.fact_check.missing:
            print(f"  MISSING: {f.text}")
        for f in s.fact_check.invented:
            print(f"  INVENTED: {f.text}")
        if s.nli:
            print(f"NLI: coverage {s.nli.coverage:.0%}, faithfulness {s.nli.faithfulness:.0%}")
            for flag in s.nli.flags:
                label, _ = flag_label(flag)
                print(f"  {'CONTRADICTION: ' + flag.sentence if flag.hard else label}")
        for problem in s.problems:
            print(f"  PROBLEM LEFT FOR REVIEW: {problem}")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
