"""Step 4b: check meaning with an NLI (Natural Language Inference) model.

An NLI model reads two texts, a "premise" and a "hypothesis", and says whether the premise
ENTAILS the hypothesis (it follows), CONTRADICTS it, or is NEUTRAL (unrelated / not enough info).

Small NLI models only work well on short premises (one or two sentences). Given a paragraph, they
answer "neutral" even when the exact sentence is in it. So for each sentence we first RETRIEVE the
few most similar sentences from the other text (by word overlap, which is instant), and then run NLI
against each of those. The sentence counts as supported if any of them entails it.

We check in both directions:
- Simplified -> original ("unsupported"): is each simplified sentence backed up by the original?
  If the original CONTRADICTS a simplified sentence, the rewrite says something wrong. This is
  the most reliable check, because simplified sentences are clean and complete, so it triggers a retry.
- Original -> simplified ("lost"): is each idea from the original still in the simplified text?
  openFDA's original text often has no punctuation, so it gets split into rough fragments and
  this direction is noisier. Its flags are shown for human review only.

The model (cross-encoder/nli-deberta-v3-small, ~140M parameters) runs locally on CPU.
It is downloaded automatically the first time (about 500 MB).
"""

import re
from dataclasses import dataclass, field

from . import config

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL)
        _model.eval()
    return _tokenizer, _model


def nli(pairs: list[tuple[str, str]], batch_size: int = 32) -> list[dict[str, float]]:
    """For each (premise, hypothesis) pair return {'entailment': p, 'neutral': p, 'contradiction': p}."""
    import torch

    tokenizer, model = _load_model()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    results = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        inputs = tokenizer(
            [p for p, _ in batch], [h for _, h in batch],
            padding=True, truncation=True, max_length=256, return_tensors="pt",
        )
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        for row in probs:
            results.append({label: float(p) for label, p in zip(labels, row)})
    return results


def split_units(text: str, max_words: int = 30) -> list[str]:
    """Split text into small idea-sized pieces (sentences, bullet lines, or ~30-word slices)."""
    pieces = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    units = []
    for piece in pieces:
        words = piece.strip(" -•*").split()
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.split()) >= 3:   # skip fragments like "Warnings"
                units.append(chunk)
    return units


@dataclass
class Flag:
    direction: str      # "lost" (original idea not found) or "unsupported" (simplified idea not in original)
    kind: str           # "missing" or "contradiction"
    sentence: str       # the sentence that was flagged
    score: float        # entailment prob (for missing) or contradiction prob (for contradiction)


@dataclass
class NLIResult:
    flags: list[Flag] = field(default_factory=list)
    original_units: int = 0
    original_supported: int = 0
    simplified_units: int = 0
    simplified_supported: int = 0

    @property
    def coverage(self) -> float:
        """Share of original ideas supported by the simplified text."""
        return self.original_supported / self.original_units if self.original_units else 1.0

    @property
    def faithfulness(self) -> float:
        """Share of simplified sentences supported by the original."""
        return self.simplified_supported / self.simplified_units if self.simplified_units else 1.0

    @property
    def contradictions(self) -> list[Flag]:
        return [f for f in self.flags if f.kind == "contradiction"]

    @property
    def hard_contradictions(self) -> list[Flag]:
        """Simplified sentences the original contradicts. These trigger a retry."""
        return [f for f in self.contradictions if f.direction == "unsupported"]

    @staticmethod
    def combine(results: list["NLIResult"]) -> "NLIResult":
        """Add up the results for several chunks of one section."""
        total = NLIResult()
        for r in results:
            total.flags += r.flags
            total.original_units += r.original_units
            total.original_supported += r.original_supported
            total.simplified_units += r.simplified_units
            total.simplified_supported += r.simplified_supported
        return total


_STOPWORDS = set("a an the and or of to in on for if you your is are be it this that with as at by not do does "
                 "may can use using any other have has had from".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def _most_similar(sentence: str, candidates: list[str], k: int) -> list[str]:
    """The k candidates sharing the most content words with `sentence` (plus neighbours joined,
    because one idea is sometimes split across two sentences)."""
    options = candidates + [f"{a} {b}" for a, b in zip(candidates, candidates[1:])]
    words = _content_words(sentence)
    ranked = sorted(options, key=lambda c: len(words & _content_words(c)) / (len(words) or 1), reverse=True)
    return ranked[:k]


def _check_direction(premises: list[str], hypotheses: list[str], direction: str) -> tuple[list[Flag], int]:
    """Is each hypothesis supported by at least one of the (most similar) premises?"""
    if not hypotheses or not premises:
        return [], len(hypotheses)

    k = config.NLI_CANDIDATES
    candidates = [_most_similar(h, premises, k) for h in hypotheses]
    pairs = [(p, h) for h, cands in zip(hypotheses, candidates) for p in cands]
    scores = nli(pairs)

    flags, supported, i = [], 0, 0
    for hypothesis, cands in zip(hypotheses, candidates):
        rows = scores[i:i + len(cands)]
        i += len(cands)
        best_entail = max(r["entailment"] for r in rows)
        best_contra = max(r["contradiction"] for r in rows)
        if best_entail >= config.ENTAIL_THRESHOLD:
            supported += 1
        elif best_contra >= config.CONTRADICT_THRESHOLD:
            flags.append(Flag(direction, "contradiction", hypothesis, best_contra))
        else:
            flags.append(Flag(direction, "missing", hypothesis, best_entail))
    return flags, supported


def check_meaning(original: str, simplified: str) -> NLIResult:
    original_units, simplified_units = split_units(original), split_units(simplified)
    lost_flags, original_supported = _check_direction(simplified_units, original_units, "lost")
    unsupported_flags, simplified_supported = _check_direction(original_units, simplified_units, "unsupported")
    return NLIResult(
        flags=unsupported_flags + lost_flags,
        original_units=len(original_units),
        original_supported=original_supported,
        simplified_units=len(simplified_units),
        simplified_supported=simplified_supported,
    )
