from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sacrebleu
from rouge_score import rouge_scorer


@dataclass(frozen=True)
class TextPair:
    prediction: str
    reference: str


def _as_pairs(predictions: Iterable[str], references: Iterable[str]) -> list[TextPair]:
    pairs = [TextPair(prediction=p or "", reference=r or "") for p, r in zip(predictions, references)]
    if not pairs:
        raise ValueError("No prediction/reference pairs were provided.")
    return pairs


def lexical_metrics(predictions: Iterable[str], references: Iterable[str]) -> dict[str, float]:
    pairs = _as_pairs(predictions, references)
    preds = [p.prediction for p in pairs]
    refs = [p.reference for p in pairs]

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    for pair in pairs:
        scores = scorer.score(pair.reference, pair.prediction)
        for key in rouge_totals:
            rouge_totals[key] += scores[key].fmeasure

    count = len(pairs)
    return {
        "sacrebleu": round(float(bleu.score), 4),
        "rouge_1": round(100 * rouge_totals["rouge1"] / count, 4),
        "rouge_2": round(100 * rouge_totals["rouge2"] / count, 4),
        "rouge_l": round(100 * rouge_totals["rougeL"] / count, 4),
        "rouge_lsum": round(100 * rouge_totals["rougeLsum"] / count, 4),
        "num_samples": count,
    }

