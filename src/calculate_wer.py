"""Word and character error rate calculations."""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from src.config_loader import ExperimentConfig
from src.text_normalize import normalize_text


@dataclass
class WerResult:
    wer: float
    substitutions: float
    deletions: float
    insertions: float
    cer: float


def compute_wer(reference: str, hypothesis: str, cfg: ExperimentConfig) -> WerResult:
    policy = cfg.section("text_normalization")
    ref = normalize_text(reference, policy)
    hyp = normalize_text(hypothesis, policy)
    if not ref and not hyp:
        return WerResult(0.0, 0.0, 0.0, 0.0, 0.0)
    if not ref:
        return WerResult(1.0, 0.0, 0.0, 1.0, 1.0)
    measures = jiwer.process_words(ref, hyp)
    cer = jiwer.cer(ref, hyp)
    denom = max(len(ref.split()), 1)
    return WerResult(
        wer=float(measures.wer),
        substitutions=float(measures.substitutions) / denom,
        deletions=float(measures.deletions) / denom,
        insertions=float(measures.insertions) / denom,
        cer=float(cer),
    )
