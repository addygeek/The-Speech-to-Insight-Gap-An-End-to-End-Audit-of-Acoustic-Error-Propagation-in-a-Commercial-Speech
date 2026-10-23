"""Critical-span error detection."""

from __future__ import annotations

from src.config_loader import ExperimentConfig
from src.text_normalize import normalize_text


def span_present(span: str, text: str, cfg: ExperimentConfig) -> bool:
    policy = cfg.section("text_normalization")
    norm_span = normalize_text(span, policy)
    norm_text = normalize_text(text, policy)
    if not norm_span:
        return True
    return norm_span in norm_text


def critical_span_error_rate(
    reference: str,
    hypothesis: str,
    critical_spans: list[str],
    cfg: ExperimentConfig,
) -> tuple[float, int, int]:
    if not critical_spans:
        return 0.0, 0, 0
    errors = 0
    for span in critical_spans:
        if span_present(span, reference, cfg) != span_present(span, hypothesis, cfg):
            errors += 1
    total = len(critical_spans)
    return errors / total, errors, total
