"""Downstream output disagreement metrics."""

from __future__ import annotations

from src.text_normalize import jaccard, normalize_intent, topic_set


def sentiment_disagreement(ref_label: str, hyp_label: str) -> bool:
    return (ref_label or "").strip().lower() != (hyp_label or "").strip().lower()


def intent_disagreement(ref_intent: str, hyp_intent: str, synonym_map: dict[str, str] | None = None) -> bool:
    a = normalize_intent(ref_intent or "", synonym_map)
    b = normalize_intent(hyp_intent or "", synonym_map)
    return a != b


def topic_disagreement(ref_topics: str, hyp_topics: str) -> tuple[bool, float]:
    ref_set = topic_set([t for t in (ref_topics or "").split("|") if t])
    hyp_set = topic_set([t for t in (hyp_topics or "").split("|") if t])
    jac = jaccard(ref_set, hyp_set)
    return ref_set != hyp_set, jac
