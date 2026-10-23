"""Text normalization for WER and label comparison."""

from __future__ import annotations

import re
import string
from typing import Any


def normalize_text(text: str, policy: dict[str, Any] | None = None) -> str:
    policy = policy or {}
    out = text
    if policy.get("lowercase", True):
        out = out.lower()
    if policy.get("remove_punctuation", True):
        out = out.translate(str.maketrans("", "", string.punctuation))
    if policy.get("collapse_whitespace", True):
        out = re.sub(r"\s+", " ", out).strip()
    return out


def normalize_intent(label: str, synonym_map: dict[str, str] | None = None) -> str:
    base = normalize_text(label, {"lowercase": True, "remove_punctuation": True, "collapse_whitespace": True})
    if synonym_map:
        return synonym_map.get(base, base)
    return base


def topic_set(topics: list[str]) -> set[str]:
    return {normalize_text(t) for t in topics if t}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)
