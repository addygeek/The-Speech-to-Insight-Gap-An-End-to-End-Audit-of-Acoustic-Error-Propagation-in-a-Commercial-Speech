"""Parse Deepgram listen/read responses into normalized records."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.cache import load_json
from src.config_loader import ExperimentConfig
from src.logging_utils import setup_logging
from src.text_normalize import normalize_intent, topic_set

logger = setup_logging()


def _first_channel(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    if channels and isinstance(channels[0], dict):
        return channels[0]
    return {}


def extract_transcript(payload: dict[str, Any]) -> str:
    ch = _first_channel(payload)
    alts = ch.get("alternatives") or []
    if alts and isinstance(alts[0], dict):
        return str(alts[0].get("transcript", "")).strip()
    results = payload.get("results")
    if isinstance(results, dict):
        return str(results.get("transcript", "")).strip()
    return ""


def _to_float(val: Any) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def extract_sentiment(payload: dict[str, Any]) -> tuple[str, float | None]:
    results = payload.get("results") or {}
    sentiments = results.get("sentiments") or {}
    segments = sentiments.get("segments") or []
    if segments and isinstance(segments[0], dict):
        seg = segments[0]
        return str(seg.get("sentiment", "")).lower(), _to_float(seg.get("sentiment_score"))
    summary = sentiments.get("summary") if isinstance(sentiments, dict) else None
    if isinstance(summary, dict):
        return str(summary.get("sentiment", "")).lower(), _to_float(summary.get("sentiment_score"))
    return "", None


def extract_intents(payload: dict[str, Any], synonym_map: dict[str, str] | None = None) -> str:
    results = payload.get("results") or {}
    intents = results.get("intents") or {}
    segments = intents.get("segments") or []
    labels: list[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        for intent in seg.get("intents") or []:
            if isinstance(intent, dict) and intent.get("intent"):
                labels.append(str(intent["intent"]))
    if not labels and isinstance(intents.get("summary"), list):
        for item in intents["summary"]:
            if isinstance(item, dict) and item.get("intent"):
                labels.append(str(item["intent"]))
    if not labels:
        return ""
    return normalize_intent(labels[0], synonym_map)


def extract_topics(payload: dict[str, Any]) -> list[str]:
    results = payload.get("results") or {}
    topics_block = results.get("topics") or {}
    segments = topics_block.get("segments") or []
    topics: list[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        for topic in seg.get("topics") or []:
