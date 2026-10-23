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
            if isinstance(topic, dict) and topic.get("topic"):
                topics.append(str(topic["topic"]))
    if not topics and isinstance(topics_block.get("summary"), list):
        for item in topics_block["summary"]:
            if isinstance(item, dict) and item.get("topic"):
                topics.append(str(item["topic"]))
    return topics


def parse_intelligence(payload: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any]:
    syn = cfg.section("intent_normalization").get("synonym_map", {})
    sentiment_label, sentiment_score = extract_sentiment(payload)
    intent = extract_intents(payload, syn)
    topics = extract_topics(payload)
    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "intent_normalized": intent,
        "topics": sorted(topic_set(topics)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_transcripts_csv(cfg: ExperimentConfig) -> Path:
    out = cfg.results_dir / "transcripts.csv"
    rows: list[dict[str, Any]] = []
    listen_dir = cfg.results_dir / "raw_responses" / "listen"
    if listen_dir.is_dir():
        for path in sorted(listen_dir.glob("*.json")):
            key = path.stem
            parts = key.rsplit("_", 1)
            if len(parts) != 2:
                continue
            uid, condition = parts
            payload = load_json(path)
            body = payload.get("response") or payload
            rows.append(
                {
                    "utterance_id": uid,
                    "acoustic_condition": condition,
                    "transcript": extract_transcript(body),
                    "source": "listen",
                    "request_id": payload.get("request_id", ""),
                }
            )
    _write_csv(out, rows, ["utterance_id", "acoustic_condition", "transcript", "source", "request_id"])
    return out


def build_intelligence_csv(cfg: ExperimentConfig) -> Path:
    out = cfg.results_dir / "intelligence.csv"
    rows: list[dict[str, Any]] = []
    for stage, source_kind in (("listen", "audio_native"), ("read", "text_native")):
        dir_path = cfg.results_dir / "raw_responses" / stage
        if not dir_path.is_dir():
            continue
        for path in sorted(dir_path.glob("*.json")):
            payload = load_json(path)
            body = payload.get("response") or payload
            meta = payload.get("meta") or {}
            intel = parse_intelligence(body, cfg)
            uid = meta.get("utterance_id", path.stem.split("_")[0])
            rows.append(
                {
                    "utterance_id": uid,
                    "acoustic_condition": meta.get("condition", ""),
                    "text_source": meta.get("text_source", source_kind),
                    "pipeline_path": source_kind,
                    "transcript": extract_transcript(body) if stage == "listen" else meta.get("text", ""),
                    "sentiment_label": intel["sentiment_label"],
                    "sentiment_score": intel["sentiment_score"],
                    "intent_normalized": intel["intent_normalized"],
                    "topics": "|".join(intel["topics"]),
                    "request_id": payload.get("request_id", ""),
                }
            )
    fields = [
        "utterance_id",
        "acoustic_condition",
        "text_source",
        "pipeline_path",
        "transcript",
        "sentiment_label",
        "sentiment_score",
        "intent_normalized",
        "topics",
        "request_id",
    ]
    _write_csv(out, rows, fields)
    return out


def run_parse(cfg: ExperimentConfig) -> None:
    build_transcripts_csv(cfg)
    build_intelligence_csv(cfg)
    logger.info("Wrote parsed CSVs under %s", cfg.results_dir)
