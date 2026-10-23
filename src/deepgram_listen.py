"""Deepgram prerecorded listen (STT + optional audio intelligence)."""

from __future__ import annotations

import copy
import json
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.budget import BudgetTracker
from src.cache import ACOUSTIC_CONDITIONS, audio_path, cache_hit, load_json, raw_response_path, save_json
from src.config_loader import ExperimentConfig, require_deepgram_key
from src.failures import failure_row, log_failure
from src.generate_dataset import iter_references
from src.logging_utils import setup_logging
from src.manifest import Manifest, Stage, Status

logger = setup_logging()

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"


def _audio_minutes(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate()) / 60.0


def _query_params(cfg: ExperimentConfig, keyterms: list[str] | None = None) -> list[tuple[str, str]]:
    items = [(key, str(value)) for key, value in _listen_params(cfg).items()]
    for term in keyterms or []:
        cleaned = term.strip()
        if cleaned:
            items.append(("keyterm", cleaned))
    return items


def _listen_params(cfg: ExperimentConfig) -> dict[str, Any]:
    listen = cfg.section("deepgram").get("listen", {})
    params: dict[str, Any] = {
        "model": listen.get("model", "nova-3"),
        "language": listen.get("language", "en"),
        "smart_format": str(listen.get("smart_format", True)).lower(),
    }
    if listen.get("sentiment"):
        params["sentiment"] = "true"
    if listen.get("intents"):
        params["intents"] = "true"
    if listen.get("topics"):
        params["topics"] = "true"
    return params


def _dry_run_response(reference_text: str) -> dict[str, Any]:
    fixture_path = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "listen_response.json"
    if fixture_path.is_file():
        payload = load_json(fixture_path)
    else:
        payload = {
            "results": {
                "channels": [
                    {"alternatives": [{"transcript": reference_text, "confidence": 0.95}]}
                ],
                "sentiments": {"segments": [{"sentiment": "neutral", "sentiment_score": 0.1}]},
                "intents": {"segments": [{"intents": [{"intent": "general"}]}]},
                "topics": {"segments": [{"topics": [{"topic": "customer service"}]}]},
            }
        }
    body = copy.deepcopy(payload)
    try:
        body["results"]["channels"][0]["alternatives"][0]["transcript"] = reference_text
    except (KeyError, IndexError, TypeError):
        pass
    return body


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _post_listen(
    client: httpx.Client,
    audio_bytes: bytes,
    params: dict[str, Any] | list[tuple[str, str]],
) -> httpx.Response:
    return client.post(DEEPGRAM_LISTEN_URL, params=params, content=audio_bytes)


def _keyterms_for(rec: dict[str, Any]) -> list[str]:
    spans = rec.get("critical_spans") or []
    types = rec.get("critical_span_types") or []
    keep = {"name", "place", "amount", "date", "number"}
    chosen = [span for span, kind in zip(spans, types) if kind in keep]
    return chosen[:8]


def _subset_ids(records: list[dict[str, Any]], size: int, predicate) -> list[dict[str, Any]]:
    picked = [rec for rec in records if predicate(rec)]
    picked.sort(key=lambda rec: rec["utterance_id"])
    return picked[:size]


def run_listen(
    cfg: ExperimentConfig,
    manifest: Manifest,
    budget: BudgetTracker,
    split: str | None = "pilot",
    max_requests: int | None = None,
) -> None:
    params = _query_params(cfg)
    failures_path = cfg.results_dir / "failures.csv"
    n = 0

    for rec in iter_references(cfg, split=split):
        uid = rec["utterance_id"]
        ref_text = rec["reference_text"]
        for cond in ACOUSTIC_CONDITIONS:
            if max_requests is not None and n >= max_requests:
                manifest.save()
                budget.save(cfg.results_dir / "cost_report.json")
                return
            if manifest.should_skip(cfg.resume, uid, Stage.LISTEN, cond):
                continue
            wav = audio_path(cfg, uid, cond)
