"""Deepgram text intelligence (/v1/read) on reference and transcripts."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.budget import BudgetTracker
from src.cache import ACOUSTIC_CONDITIONS, cache_hit, load_json, raw_response_path, save_json
from src.config_loader import ExperimentConfig, require_deepgram_key
from src.failures import failure_row, log_failure
from src.generate_dataset import iter_references
from src.logging_utils import setup_logging
from src.manifest import Manifest, Stage, Status
from src.parse_responses import extract_transcript

logger = setup_logging()

DEEPGRAM_READ_URL = "https://api.deepgram.com/v1/read"


def _read_params(cfg: ExperimentConfig) -> dict[str, Any]:
    read = cfg.section("deepgram").get("read", {})
    params: dict[str, Any] = {"language": str(read.get("language", "en"))}
    if read.get("sentiment"):
        params["sentiment"] = "true"
    if read.get("intents"):
        params["intents"] = "true"
    if read.get("topics"):
        params["topics"] = "true"
    return params


def _dry_run_response(text: str) -> dict[str, Any]:
    fixture_path = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "read_response.json"
    if fixture_path.is_file():
        return copy.deepcopy(load_json(fixture_path))
    return {
        "results": {
            "sentiments": {"segments": [{"sentiment": "neutral", "sentiment_score": 0.0}]},
            "intents": {"segments": [{"intents": [{"intent": "general"}]}]},
            "topics": {"segments": [{"topics": [{"topic": "customer service"}]}]},
        }
    }


def _transcript_for_condition(cfg: ExperimentConfig, uid: str, condition: str) -> str:
    listen_path = raw_response_path(cfg, "listen", f"{uid}_{condition}")
    if not cache_hit(listen_path):
        return ""
    payload = load_json(listen_path)
    body = payload.get("response") or payload
    return extract_transcript(body)


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _post_read(client: httpx.Client, text: str, params: dict[str, Any]) -> httpx.Response:
    return client.post(DEEPGRAM_READ_URL, params=params, json={"text": text})


def run_read(
    cfg: ExperimentConfig,
    manifest: Manifest,
    budget: BudgetTracker,
    split: str | None = "pilot",
    max_requests: int | None = None,
) -> None:
    params = _read_params(cfg)
    failures_path = cfg.results_dir / "failures.csv"
    n = 0

    for rec in iter_references(cfg, split=split):
        uid = rec["utterance_id"]
        ref = rec["reference_text"]
        sources: list[tuple[str, str, str]] = [("reference", "", ref)]
        for cond in ACOUSTIC_CONDITIONS:
            tr = _transcript_for_condition(cfg, uid, cond)
            if tr:
                sources.append((f"transcript_{cond}", cond, tr))

        for text_source, condition, text in sources:
            if max_requests is not None and n >= max_requests:
                manifest.save()
                budget.save(cfg.results_dir / "cost_report.json")
                return
            if manifest.should_skip(cfg.resume, uid, Stage.READ, text_source):
                continue
            cache_key = f"{uid}_{text_source}"
            out_json = raw_response_path(cfg, "read", cache_key)
            if cache_hit(out_json):
                manifest.record(uid, Stage.READ, Status.SUCCESS, condition=text_source, meta="cache_hit")
                continue

            budget.add("read_estimate", budget.estimate_read(1))
            budget.check()

            if cfg.dry_run:
                body = _dry_run_response(text)
                save_json(
                    out_json,
                    {
                        "dry_run": True,
                        "meta": {
                            "utterance_id": uid,
                            "condition": condition,
                            "text_source": text_source,
                            "text": text,
                        },
                        "request_id": f"dry-{cache_key}",
                        "response": body,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                manifest.record(uid, Stage.READ, Status.SUCCESS, condition=text_source, meta="dry_run")
                n += 1
                continue

            api_key = require_deepgram_key()
            try:
                with httpx.Client(
                    headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
                    timeout=120.0,
                ) as client:
                    resp = _post_read(client, text, params)
                if resp.status_code in (401, 402, 403):
                    log_failure(
                        failures_path,
                        failure_row("read", uid, text_source, str(resp.status_code), "auth/billing"),
                    )
                    manifest.record(uid, Stage.READ, Status.FAILED, condition=text_source, error=str(resp.status_code))
                    continue
                resp.raise_for_status()
                req_id = resp.headers.get("dg-request-id", "")
                body = resp.json()
                save_json(
                    out_json,
                    {
                        "meta": {
                            "utterance_id": uid,
                            "condition": condition,
                            "text_source": text_source,
                            "text": text,
                        },
                        "request_id": req_id,
                        "params": params,
                        "response": body,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                manifest.record(uid, Stage.READ, Status.SUCCESS, condition=text_source, request_id=req_id)
                n += 1
            except httpx.HTTPError as exc:
                log_failure(failures_path, failure_row("read", uid, text_source, error=str(exc)))
                manifest.record(uid, Stage.READ, Status.FAILED, condition=text_source, error=str(exc))
                logger.exception("Read failed %s %s", uid, text_source)

    manifest.save()
    budget.save(cfg.results_dir / "cost_report.json")
