"""Deepgram text intelligence (/v1/read) on reference and transcripts."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.budget import BudgetTracker
from src.cache import ACOUSTIC_CONDITIONS, cache_hit, complete_response, load_json, raw_response_path, save_json
from src.request_pool import POOL_WORKERS, run_pool
from src.config_loader import ExperimentConfig, require_deepgram_key
from src.deepgram_tts import post_with_backoff
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


class _ReadJob(NamedTuple):
    uid: str
    text_source: str
    condition: str
    text: str
    out_json: Path
    cache_key: str


def run_read(
    cfg: ExperimentConfig,
    manifest: Manifest,
    budget: BudgetTracker,
    split: str | None = "pilot",
    max_requests: int | None = None,
) -> None:
    params = _read_params(cfg)
    failures_path = cfg.results_dir / "failures.csv"
    live = not cfg.dry_run
    jobs: list[_ReadJob] = []

    for rec in iter_references(cfg, split=split):
        uid = rec["utterance_id"]
        ref = rec["reference_text"]
        sources: list[tuple[str, str, str]] = [("reference", "", ref)]
        for cond in ACOUSTIC_CONDITIONS:
            transcript = _transcript_for_condition(cfg, uid, cond)
            if transcript:
                sources.append((f"transcript_{cond}", cond, transcript))

        for text_source, condition, text in sources:
            cache_key = f"{uid}_{text_source}"
            out_json = raw_response_path(cfg, "read", cache_key)
            if manifest.should_skip(cfg.resume, uid, Stage.READ, text_source) and complete_response(out_json, live):
                continue
            if complete_response(out_json, live):
                manifest.record(uid, Stage.READ, Status.SUCCESS, condition=text_source, meta="cache_hit")
                continue
            jobs.append(_ReadJob(uid, text_source, condition, text, out_json, cache_key))

    if max_requests is not None:
        jobs = jobs[:max_requests]
    logger.info("read pending %s workers %s", len(jobs), min(POOL_WORKERS, len(jobs) or 1))
    progress = {"done": 0}

    def worker(job: _ReadJob, client: httpx.Client | None, gate, book) -> None:
        if cfg.dry_run:
            save_json(
                job.out_json,
                {
                    "dry_run": True,
                    "meta": {
                        "utterance_id": job.uid,
                        "condition": job.condition,
                        "text_source": job.text_source,
                        "text": job.text,
                    },
                    "request_id": f"dry-{job.cache_key}",
                    "response": _dry_run_response(job.text),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            with book:
                budget.add("read_estimate", budget.estimate_read(1))
                budget.check()
                manifest.record(job.uid, Stage.READ, Status.SUCCESS, condition=job.text_source, meta="dry_run")
                progress["done"] += 1
                if progress["done"] == len(jobs) or progress["done"] % 25 == 0:
                    manifest.save()
                    logger.info("read %s/%s", progress["done"], len(jobs))
            return

        assert client is not None
        try:
            resp = post_with_backoff(
                lambda: _post_read(client, job.text, params),
                on_status=gate.note_status,
            )
        except httpx.HTTPError as exc:
            with book:
                log_failure(failures_path, failure_row("read", job.uid, job.text_source, error=str(exc)))
                manifest.record(job.uid, Stage.READ, Status.FAILED, condition=job.text_source, error=str(exc))
                progress["done"] += 1
            logger.exception("Read failed %s %s", job.uid, job.text_source)
            return

        if resp.status_code in (401, 402, 403):
            with book:
                log_failure(
                    failures_path,
                    failure_row("read", job.uid, job.text_source, str(resp.status_code), "auth/billing"),
                )
                manifest.record(job.uid, Stage.READ, Status.FAILED, condition=job.text_source, error=str(resp.status_code))
            gate.request_stop()
            return

        try:
            resp.raise_for_status()
            req_id = resp.headers.get("dg-request-id", "")
            body = resp.json()
        except httpx.HTTPError as exc:
            with book:
                log_failure(failures_path, failure_row("read", job.uid, job.text_source, error=str(exc)))
                manifest.record(job.uid, Stage.READ, Status.FAILED, condition=job.text_source, error=str(exc))
                progress["done"] += 1
            logger.exception("Read failed %s %s", job.uid, job.text_source)
            return

        save_json(
            job.out_json,
            {
                "meta": {
                    "utterance_id": job.uid,
                    "condition": job.condition,
                    "text_source": job.text_source,
                    "text": job.text,
                },
                "request_id": req_id,
                "params": params,
                "response": body,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        with book:
            budget.add("read_estimate", budget.estimate_read(1))
            budget.check()
            manifest.record(job.uid, Stage.READ, Status.SUCCESS, condition=job.text_source, request_id=req_id)
            progress["done"] += 1
            if progress["done"] == len(jobs) or progress["done"] % 25 == 0:
                manifest.save()
                budget.save(cfg.results_dir / "cost_report.json")
                logger.info("read %s/%s", progress["done"], len(jobs))

    client_factory = None
    if live and jobs:
        api_key = require_deepgram_key()

        def client_factory() -> httpx.Client:
            return httpx.Client(
                headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
                timeout=120.0,
            )

    run_pool(jobs, worker, client_factory=client_factory, workers=POOL_WORKERS)
    manifest.save()
    budget.save(cfg.results_dir / "cost_report.json")
