"""Deepgram prerecorded listen (STT + optional audio intelligence)."""

from __future__ import annotations

import copy
import json
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.budget import BudgetTracker
from src.cache import (
    ACOUSTIC_CONDITIONS,
    audio_path,
    cache_hit,
    complete_response,
    load_json,
    raw_response_path,
    reusable,
    save_json,
)
from src.request_pool import POOL_WORKERS, run_pool
from src.config_loader import ExperimentConfig, require_deepgram_key
from src.deepgram_tts import post_with_backoff
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


class _ListenJob(NamedTuple):
    uid: str
    cond: str
    ref_text: str
    wav: Path
    out_json: Path
    cache_key: str
    minutes: float


def _checkpoint(manifest: Manifest, budget: BudgetTracker, cfg: ExperimentConfig, done: int, total: int, label: str) -> None:
    if done == total or done % 25 == 0:
        manifest.save()
        budget.save(cfg.results_dir / "cost_report.json")
        logger.info("%s %s/%s", label, done, total)


def run_listen(
    cfg: ExperimentConfig,
    manifest: Manifest,
    budget: BudgetTracker,
    split: str | None = "pilot",
    max_requests: int | None = None,
) -> None:
    params = _query_params(cfg)
    failures_path = cfg.results_dir / "failures.csv"
    live = not cfg.dry_run
    jobs: list[_ListenJob] = []

    for rec in iter_references(cfg, split=split):
        uid = rec["utterance_id"]
        ref_text = rec["reference_text"]
        for cond in ACOUSTIC_CONDITIONS:
            wav = audio_path(cfg, uid, cond)
            cache_key = f"{uid}_{cond}"
            out_json = raw_response_path(cfg, "listen", cache_key)
            if manifest.should_skip(cfg.resume, uid, Stage.LISTEN, cond) and complete_response(out_json, live):
                continue
            if complete_response(out_json, live):
                manifest.record(uid, Stage.LISTEN, Status.SUCCESS, condition=cond, meta="cache_hit")
                continue
            if not cache_hit(wav):
                manifest.record(uid, Stage.LISTEN, Status.FAILED, condition=cond, error="missing wav")
                continue
            jobs.append(
                _ListenJob(uid, cond, ref_text, wav, out_json, cache_key, _audio_minutes(wav))
            )

    if max_requests is not None:
        jobs = jobs[:max_requests]
    logger.info("listen pending %s workers %s", len(jobs), min(POOL_WORKERS, len(jobs) or 1))
    progress = {"done": 0}

    def worker(job: _ListenJob, client: httpx.Client | None, gate, book) -> None:
        if cfg.dry_run:
            save_json(
                job.out_json,
                {
                    "dry_run": True,
                    "meta": {"utterance_id": job.uid, "condition": job.cond, "text_source": "audio"},
                    "request_id": f"dry-{job.cache_key}",
                    "response": _dry_run_response(job.ref_text),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            with book:
                budget.add("stt_estimate", budget.estimate_stt_minutes(job.minutes))
                budget.check()
                manifest.record(job.uid, Stage.LISTEN, Status.SUCCESS, condition=job.cond, meta="dry_run")
                progress["done"] += 1
                _checkpoint(manifest, budget, cfg, progress["done"], len(jobs), "listen")
            return

        assert client is not None
        try:
            audio_bytes = job.wav.read_bytes()
            resp = post_with_backoff(
                lambda: _post_listen(client, audio_bytes, params),
                on_status=gate.note_status,
            )
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            with book:
                log_failure(failures_path, failure_row("listen", job.uid, job.cond, error=str(exc)))
                manifest.record(job.uid, Stage.LISTEN, Status.FAILED, condition=job.cond, error=str(exc))
                progress["done"] += 1
                _checkpoint(manifest, budget, cfg, progress["done"], len(jobs), "listen")
            logger.exception("Listen failed %s %s", job.uid, job.cond)
            return

        if resp.status_code in (401, 402, 403):
            with book:
                log_failure(
                    failures_path,
                    failure_row("listen", job.uid, job.cond, str(resp.status_code), "auth/billing"),
                )
                manifest.record(job.uid, Stage.LISTEN, Status.FAILED, condition=job.cond, error=str(resp.status_code))
                progress["done"] += 1
                _checkpoint(manifest, budget, cfg, progress["done"], len(jobs), "listen")
            gate.request_stop()
            return

        try:
            resp.raise_for_status()
            req_id = resp.headers.get("dg-request-id", "")
            body = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            with book:
                log_failure(failures_path, failure_row("listen", job.uid, job.cond, error=str(exc)))
                manifest.record(job.uid, Stage.LISTEN, Status.FAILED, condition=job.cond, error=str(exc))
                progress["done"] += 1
                _checkpoint(manifest, budget, cfg, progress["done"], len(jobs), "listen")
            logger.exception("Listen failed %s %s", job.uid, job.cond)
            return

        save_json(
            job.out_json,
            {
                "meta": {"utterance_id": job.uid, "condition": job.cond, "text_source": "audio"},
                "request_id": req_id,
                "params": params,
                "response": body,
                "audio_minutes": job.minutes,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        with book:
            budget.add("stt_estimate", budget.estimate_stt_minutes(job.minutes))
            budget.check()
            manifest.record(job.uid, Stage.LISTEN, Status.SUCCESS, condition=job.cond, request_id=req_id)
            progress["done"] += 1
            _checkpoint(manifest, budget, cfg, progress["done"], len(jobs), "listen")

    client_factory = None
    if live and jobs:
        api_key = require_deepgram_key()

        def client_factory() -> httpx.Client:
            return httpx.Client(
                headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
                timeout=180.0,
            )

    run_pool(jobs, worker, client_factory=client_factory, workers=POOL_WORKERS)

    if split == "full":
        records = list(iter_references(cfg, split=split))
        _run_optional_passes(cfg, manifest, budget, records, failures_path, max_requests, len(jobs))

    manifest.save()
    budget.save(cfg.results_dir / "cost_report.json")


def _run_optional_passes(
    cfg: ExperimentConfig,
    manifest: Manifest,
    budget: BudgetTracker,
    records: list[dict[str, Any]],
    failures_path: Path,
    max_requests: int | None,
    n_done: int,
) -> None:
    dg = cfg.section("deepgram")
    jobs: list[tuple[dict[str, Any], str, list[str], str]] = []
    repeat_cfg = dg.get("repeat", {})
    if repeat_cfg.get("enabled"):
        cond = str(repeat_cfg.get("condition", "clean"))
        for rec in _subset_ids(records, int(repeat_cfg.get("subset_size", 40)), lambda _r: True):
            jobs.append((rec, cond, [], f"{cond}_repeat"))
    keyterm_cfg = dg.get("keyterm", {})
    if keyterm_cfg.get("enabled"):
        cond = str(keyterm_cfg.get("condition", "strong"))
        for rec in _subset_ids(
            records,
            int(keyterm_cfg.get("subset_size", 150)),
            lambda r: bool(_keyterms_for(r)),
        ):
            jobs.append((rec, cond, _keyterms_for(rec), f"{cond}_keyterm"))

    n = n_done
    for rec, cond, keyterms, label in jobs:
        if max_requests is not None and n >= max_requests:
            return
        uid = rec["utterance_id"]
        wav = audio_path(cfg, uid, cond)
        cache_key = f"{uid}_{label}"
        out_json = raw_response_path(cfg, "listen", cache_key)
        if manifest.should_skip(cfg.resume, uid, Stage.LISTEN, label) and reusable(out_json, not cfg.dry_run):
            continue
        if reusable(out_json, not cfg.dry_run) or not cache_hit(wav):
            continue
        minutes = _audio_minutes(wav)
        budget.add("stt_estimate", budget.estimate_stt_minutes(minutes, keyterm=bool(keyterms)))
        budget.check()
        params = _query_params(cfg, keyterms)
        if cfg.dry_run:
            save_json(
                out_json,
                {
                    "dry_run": True,
                    "meta": {"utterance_id": uid, "condition": cond, "text_source": label},
                    "request_id": f"dry-{cache_key}",
                    "response": _dry_run_response(rec["reference_text"]),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            manifest.record(uid, Stage.LISTEN, Status.SUCCESS, condition=label, meta="dry_run")
            n += 1
            continue
        api_key = require_deepgram_key()
        try:
            with httpx.Client(
                headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
                timeout=180.0,
            ) as client:
                resp = post_with_backoff(lambda: _post_listen(client, wav.read_bytes(), params))
            if resp.status_code in (401, 402, 403):
                log_failure(failures_path, failure_row("listen", uid, label, str(resp.status_code), "auth/billing"))
                manifest.record(uid, Stage.LISTEN, Status.FAILED, condition=label, error=str(resp.status_code))
                continue
            resp.raise_for_status()
            req_id = resp.headers.get("dg-request-id", "")
            save_json(
                out_json,
                {
                    "meta": {"utterance_id": uid, "condition": cond, "text_source": label, "keyterms": keyterms},
                    "request_id": req_id,
                    "response": resp.json(),
                    "audio_minutes": minutes,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            manifest.record(uid, Stage.LISTEN, Status.SUCCESS, condition=label, request_id=req_id)
            n += 1
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log_failure(failures_path, failure_row("listen", uid, label, error=str(exc)))
            manifest.record(uid, Stage.LISTEN, Status.FAILED, condition=label, error=str(exc))
            logger.exception("Optional listen failed %s %s", uid, label)
