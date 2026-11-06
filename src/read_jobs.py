"""Generic cached /v1/read and /v1/listen job runners for the follow-up experiments."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import httpx

from src.budget import BudgetTracker
from src.cache import complete_response, save_json
from src.config_loader import ExperimentConfig, require_deepgram_key
from src.deepgram_listen import _audio_minutes, _post_listen, _query_params
from src.deepgram_read import _post_read, _read_params
from src.deepgram_tts import post_with_backoff
from src.failures import failure_row, log_failure
from src.logging_utils import setup_logging
from src.request_pool import POOL_WORKERS, run_pool

logger = setup_logging()


TRANSCRIPT_PARAMS = {"model", "language", "smart_format"}


class ReadJob(NamedTuple):
    uid: str
    label: str
    text: str
    out_json: Path
    meta: dict[str, Any]


class ListenJob(NamedTuple):
    uid: str
    label: str
    wav: Path
    out_json: Path
    meta: dict[str, Any]


def _pending(jobs: list, live: bool) -> list:
    return [job for job in jobs if not complete_response(job.out_json, live)]


def run_read_jobs(cfg: ExperimentConfig, budget: BudgetTracker, jobs: list[ReadJob], params: dict[str, Any] | None = None) -> int:
    """Run the pending read jobs live. Returns the number saved. Refuses to run in dry-run mode."""
    if cfg.dry_run:
        raise RuntimeError("Experiment runners need --live; dry-run would write placeholder responses.")
    todo = _pending(jobs, True)
    if not todo:
        return 0
    params = params or _read_params(cfg)
    failures = cfg.results_dir / "failures.csv"
    done = {"n": 0}
    api_key = require_deepgram_key()
    logger.info("read jobs pending %s of %s", len(todo), len(jobs))

    def worker(job: ReadJob, client: httpx.Client, gate, book) -> None:
        try:
            resp = post_with_backoff(lambda: _post_read(client, job.text, params), on_status=gate.note_status)
        except httpx.HTTPError as exc:
            with book:
                log_failure(failures, failure_row("read", job.uid, job.label, error=str(exc)))
            return
        if resp.status_code in (401, 402, 403):
            with book:
                log_failure(failures, failure_row("read", job.uid, job.label, str(resp.status_code), "auth/billing"))
            gate.request_stop()
            return
        try:
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            with book:
                log_failure(failures, failure_row("read", job.uid, job.label, error=str(exc)))
            return
        save_json(
            job.out_json,
            {
                "meta": {**job.meta, "utterance_id": job.uid, "text_source": job.label, "text": job.text},
                "request_id": resp.headers.get("dg-request-id", ""),
                "params": params,
                "response": body,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        with book:
            budget.add("read_estimate", budget.estimate_read(1))
            done["n"] += 1
            if done["n"] % 100 == 0:
                budget.save(cfg.results_dir / "cost_report.json")
                logger.info("read jobs %s/%s", done["n"], len(todo))

    def factory() -> httpx.Client:
        return httpx.Client(
            headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"}, timeout=120.0
        )

    run_pool(todo, worker, client_factory=factory, workers=POOL_WORKERS)
    budget.save(cfg.results_dir / "cost_report.json")
    return done["n"]


def run_listen_jobs(cfg: ExperimentConfig, budget: BudgetTracker, jobs: list[ListenJob]) -> int:
    if cfg.dry_run:
        raise RuntimeError("Experiment runners need --live; dry-run would write placeholder responses.")
    todo = _pending(jobs, True)
    if not todo:
        return 0
    params = [(k, v) for k, v in _query_params(cfg) if k in TRANSCRIPT_PARAMS]  # labels come from /v1/read, so skip audio intelligence
    failures = cfg.results_dir / "failures.csv"
    done = {"n": 0}
    api_key = require_deepgram_key()
    logger.info("listen jobs pending %s of %s", len(todo), len(jobs))

    def worker(job: ListenJob, client: httpx.Client, gate, book) -> None:
        minutes = _audio_minutes(job.wav)
        try:
            resp = post_with_backoff(lambda: _post_listen(client, job.wav.read_bytes(), params), on_status=gate.note_status)
        except httpx.HTTPError as exc:
            with book:
                log_failure(failures, failure_row("listen", job.uid, job.label, error=str(exc)))
            return
        if resp.status_code in (401, 402, 403):
            with book:
                log_failure(failures, failure_row("listen", job.uid, job.label, str(resp.status_code), "auth/billing"))
            gate.request_stop()
            return
        try:
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            with book:
                log_failure(failures, failure_row("listen", job.uid, job.label, error=str(exc)))
            return
        save_json(
            job.out_json,
            {
                "meta": {**job.meta, "utterance_id": job.uid, "condition": job.label, "text_source": "audio"},
                "request_id": resp.headers.get("dg-request-id", ""),
                "params": dict(params),
                "response": body,
                "audio_minutes": minutes,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        with book:
            budget.add("stt_estimate", budget.estimate_stt_minutes(minutes))
            done["n"] += 1
            if done["n"] % 100 == 0:
                budget.save(cfg.results_dir / "cost_report.json")
                logger.info("listen jobs %s/%s", done["n"], len(todo))

    def factory() -> httpx.Client:
        return httpx.Client(headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}, timeout=180.0)

    run_pool(todo, worker, client_factory=factory, workers=POOL_WORKERS)
    budget.save(cfg.results_dir / "cost_report.json")
    return done["n"]
