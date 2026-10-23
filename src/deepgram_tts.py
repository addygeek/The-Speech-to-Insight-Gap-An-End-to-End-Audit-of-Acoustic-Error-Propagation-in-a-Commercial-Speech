"""Deepgram Aura TTS client with caching."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.budget import BudgetTracker
from src.cache import audio_path, cache_hit, raw_response_path, save_json
from src.config_loader import ExperimentConfig, require_deepgram_key
from src.generate_dataset import iter_references
from src.logging_utils import setup_logging
from src.manifest import Manifest, Stage, Status
from src.perturb_audio import write_dry_run_tone

logger = setup_logging()

DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _post_tts(client: httpx.Client, text: str, params: dict[str, Any]) -> httpx.Response:
    return client.post(DEEPGRAM_SPEAK_URL, params=params, json={"text": text})


def run_tts(
    cfg: ExperimentConfig,
    manifest: Manifest,
    budget: BudgetTracker,
    split: str | None = "pilot",
    max_requests: int | None = None,
) -> None:
    tts_cfg = cfg.section("audio").get("tts", {})
    model = tts_cfg.get("model", "aura-2")
    voice = tts_cfg.get("voice", "aura-asteria-en")
    encoding = tts_cfg.get("encoding", "linear16")
    sample_rate = int(tts_cfg.get("sample_rate", 24000))
    n = 0

    for rec in iter_references(cfg, split=split):
        if max_requests is not None and n >= max_requests:
            break
        uid = rec["utterance_id"]
        text = rec["reference_text"]
        out_wav = audio_path(cfg, uid, "clean")
        cache_key = f"{uid}_clean"

        if manifest.should_skip(cfg.resume, uid, Stage.TTS):
            continue
        if cache_hit(out_wav):
            manifest.record(uid, Stage.TTS, Status.SUCCESS, meta="cache_hit")
            continue

        est = budget.estimate_tts(len(text))
        budget.add("tts_estimate", est)
        budget.check()

        if cfg.dry_run:
            write_dry_run_tone(out_wav, sample_rate=sample_rate)
            save_json(
                raw_response_path(cfg, "tts", cache_key),
                {"dry_run": True, "text": text, "timestamp_utc": datetime.now(timezone.utc).isoformat()},
            )
            manifest.record(uid, Stage.TTS, Status.SUCCESS, meta="dry_run")
            n += 1
            continue

        api_key = require_deepgram_key()
        params: dict[str, Any] = {"model": model, "encoding": encoding, "sample_rate": sample_rate}
        if voice:
            params["voice"] = voice

        try:
            with httpx.Client(
                headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
                timeout=120.0,
            ) as client:
                resp = _post_tts(client, text, params)
            if resp.status_code in (401, 402, 403):
                manifest.record(uid, Stage.TTS, Status.FAILED, error=f"HTTP {resp.status_code}")
                raise RuntimeError(f"TTS auth/billing error: HTTP {resp.status_code}")
            resp.raise_for_status()
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            out_wav.write_bytes(resp.content)
            req_id = resp.headers.get("dg-request-id", "")
            save_json(
                raw_response_path(cfg, "tts", cache_key),
                {
                    "utterance_id": uid,
                    "text": text,
                    "params": params,
                    "request_id": req_id,
                    "status_code": resp.status_code,
                    "bytes": len(resp.content),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            manifest.record(uid, Stage.TTS, Status.SUCCESS, request_id=req_id)
            n += 1
        except httpx.HTTPError as exc:
            manifest.record(uid, Stage.TTS, Status.FAILED, error=str(exc))
            logger.exception("TTS failed for %s", uid)

    manifest.save()
    budget.save(cfg.results_dir / "cost_report.json")
