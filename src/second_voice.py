"""Second and third TTS voices (E7): checks that the clean-speech robustness is not specific to one voice.

A seeded subset of the references is spoken by two other Aura-2 voices, clean and with 0 dB pink noise,
then passed through the same listen and read calls as the main study.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np

from src.budget import BudgetTracker
from src.cache import cache_hit, load_json, save_json
from src.config_loader import ExperimentConfig, require_deepgram_key
from src.deepgram_tts import _post_tts, post_with_backoff, repair_wav
from src.extra_conditions import extra_audio_path, listen_extra_path, read_extra_path, _seed
from src.generate_dataset import iter_references
from src.logging_utils import setup_logging
from src.parse_responses import extract_transcript
from src.perturb_audio import _pink_noise, _read_wav_mono, _write_wav_mono, mix_at_snr
from src.read_jobs import ListenJob, ReadJob, run_listen_jobs, run_read_jobs

logger = setup_logging()

VOICES = {"orion": "aura-2-orion-en", "luna": "aura-2-luna-en"}
SUBSET = 150
TTS_WORKERS = 6
HARSH_SNR_DB = 0.0
VOICE_CONDITIONS = tuple(f"{v}_{c}" for v in VOICES for c in ("clean", "harsh"))


def synthesize(cfg: ExperimentConfig, budget: BudgetTracker, records: list[dict[str, Any]]) -> None:
    tts = cfg.section("audio").get("tts", {})
    api_key = require_deepgram_key()
    todo = [(rec, voice, model) for rec in records for voice, model in VOICES.items() if not cache_hit(extra_audio_path(cfg, rec["utterance_id"], f"{voice}_clean"))]
    logger.info("tts pending %s", len(todo))

    def one(item: tuple[dict[str, Any], str, str]) -> None:
        rec, voice, model = item
        params = {"model": model, "encoding": tts.get("encoding", "linear16"), "container": "wav", "sample_rate": int(tts.get("sample_rate", 24000))}
        with httpx.Client(headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"}, timeout=120.0) as client:
            resp = post_with_backoff(lambda: _post_tts(client, rec["reference_text"], params))
        if resp.status_code in (401, 402, 403):
            raise RuntimeError(f"TTS auth/billing error: HTTP {resp.status_code}")
        resp.raise_for_status()
        out = extra_audio_path(cfg, rec["utterance_id"], f"{voice}_clean")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(repair_wav(resp.content))

    with ThreadPoolExecutor(TTS_WORKERS) as pool:
        list(pool.map(one, todo))
    for _ in todo:
        budget.add("tts_estimate", budget.estimate_tts(len(records[0]["reference_text"])))
    # 0 dB pink noise version of each voice, same generator and per-file seeding as the main study.
    for rec in records:
        for voice in VOICES:
            clean = extra_audio_path(cfg, rec["utterance_id"], f"{voice}_clean")
            noisy = extra_audio_path(cfg, rec["utterance_id"], f"{voice}_harsh")
            if cache_hit(clean) and not cache_hit(noisy):
                sig, sr = _read_wav_mono(clean)
                rng = np.random.default_rng(_seed(rec["utterance_id"], f"{voice}_harsh"))
                _write_wav_mono(noisy, mix_at_snr(sig, _pink_noise(len(sig), rng), HARSH_SNR_DB), sr)
                save_json(noisy.with_suffix(".meta.json"), {"utterance_id": rec["utterance_id"], "condition": f"{voice}_harsh", "snr_db": HARSH_SNR_DB})


def run_second_voice(cfg: ExperimentConfig, budget: BudgetTracker, split: str | None = "full") -> None:
    records = sorted(iter_references(cfg, split=split), key=lambda r: r["utterance_id"])[:SUBSET]
    synthesize(cfg, budget, records)
    listen = [
        ListenJob(r["utterance_id"], c, extra_audio_path(cfg, r["utterance_id"], c), listen_extra_path(cfg, r["utterance_id"], c), {})
        for r in records
        for c in VOICE_CONDITIONS
        if cache_hit(extra_audio_path(cfg, r["utterance_id"], c))
    ]
    run_listen_jobs(cfg, budget, listen)
    reads: list[ReadJob] = []
    for r in records:
        for c in VOICE_CONDITIONS:
            path = listen_extra_path(cfg, r["utterance_id"], c)
            if cache_hit(path):
                payload = load_json(path)
                text = extract_transcript(payload.get("response") or payload)
                if text:
                    reads.append(ReadJob(r["utterance_id"], f"transcript_{c}", text, read_extra_path(cfg, r["utterance_id"], c), {"condition": c}))
    run_read_jobs(cfg, budget, reads)
