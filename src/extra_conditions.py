"""Harsher acoustic conditions (E6): low-SNR pink noise, speech babble, reverberation, telephone codec.

Everything is generated from the clean TTS audio and per-file hash seeds, so the set is reproducible
with no external noise corpus. Babble is the sum of other clean utterances in the dataset.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy import signal

from src.budget import BudgetTracker
from src.cache import audio_path, cache_hit, raw_response_path, save_json
from src.config_loader import ExperimentConfig
from src.generate_dataset import iter_references
from src.logging_utils import setup_logging
from src.parse_responses import extract_transcript
from src.perturb_audio import _pink_noise, _read_wav_mono, _write_wav_mono, mix_at_snr
from src.read_jobs import ListenJob, ReadJob, run_listen_jobs, run_read_jobs

logger = setup_logging()

BABBLE_TALKERS = 6
PHONE_RATE = 8000

# name -> (kind, parameter)
EXTRA_CONDITIONS: dict[str, tuple[str, float]] = {
    "pinkm5": ("pink", -5.0),
    "pinkm10": ("pink", -10.0),
    "babble5": ("babble", 5.0),
    "babble0": ("babble", 0.0),
    "reverb": ("reverb", 1.0),
    "phone": ("phone", 0.0),
    "phonebabble": ("phone_babble", 5.0),
}


def _seed(uid: str, cond: str) -> int:
    return int(hashlib.sha256(f"42:{uid}:{cond}".encode()).hexdigest()[:8], 16)


def make_babble(length: int, others: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Sum of looped, randomly offset utterances. Needs at least one other signal."""
    out = np.zeros(length)
    for clip in others:
        offset = int(rng.integers(0, max(len(clip), 1)))
        rolled = np.roll(clip, offset)
        reps = math.ceil(length / len(rolled))
        out += np.tile(rolled, reps)[:length] / (np.std(rolled) + 1e-9)
    return out


def reverb(sig: np.ndarray, sr: int, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic exponentially decaying noise impulse response, then restore level."""
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir**2))
    wet = signal.fftconvolve(sig, ir)[: len(sig)]
    return wet * (np.sqrt(np.mean(sig**2) / (np.mean(wet**2) + 1e-12)))


def telephone(sig: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """300-3400 Hz band-pass, 8 kHz sampling, 8-bit mu-law companding."""
    sos = signal.butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    narrow = signal.sosfilt(sos, sig)
    down = signal.resample_poly(narrow, PHONE_RATE, sr)
    peak = np.max(np.abs(down)) + 1e-9
    x = down / peak
    mu = 255.0
    compressed = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round(compressed * 127) / 127
    expanded = np.sign(quantized) * np.expm1(np.abs(quantized) * np.log1p(mu)) / mu
    return expanded * peak, PHONE_RATE


def apply_condition(name: str, sig: np.ndarray, sr: int, rng: np.random.Generator, others: list[np.ndarray]) -> tuple[np.ndarray, int]:
    kind, param = EXTRA_CONDITIONS[name]
    if kind == "pink":
        return mix_at_snr(sig, _pink_noise(len(sig), rng), param), sr
    if kind == "babble":
        return mix_at_snr(sig, make_babble(len(sig), others, rng), param), sr
    if kind == "reverb":
        return reverb(sig, sr, 0.8, rng), sr
    if kind == "phone":
        return telephone(sig, sr)
    if kind == "phone_babble":
        mixed = mix_at_snr(sig, make_babble(len(sig), others, rng), param)
        return telephone(mixed, sr)
    raise KeyError(name)


def _others(cfg: ExperimentConfig, uids: list[str], uid: str, cond: str) -> list[np.ndarray]:
    rng = np.random.default_rng(_seed(uid, cond + "_pick"))
    pool = [u for u in uids if u != uid]
    picked = rng.choice(len(pool), size=min(BABBLE_TALKERS, len(pool)), replace=False)
    return [_read_wav_mono(audio_path(cfg, pool[i], "clean"))[0] for i in picked]


def extra_audio_path(cfg: ExperimentConfig, uid: str, cond: str) -> Path:
    return cfg.root / "data" / "audio_extra" / uid / f"{cond}.wav"


def generate_extra_audio(cfg: ExperimentConfig, conditions: list[str] | None = None, split: str | None = "full") -> int:
    conditions = conditions or list(EXTRA_CONDITIONS)
    records = list(iter_references(cfg, split=split))
    uids = [r["utterance_id"] for r in records]
    made = 0
    for rec in records:
        uid = rec["utterance_id"]
        clean = audio_path(cfg, uid, "clean")
        if not cache_hit(clean):
            continue
        sig, sr = _read_wav_mono(clean)
        for cond in conditions:
            out = extra_audio_path(cfg, uid, cond)
            if cache_hit(out):
                continue
            kind, param = EXTRA_CONDITIONS[cond]
            others = _others(cfg, uids, uid, cond) if "babble" in kind else []
            audio, out_sr = apply_condition(cond, sig, sr, np.random.default_rng(_seed(uid, cond)), others)
            peak = float(np.max(np.abs(audio)))
            if peak > 32000:  # rescale instead of clipping so low SNR stays a noise effect, not distortion
                audio = audio * (32000 / peak)
            _write_wav_mono(out, audio, out_sr)
            save_json(out.with_suffix(".meta.json"), {"utterance_id": uid, "condition": cond, "kind": kind, "param": param, "seed": _seed(uid, cond)})
            made += 1
    logger.info("extra audio files written: %s", made)
    return made


def listen_extra_path(cfg: ExperimentConfig, uid: str, cond: str) -> Path:
    return raw_response_path(cfg, "listen_extra", f"{uid}_{cond}")


def read_extra_path(cfg: ExperimentConfig, uid: str, cond: str) -> Path:
    return raw_response_path(cfg, "read_extra", f"{uid}_{cond}")


def run_extra_conditions(cfg: ExperimentConfig, budget: BudgetTracker, conditions: list[str] | None = None, split: str | None = "full", items: int | None = None) -> None:
    """`items` keeps the first n references in file order (hash-random ids), since the listen endpoint is slow."""
    conditions = conditions or list(EXTRA_CONDITIONS)
    generate_extra_audio(cfg, conditions, split)
    records = list(iter_references(cfg, split=split))[:items]
    listen = [
        ListenJob(r["utterance_id"], c, extra_audio_path(cfg, r["utterance_id"], c), listen_extra_path(cfg, r["utterance_id"], c), {})
        for r in records
        for c in conditions
        if cache_hit(extra_audio_path(cfg, r["utterance_id"], c))
    ]
    run_listen_jobs(cfg, budget, listen)
    from src.cache import load_json  # local import keeps the module header small

    reads: list[ReadJob] = []
    for r in records:
        for c in conditions:
            path = listen_extra_path(cfg, r["utterance_id"], c)
            if not cache_hit(path):
                continue
            payload = load_json(path)
            text = extract_transcript(payload.get("response") or payload)
            if text:
                reads.append(ReadJob(r["utterance_id"], f"transcript_{c}", text, read_extra_path(cfg, r["utterance_id"], c), {"condition": c}))
    run_read_jobs(cfg, budget, reads)

