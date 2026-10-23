"""Create mild and strong degraded audio variants."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from src.cache import audio_path, cache_hit, perturbation_meta_path, save_json
from src.config_loader import ExperimentConfig
from src.generate_dataset import iter_references
from src.logging_utils import setup_logging
from src.manifest import Manifest, Stage, Status

logger = setup_logging()


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        width = wf.getsampwidth()
    if width != 2:
        raise ValueError(f"Expected 16-bit WAV, got width {width}")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sample_rate


def _write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    fft = np.fft.rfft(white)
    freqs = np.arange(len(fft))
    scale = np.where(freqs == 0, 1.0, 1.0 / np.sqrt(freqs))
    pink = np.fft.irfft(fft * scale, n=n)
    return pink


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_power = np.mean(signal**2) + 1e-12
    noise = noise[: len(signal)]
    noise_power = np.mean(noise**2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scaled_noise = noise * math.sqrt(target_noise_power / noise_power)
    return signal + scaled_noise


def perturb_clean_to_condition(
    cfg: ExperimentConfig,
    clean_path: Path,
    out_path: Path,
    snr_db: float,
    seed: int,
    meta: dict[str, Any],
) -> None:
    signal, sr = _read_wav_mono(clean_path)
    rng = np.random.default_rng(seed)
    noise = _pink_noise(len(signal), rng)
    mixed = mix_at_snr(signal, noise, snr_db)
    _write_wav_mono(out_path, mixed, sr)
    uid = str(meta.get("utterance_id", out_path.parent.name))
    cond = str(meta.get("condition", out_path.stem))
    save_json(
        perturbation_meta_path(cfg, uid, cond),
        {**meta, "snr_db": snr_db, "seed": seed, "output": str(out_path)},
    )


def write_dry_run_tone(path: Path, sample_rate: int = 24000, seconds: float = 1.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = (0.2 * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(tone.tobytes())


def run_perturb(cfg: ExperimentConfig, manifest: Manifest, split: str | None = "pilot") -> None:
    pert_cfg = cfg.section("audio").get("perturbation", {})
    mild_snr = float(pert_cfg.get("mild_snr_db", 15))
    strong_snr = float(pert_cfg.get("strong_snr_db", 8))
    seed_base = int(pert_cfg.get("seed", 42))

    for rec in iter_references(cfg, split=split):
        uid = rec["utterance_id"]
        clean = audio_path(cfg, uid, "clean")
        if not cache_hit(clean) and cfg.dry_run:
            write_dry_run_tone(clean)
        if not cache_hit(clean):
            manifest.record(uid, Stage.PERTURB, Status.FAILED, condition="all", error="missing clean wav")
            continue

        for cond, snr in (("mild", mild_snr), ("strong", strong_snr)):
            if manifest.should_skip(cfg.resume, uid, Stage.PERTURB, cond):
                continue
            out = audio_path(cfg, uid, cond)
            try:
                if cfg.dry_run and not cache_hit(out):
                    write_dry_run_tone(out)
                    save_json(
                        perturbation_meta_path(cfg, uid, cond),
                        {
                            "condition": cond,
                            "snr_db": snr,
                            "dry_run": True,
                            "noise_source": "tone_placeholder",
                        },
                    )
                elif not cache_hit(out):
                    perturb_clean_to_condition(
                        cfg,
                        clean,
                        out,
                        snr_db=snr,
                        seed=seed_base + hash(uid + cond) % 10000,
                        meta={
                            "utterance_id": uid,
                            "condition": cond,
                            "noise_source": pert_cfg.get("noise_source", "generated_pink"),
                        },
                    )
                manifest.record(uid, Stage.PERTURB, Status.SUCCESS, condition=cond)
            except (OSError, ValueError) as exc:
                manifest.record(uid, Stage.PERTURB, Status.FAILED, condition=cond, error=str(exc))
                logger.exception("Perturb failed for %s %s", uid, cond)
    manifest.save()
