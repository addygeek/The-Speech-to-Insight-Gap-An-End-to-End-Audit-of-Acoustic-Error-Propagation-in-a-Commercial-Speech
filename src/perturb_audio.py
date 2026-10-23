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
