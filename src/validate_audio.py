"""Validate generated audio files."""

from __future__ import annotations

import hashlib
import wave

from src.cache import ACOUSTIC_CONDITIONS, audio_path, cache_hit
from src.config_loader import ExperimentConfig
from src.generate_dataset import iter_references
from src.logging_utils import setup_logging
from src.manifest import Manifest, Stage, Status

logger = setup_logging()


def _file_hash(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _wav_duration(path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def validate_utterance_audio(cfg: ExperimentConfig, utterance_id: str) -> list[str]:
    errors: list[str] = []
    hashes: dict[str, str] = {}
    for cond in ACOUSTIC_CONDITIONS:
        p = audio_path(cfg, utterance_id, cond)
        if not cache_hit(p):
            errors.append(f"{cond}: missing or empty")
            continue
        try:
            dur = _wav_duration(p)
            if dur < 0.2:
                errors.append(f"{cond}: duration too short ({dur:.2f}s)")
            hashes[cond] = _file_hash(p)
        except wave.Error as exc:
            errors.append(f"{cond}: invalid wav ({exc})")
    if len(hashes) == len(ACOUSTIC_CONDITIONS) and len(set(hashes.values())) == 1 and not cfg.dry_run:
        errors.append("all conditions identical (expected degradation differences)")
    return errors


def run_validate_audio(cfg: ExperimentConfig, manifest: Manifest, split: str | None = "pilot") -> None:
    for rec in iter_references(cfg, split=split):
        uid = rec["utterance_id"]
        if manifest.should_skip(cfg.resume, uid, Stage.VALIDATE_AUDIO):
            continue
        errs = validate_utterance_audio(cfg, uid)
        if errs:
            manifest.record(uid, Stage.VALIDATE_AUDIO, Status.FAILED, error="; ".join(errs))
            logger.warning("Audio validation %s: %s", uid, errs)
        else:
            manifest.record(uid, Stage.VALIDATE_AUDIO, Status.SUCCESS)
    manifest.save()
