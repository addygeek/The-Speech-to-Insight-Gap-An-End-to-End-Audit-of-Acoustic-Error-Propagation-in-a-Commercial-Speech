"""Path helpers for cached audio and API responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config_loader import ExperimentConfig

ACOUSTIC_CONDITIONS = ("clean", "mild", "strong")


def ensure_dirs(cfg: ExperimentConfig) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "raw_responses" / "tts").mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "raw_responses" / "listen").mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "raw_responses" / "read").mkdir(parents=True, exist_ok=True)
    cfg.audio_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)


def audio_path(cfg: ExperimentConfig, utterance_id: str, condition: str) -> Path:
    return cfg.audio_dir / utterance_id / f"{condition}.wav"


def perturbation_meta_path(cfg: ExperimentConfig, utterance_id: str, condition: str) -> Path:
    return cfg.audio_dir / utterance_id / f"{condition}.meta.json"


def raw_response_path(cfg: ExperimentConfig, stage: str, key: str) -> Path:
    return cfg.results_dir / "raw_responses" / stage / f"{key}.json"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"data": data}


def cache_hit(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0
