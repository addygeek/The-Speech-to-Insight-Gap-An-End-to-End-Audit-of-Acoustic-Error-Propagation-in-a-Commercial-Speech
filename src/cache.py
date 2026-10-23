"""Path helpers for cached audio and API responses."""

from __future__ import annotations

import json
import os
import shutil
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
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"data": data}


def cache_hit(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def is_dry_run_payload(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".json":
        return False
    try:
        return bool(load_json(path).get("dry_run"))
    except (json.JSONDecodeError, OSError):
        return False


def reusable(path: Path, live: bool) -> bool:
    """A live run does not reuse a file written by a dry run."""
    if not cache_hit(path):
        return False
    if live and is_dry_run_payload(path):
        path.unlink()
        return False
    return True


def complete_response(path: Path, live: bool) -> bool:
    """True when a saved API payload is intact and safe to skip on resume."""
    if not reusable(path, live):
        return False
    try:
        payload = load_json(path)
    except (json.JSONDecodeError, OSError):
        return False
    return "response" in payload or bool(payload.get("dry_run"))


def _remove_children(directory: Path, keep_names: set[str]) -> None:
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.name in keep_names:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def clear_generated(cfg: ExperimentConfig) -> None:
    """Drop previous audio, tables, figures, and result files before a new run."""
    _remove_children(cfg.audio_dir, {".gitkeep"})
    _remove_children(cfg.results_dir, {".gitkeep"})
    for pattern in ("*.pdf", "*.png"):
        for path in cfg.figures_dir.glob(pattern):
            path.unlink()
    for path in cfg.tables_dir.glob("*.tex"):
        path.unlink()
    for path in (cfg.references_jsonl, cfg.manifest_csv):
        if path.is_file():
            path.unlink()
    ensure_dirs(cfg)
