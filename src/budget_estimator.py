"""Preflight cost extrapolation from pilot manifest."""

from __future__ import annotations

import json
import wave
from pathlib import Path

from src.budget import BudgetTracker
from src.cache import ACOUSTIC_CONDITIONS, audio_path
from src.config_loader import ExperimentConfig
from src.generate_dataset import load_references
from src.logging_utils import setup_logging

logger = setup_logging()


def _audio_minutes(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate()) / 60.0


def run_preflight_cost(cfg: ExperimentConfig, target_utterances: int, split: str = "pilot") -> dict:
    budget = BudgetTracker(cfg)
    refs = [r for r in load_references(cfg.references_jsonl) if r.get("dataset_split") == split]
    if not refs:
        refs = load_references(cfg.references_jsonl)
    pilot_n = max(len(refs), 1)

    total_chars = sum(len(r["reference_text"]) for r in refs)
    total_minutes = 0.0
    for r in refs:
        uid = r["utterance_id"]
        for cond in ACOUSTIC_CONDITIONS:
            p = audio_path(cfg, uid, cond)
            if p.is_file():
                total_minutes += _audio_minutes(p)

    chars_per_utt = total_chars / pilot_n
    words = sum(len(r["reference_text"].split()) for r in refs)
    words_per_min = 150.0
    estimated_minutes_per_cond = (words / pilot_n) / words_per_min
    measured_minutes_per_cond = (
        total_minutes / (pilot_n * len(ACOUSTIC_CONDITIONS)) if total_minutes else 0.0
    )
    # Dry-run WAVs are 1.5s beeps. Use speaking-rate minutes until real Aura audio exists.
    if measured_minutes_per_cond >= 0.05:
        minutes_per_cond = measured_minutes_per_cond
        duration_source = "audio_files"
    else:
        minutes_per_cond = estimated_minutes_per_cond
        duration_source = "words_at_150_wpm"

    est_tts = budget.estimate_tts(int(chars_per_utt * target_utterances))
    est_stt = budget.estimate_stt_minutes(minutes_per_cond * target_utterances * len(ACOUSTIC_CONDITIONS))
    read_per_utt = 1 + len(ACOUSTIC_CONDITIONS)
    est_read = budget.estimate_read(read_per_utt * target_utterances)
    total = est_tts + est_stt + est_read

    report = {
        "pilot_utterances_observed": pilot_n,
        "target_utterances": target_utterances,
        "estimated_usd": {"tts": est_tts, "stt": est_stt, "read": est_read, "total": total},
        "budget_max_usd": budget.max_usd,
        "within_budget": total <= budget.max_usd,
        "duration_source": duration_source,
        "estimated_unique_minutes": round(minutes_per_cond * target_utterances, 1),
    }
    out = cfg.results_dir / "cost_report.json"
    existing: dict = {}
    if out.is_file():
        with out.open(encoding="utf-8") as f:
            existing = json.load(f)
    existing["preflight"] = report
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    logger.info(
        "Preflight estimate for %d utterances: $%.2f (cap $%.2f)",
        target_utterances,
        total,
        budget.max_usd,
    )
    return report
