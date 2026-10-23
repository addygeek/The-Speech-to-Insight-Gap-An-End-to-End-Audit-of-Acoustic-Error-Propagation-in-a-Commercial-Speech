"""Aggregate metrics and bootstrap confidence intervals."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.calculate_critical_errors import critical_span_error_rate
from src.calculate_disagreement import intent_disagreement, sentiment_disagreement, topic_disagreement
from src.calculate_wer import compute_wer
from src.config_loader import ExperimentConfig
from src.generate_dataset import load_references
from src.logging_utils import setup_logging
from src.parse_responses import run_parse

logger = setup_logging()


def _bootstrap_mean(values: np.ndarray, n_samples: int, seed: int) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_samples)]
    arr = np.array(means)
    return float(np.mean(values)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def run_metrics(cfg: ExperimentConfig) -> pd.DataFrame:
    run_parse(cfg)
    refs = {r["utterance_id"]: r for r in load_references(cfg.references_jsonl)}
    intel_path = cfg.results_dir / "intelligence.csv"
    trans_path = cfg.results_dir / "transcripts.csv"
    if not intel_path.is_file() or not trans_path.is_file():
        logger.warning("Missing intelligence or transcripts CSV")
        return pd.DataFrame()

    intel = pd.read_csv(intel_path)
    transcripts = pd.read_csv(trans_path)
    syn = cfg.section("intent_normalization").get("synonym_map", {})
    ref_intel = intel[intel["text_source"] == "reference"].set_index("utterance_id")
    rows: list[dict[str, Any]] = []

    for _, tr in transcripts.iterrows():
        uid = tr["utterance_id"]
        cond = tr["acoustic_condition"]
        rec = refs.get(uid)
        if not rec:
            continue
        ref_text = rec["reference_text"]
        hyp = tr["transcript"]
        wer = compute_wer(ref_text, hyp, cfg)
        crit_rate, crit_err, crit_n = critical_span_error_rate(
            ref_text, hyp, rec.get("critical_spans", []), cfg
        )
        row: dict[str, Any] = {
            "utterance_id": uid,
            "acoustic_condition": cond,
            "scenario": rec.get("scenario", ""),
            "wer": wer.wer,
            "cer": wer.cer,
            "substitutions_rate": wer.substitutions,
            "deletions_rate": wer.deletions,
            "insertions_rate": wer.insertions,
            "critical_span_error_rate": crit_rate,
            "critical_span_errors": crit_err,
            "critical_span_total": crit_n,
        }
        if uid in ref_intel.index:
