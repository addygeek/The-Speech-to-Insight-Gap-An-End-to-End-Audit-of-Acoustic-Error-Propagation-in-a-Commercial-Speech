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
            ref_row = ref_intel.loc[uid]
            read_key = f"transcript_{cond}"
            read_rows = intel[(intel["utterance_id"] == uid) & (intel["text_source"] == read_key)]
            listen_rows = intel[
                (intel["utterance_id"] == uid)
                & (intel["pipeline_path"] == "audio_native")
                & (intel["acoustic_condition"] == cond)
            ]
            if not read_rows.empty:
                rr = read_rows.iloc[0]
                row["sentiment_disagreement_ref_vs_transcript"] = sentiment_disagreement(
                    str(ref_row["sentiment_label"]), str(rr["sentiment_label"])
                )
                row["intent_disagreement_ref_vs_transcript"] = intent_disagreement(
                    str(ref_row["intent_normalized"]), str(rr["intent_normalized"]), syn
                )
                changed, jac = topic_disagreement(str(ref_row["topics"]), str(rr["topics"]))
                row["topic_set_changed_ref_vs_transcript"] = changed
                row["topic_jaccard_ref_vs_transcript"] = jac
            if not listen_rows.empty and not read_rows.empty:
                lr = listen_rows.iloc[0]
                rr = read_rows.iloc[0]
                row["sentiment_disagreement_audio_vs_text"] = sentiment_disagreement(
                    str(lr["sentiment_label"]), str(rr["sentiment_label"])
                )
                row["intent_disagreement_audio_vs_text"] = intent_disagreement(
                    str(lr["intent_normalized"]), str(rr["intent_normalized"]), syn
                )
                changed, jac = topic_disagreement(str(lr["topics"]), str(rr["topics"]))
                row["topic_set_changed_audio_vs_text"] = changed
                row["topic_jaccard_audio_vs_text"] = jac
        rows.append(row)

    df = pd.DataFrame(rows)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.results_dir / "metrics.csv", index=False)
    return df


def merge_summary(cfg: ExperimentConfig, section: dict[str, Any]) -> None:
    path = cfg.results_dir / "summary.json"
    summary: dict[str, Any] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            summary = json.load(f)
    summary["statistics"] = section
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_statistics(cfg: ExperimentConfig) -> dict[str, Any]:
    df = run_metrics(cfg)
    stats_cfg = cfg.section("statistics")
    n_boot = int(stats_cfg.get("bootstrap_samples", 2000))
    seed = cfg.random_seed
    summary: dict[str, Any] = {"n_rows": int(len(df)), "by_condition": {}, "associations": {}}

    if df.empty:
        merge_summary(cfg, summary)
        return summary

    for cond, group in df.groupby("acoustic_condition"):
        mean, lo, hi = _bootstrap_mean(group["wer"].to_numpy(), n_boot, seed)
        sent_col = "sentiment_disagreement_ref_vs_transcript"
        sent_rate = float(group[sent_col].mean()) if sent_col in group else float("nan")
        summary["by_condition"][str(cond)] = {
            "n": int(len(group)),
            "wer_mean": mean,
            "wer_ci95": [lo, hi],
            "sentiment_disagreement_rate": sent_rate,
        }

    sent_col = "sentiment_disagreement_ref_vs_transcript"
    if "wer" in df.columns and sent_col in df.columns:
        sub = df.dropna(subset=["wer", sent_col])
        if len(sub) >= 5 and sub[sent_col].astype(int).nunique() > 1 and sub["wer"].nunique() > 1:
            rho, pval = spearmanr(sub["wer"], sub[sent_col].astype(int))
            summary["associations"]["wer_vs_sentiment_disagreement"] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n": int(len(sub)),
            }

    exploratory_min = int(stats_cfg.get("exploratory_min_utterances", 30))
    summary["exploratory"] = len(df["utterance_id"].unique()) < exploratory_min
    merge_summary(cfg, summary)
    return summary
