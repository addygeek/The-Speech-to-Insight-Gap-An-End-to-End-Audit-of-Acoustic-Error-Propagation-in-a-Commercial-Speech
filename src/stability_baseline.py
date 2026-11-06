"""Stability baseline (E1), exact-match subset (E2), and formatting control (E3).

E1 sends identical text to /v1/read several times. Label disagreement between those repeats
is the noise floor: any cross-condition disagreement has to be read against it.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import numpy as np
import pandas as pd

from src.budget import BudgetTracker
from src.cache import ACOUSTIC_CONDITIONS, cache_hit, load_json, raw_response_path
from src.config_loader import ExperimentConfig
from src.generate_dataset import iter_references
from src.logging_utils import setup_logging
from src.parse_responses import extract_transcript, parse_intelligence
from src.read_jobs import ReadJob, run_read_jobs
from src.statistical_analysis import _bootstrap_mean
from src.text_normalize import jaccard, normalize_text

logger = setup_logging()

REPEAT_SOURCES = ("reference", "transcript_clean")
DEFAULT_REPEATS = 3


def _repeat_path(cfg: ExperimentConfig, uid: str, source: str, rep: int):
    return raw_response_path(cfg, "read_repeat", f"{uid}_{source}_r{rep}")


def _format_path(cfg: ExperimentConfig, uid: str, source: str):
    return raw_response_path(cfg, "read_format", f"{uid}_{source}")


def _source_text(cfg: ExperimentConfig, rec: dict[str, Any], source: str) -> str:
    if source == "reference":
        return rec["reference_text"]
    cond = source.removeprefix("transcript_")
    path = raw_response_path(cfg, "listen", f"{rec['utterance_id']}_{cond}")
    if not cache_hit(path):
        return ""
    payload = load_json(path)
    return extract_transcript(payload.get("response") or payload)


def build_repeat_jobs(cfg: ExperimentConfig, records: list[dict[str, Any]], repeats: int) -> list[ReadJob]:
    jobs: list[ReadJob] = []
    for rec in records:
        for source in REPEAT_SOURCES:
            text = _source_text(cfg, rec, source)
            if not text:
                continue
            for rep in range(1, repeats + 1):
                jobs.append(
                    ReadJob(rec["utterance_id"], source, text, _repeat_path(cfg, rec["utterance_id"], source, rep), {"repeat": rep})
                )
    return jobs


def build_format_jobs(cfg: ExperimentConfig, records: list[dict[str, Any]]) -> list[ReadJob]:
    """Reference and clean transcript with case and punctuation removed from both."""
    jobs: list[ReadJob] = []
    for rec in records:
        for source in REPEAT_SOURCES:
            text = normalize_text(_source_text(cfg, rec, source))
            if text:
                label = f"{source}_norm"
                jobs.append(ReadJob(rec["utterance_id"], label, text, _format_path(cfg, rec["utterance_id"], label), {"formatting": "normalized"}))
    return jobs


def run_stability(cfg: ExperimentConfig, budget: BudgetTracker, repeats: int = DEFAULT_REPEATS, split: str | None = "full") -> int:
    records = list(iter_references(cfg, split=split))
    jobs = build_repeat_jobs(cfg, records, repeats) + build_format_jobs(cfg, records)
    return run_read_jobs(cfg, budget, jobs)


def _labels(cfg: ExperimentConfig, path) -> dict[str, Any] | None:
    if not cache_hit(path):
        return None
    payload = load_json(path)
    body = payload.get("response")
    if not body:
        return None
    intel = parse_intelligence(body, cfg)
    return {"intent": intel["intent_normalized"], "topics": set(intel["topics"]), "sentiment": intel["sentiment_label"]}


def _pair_stats(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {
        "intent_disagree": float(a["intent"] != b["intent"]),
        "topic_disagree": float(a["topics"] != b["topics"]),
        "topic_jaccard": jaccard(a["topics"], b["topics"]),
        "sentiment_disagree": float(a["sentiment"] != b["sentiment"]),
    }


def _mean_ci(values: list[float], seed: int, n_boot: int) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    mean, lo, hi = _bootstrap_mean(np.asarray(values, dtype=float), n_boot, seed)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(values)}


def _summarize(rows: list[dict[str, float]], seed: int, n_boot: int) -> dict[str, dict[str, float]]:
    return {key: _mean_ci([r[key] for r in rows], seed, n_boot) for key in ("intent_disagree", "topic_disagree", "topic_jaccard", "sentiment_disagree")}


def _per_item_pairwise(copies: list[dict[str, Any]]) -> dict[str, float] | None:
    """Mean over all pairs of repeated calls for one item."""
    if len(copies) < 2:
        return None
    pairs = [_pair_stats(a, b) for a, b in itertools.combinations(copies, 2)]
    return {key: float(np.mean([p[key] for p in pairs])) for key in pairs[0]}


def analyze_stability(cfg: ExperimentConfig, repeats: int = DEFAULT_REPEATS, split: str | None = "full") -> dict[str, Any]:
    n_boot = int(cfg.section("statistics").get("bootstrap_samples", 2000))
    seed = int(cfg.raw.get("project", {}).get("random_seed", 42))
    records = list(iter_references(cfg, split=split))
    uids = [r["utterance_id"] for r in records]

    floor_rows: dict[str, list[dict[str, float]]] = {s: [] for s in REPEAT_SOURCES}
    floor_by_uid: dict[str, dict[str, dict[str, float]]] = {s: {} for s in REPEAT_SOURCES}
    for uid in uids:
        for source in REPEAT_SOURCES:
            base = _labels(cfg, raw_response_path(cfg, "read", f"{uid}_{source}"))
            copies = [base] if base else []
            copies += [c for c in (_labels(cfg, _repeat_path(cfg, uid, source, r)) for r in range(1, repeats + 1)) if c]
            stat = _per_item_pairwise(copies)
            if stat:
                floor_rows[source].append(stat)
                floor_by_uid[source][uid] = stat

    # Cross-condition comparisons on single calls (the original design).
    cross: dict[str, list[dict[str, float]]] = {}
    exact: dict[str, dict[str, Any]] = {}
    ref_labels = {u: _labels(cfg, raw_response_path(cfg, "read", f"{u}_reference")) for u in uids}
    for cond in ACOUSTIC_CONDITIONS:
        rows, rows_exact, rows_inexact, rows_bytes = [], [], [], []
        floor_on_exact: list[dict[str, float]] = []
        for rec in records:
            uid = rec["utterance_id"]
            ref, hyp = ref_labels[uid], _labels(cfg, raw_response_path(cfg, "read", f"{uid}_transcript_{cond}"))
            if not ref or not hyp:
                continue
            stat = _pair_stats(ref, hyp)
            rows.append(stat)
            same = normalize_text(_source_text(cfg, rec, f"transcript_{cond}")) == normalize_text(rec["reference_text"])
            (rows_exact if same else rows_inexact).append(stat)
            if _source_text(cfg, rec, f"transcript_{cond}") == rec["reference_text"]:
                rows_bytes.append(stat)
            if same and uid in floor_by_uid["reference"]:
                floor_on_exact.append(floor_by_uid["reference"][uid])
        cross[f"reference_vs_{cond}"] = rows
        exact[cond] = {
            "n_exact": len(rows_exact),
            "n_inexact": len(rows_inexact),
            "exact_match": _summarize(rows_exact, seed, n_boot),
            "not_exact": _summarize(rows_inexact, seed, n_boot),
            "byte_identical": _summarize(rows_bytes, seed, n_boot) | {"items": len(rows_bytes)},
            "reference_repeat_floor_same_items": _summarize(floor_on_exact, seed, n_boot),
        }

    # Clean vs harsh transcript labels where the two transcripts are identical.
    same_text: list[dict[str, float]] = []
    for rec in records:
        uid = rec["utterance_id"]
        if _source_text(cfg, rec, "transcript_clean") and _source_text(cfg, rec, "transcript_clean") == _source_text(cfg, rec, "transcript_harsh"):
            a = _labels(cfg, raw_response_path(cfg, "read", f"{uid}_transcript_clean"))
            b = _labels(cfg, raw_response_path(cfg, "read", f"{uid}_transcript_harsh"))
            if a and b:
                same_text.append(_pair_stats(a, b))

    # Formatting control.
    fmt: dict[str, list[dict[str, float]]] = {"reference_norm_vs_clean_norm": [], "reference_vs_clean_original": []}
    for uid in uids:
        a, b = _labels(cfg, _format_path(cfg, uid, "reference_norm")), _labels(cfg, _format_path(cfg, uid, "transcript_clean_norm"))
        if a and b:
            fmt["reference_norm_vs_clean_norm"].append(_pair_stats(a, b))
            ref, hyp = ref_labels[uid], _labels(cfg, raw_response_path(cfg, "read", f"{uid}_transcript_clean"))
            if ref and hyp:
                fmt["reference_vs_clean_original"].append(_pair_stats(ref, hyp))

    empty = {}
    for label in ("reference_norm", "transcript_clean_norm"):
        got = [_labels(cfg, _format_path(cfg, u, label)) for u in uids]
        got = [g for g in got if g]
        empty[label] = {"items": len(got), "no_intent": float(np.mean([not g["intent"] for g in got])) if got else float("nan"), "no_topics": float(np.mean([not g["topics"] for g in got])) if got else float("nan")}

    result = {
        "unpunctuated_input_empty_labels": empty,
        "repeats_per_item": repeats + 1,
        "noise_floor": {s: _summarize(floor_rows[s], seed, n_boot) | {"items": len(floor_rows[s])} for s in REPEAT_SOURCES},
        "cross_condition": {k: _summarize(v, seed, n_boot) for k, v in cross.items()},
        "clean_vs_harsh_identical_transcript": _summarize(same_text, seed, n_boot) | {"items": len(same_text)},
        "exact_match_subset": exact,
        "formatting_control": {k: _summarize(v, seed, n_boot) | {"items": len(v)} for k, v in fmt.items()},
    }
    out = cfg.results_dir / "stability.json"
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    logger.info("Wrote %s", out)
    return result


def stability_frame(result: dict[str, Any]) -> pd.DataFrame:
    """Flat table for the paper: intent and topic disagreement per comparison."""
    rows = []

    def add(name: str, block: dict[str, Any]) -> None:
        rows.append(
            {
                "comparison": name,
                "intent_disagree": block["intent_disagree"]["mean"],
                "intent_lo": block["intent_disagree"]["lo"],
                "intent_hi": block["intent_disagree"]["hi"],
                "topic_disagree": block["topic_disagree"]["mean"],
                "topic_lo": block["topic_disagree"]["lo"],
                "topic_hi": block["topic_disagree"]["hi"],
                "sentiment_disagree": block["sentiment_disagree"]["mean"],
                "n": block["intent_disagree"]["n"],
            }
        )

    for source, block in result["noise_floor"].items():
        add(f"same text repeated ({source})", block)
    for name, block in result["cross_condition"].items():
        add(name.replace("_", " "), block)
    add("clean vs 0 dB, identical transcript", result["clean_vs_harsh_identical_transcript"])
    for cond, block in result["exact_match_subset"].items():
        add(f"reference vs {cond}, exact-match items only", block["exact_match"])
    for cond, block in result["exact_match_subset"].items():
        add(f"reference vs {cond}, byte-identical text only", block["byte_identical"])
    return pd.DataFrame(rows)
