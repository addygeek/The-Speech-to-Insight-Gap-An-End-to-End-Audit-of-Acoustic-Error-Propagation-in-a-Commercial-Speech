"""Validate reference dataset schema and coverage."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from src.config_loader import ExperimentConfig
from src.generate_dataset import load_references
from src.logging_utils import setup_logging

logger = setup_logging()

REQUIRED_FIELDS = [
    "utterance_id",
    "reference_text",
    "scenario",
    "critical_spans",
    "critical_span_types",
]


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    texts: list[str] = []
    scenario_counts: Counter[str] = Counter()
    span_type_counts: Counter[str] = Counter()

    for i, rec in enumerate(records):
        for field in REQUIRED_FIELDS:
            if field not in rec:
                errors.append(f"row {i}: missing {field}")
            elif field == "critical_spans" and rec.get(field) == []:
                errors.append(f"row {i}: empty critical_spans")
        text = rec.get("reference_text", "")
        texts.append(text.strip().lower())
        scenario_counts[rec.get("scenario", "unknown")] += 1
        for st in rec.get("critical_span_types", []):
            span_type_counts[st] += 1

    dupes = len(texts) - len(set(texts))
    if dupes:
        errors.append(f"duplicate reference_text count: {dupes}")

    return {
        "n_utterances": len(records),
        "scenario_distribution": dict(scenario_counts),
        "critical_span_type_counts": dict(span_type_counts),
        "errors": errors,
        "valid": len(errors) == 0,
    }


def merge_summary(cfg: ExperimentConfig, section: dict[str, Any]) -> None:
    path = cfg.results_dir / "summary.json"
    summary: dict[str, Any] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            summary = json.load(f)
    summary["dataset_validation"] = section
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_validate_dataset(cfg: ExperimentConfig) -> dict[str, Any]:
    records = load_references(cfg.references_jsonl)
    report = validate_records(records)
    merge_summary(cfg, report)
    if report["valid"]:
        logger.info("Dataset validation passed (%d utterances)", report["n_utterances"])
    else:
        logger.error("Dataset validation failed: %s", report["errors"])
    return report
