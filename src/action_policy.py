"""Closed-label routing and a downstream action (E5, E9).

A simple, published policy turns API labels into the action an agent would take:
route = the queue the open intent phrase maps to; escalate = sentiment is negative.
Action disagreement then measures wrong actions directly, and the repeated-call floor shows how many
come from label instability alone.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.cache import ACOUSTIC_CONDITIONS, raw_response_path
from src.config_loader import ExperimentConfig
from src.extra_conditions import EXTRA_CONDITIONS, read_extra_path
from src.generate_dataset import iter_references
from src.second_voice import VOICE_CONDITIONS
from src.llm_client import JUDGE_MODEL, map_batches
from src.logging_utils import setup_logging
from src.stability_baseline import DEFAULT_REPEATS, _labels, _repeat_path
from src.statistical_analysis import _bootstrap_mean

logger = setup_logging()

QUEUES = (
    "customer_support",
    "appointment_scheduling",
    "travel_booking",
    "banking_payments",
    "delivery_tracking",
    "technical_support",
    "complaints_compliments",
    "requests_cancellations",
)
BATCH = 25
ROUTE_PROMPT = (
    "Route each phrase (a caller intent from a speech-analytics API) to exactly one queue from this list: "
    + ", ".join(QUEUES)
    + ".\nReturn JSON: {\"answers\": [{\"id\": <id>, \"queue\": \"<queue>\"}, ...]}\n\nPhrases:\n"
)


def route_intents(cfg: ExperimentConfig, phrases: set[str]) -> dict[str, str]:
    """Map each unique intent phrase to a queue with the pinned judge model at temperature 0."""
    ordered = sorted(p for p in phrases if p)
    prompts, chunks = [], []
    for start in range(0, len(ordered), BATCH):
        chunk = ordered[start : start + BATCH]
        prompts.append(ROUTE_PROMPT + "\n".join(json.dumps({"id": i, "phrase": p}) for i, p in enumerate(chunk)))
        chunks.append(chunk)
    routed: dict[str, str] = {}
    for chunk, answer in zip(chunks, map_batches(cfg, prompts, JUDGE_MODEL, 0.0)):
        for item in answer.get("answers", []):
            try:
                if item["queue"] in QUEUES:
                    routed[chunk[int(item["id"])]] = item["queue"]
            except (KeyError, IndexError, ValueError, TypeError):
                continue
    return routed


def action_of(labels: dict[str, Any], routes: dict[str, str]) -> tuple[str, bool] | None:
    queue = routes.get(labels["intent"])
    return None if queue is None else (queue, labels["sentiment"] == "negative")


def _rate(flags: list[float], n_boot: int) -> dict[str, float]:
    if not flags:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    mean, lo, hi = _bootstrap_mean(np.asarray(flags, dtype=float), n_boot, 42)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(flags)}


def analyze_actions(cfg: ExperimentConfig, split: str | None = "full", repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    n_boot = int(cfg.section("statistics").get("bootstrap_samples", 2000))
    records = list(iter_references(cfg, split=split))

    def sources(uid: str) -> dict[str, dict[str, Any] | None]:
        out = {"reference": _labels(cfg, raw_response_path(cfg, "read", f"{uid}_reference"))}
        for r in range(1, repeats + 1):
            out[f"reference_r{r}"] = _labels(cfg, _repeat_path(cfg, uid, "reference", r))
        for c in ACOUSTIC_CONDITIONS:
            out[c] = _labels(cfg, raw_response_path(cfg, "read", f"{uid}_transcript_{c}"))
        for c in (*EXTRA_CONDITIONS, *VOICE_CONDITIONS):
            out[c] = _labels(cfg, read_extra_path(cfg, uid, c))
        return out

    table = {rec["utterance_id"]: sources(rec["utterance_id"]) for rec in records}
    phrases = {lab["intent"] for row in table.values() for lab in row.values() if lab}
    routes = route_intents(cfg, phrases)
    logger.info("routed %s of %s unique intent phrases", len(routes), len(phrases))

    expected = {r["utterance_id"]: r["scenario"] for r in records}
    ref_accuracy = [float(routes.get(row["reference"]["intent"]) == expected[uid]) for uid, row in table.items() if row["reference"] and row["reference"]["intent"] in routes]

    def compare(left: str, right: str) -> dict[str, dict[str, float]]:
        route_flags, escalate_flags, action_flags = [], [], []
        for row in table.values():
            a = action_of(row[left], routes) if row.get(left) else None
            b = action_of(row[right], routes) if row.get(right) else None
            if a is None or b is None:
                continue
            route_flags.append(float(a[0] != b[0]))
            escalate_flags.append(float(a[1] != b[1]))
            action_flags.append(float(a != b))
        return {"route": _rate(route_flags, n_boot), "escalate": _rate(escalate_flags, n_boot), "action": _rate(action_flags, n_boot)}

    out: dict[str, Any] = {
        "queues": list(QUEUES),
        "routed_phrases": len(routes),
        "unique_phrases": len(phrases),
        "reference_route_matches_scenario": _rate(ref_accuracy, n_boot),
        "same_text_repeat_floor": compare("reference", "reference_r1"),
        "vs_reference": {c: compare("reference", c) for c in (*ACOUSTIC_CONDITIONS, *EXTRA_CONDITIONS, *VOICE_CONDITIONS)},
    }
    path = cfg.results_dir / "actions.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    logger.info("Wrote %s", path)
    return out
