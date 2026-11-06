"""Second labeler (E8): does an LLM intent extractor show the same repeat instability, and does temperature drive it?

Same task as Deepgram text intelligence: free-text intent and a sentiment label for one utterance.
Each model labels the same reference texts several times at temperature 0 and temperature 1.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import numpy as np

from src.config_loader import ExperimentConfig
from src.generate_dataset import iter_references
from src.llm_client import map_batches
from src.logging_utils import setup_logging
from src.semantic_match import judge_pairs
from src.statistical_analysis import _bootstrap_mean
from src.text_normalize import normalize_intent

logger = setup_logging()

MODELS = ("openai/gpt-4o-mini", "anthropic/claude-haiku-5.5")
TEMPERATURES = (0.0, 1.0)
REPEATS = 3
SUBSET = 200
PROMPT = (
    "You analyse one customer-service utterance. State the caller's intent as a short verb phrase and the sentiment "
    "(positive, neutral, or negative).\nReturn JSON: {\"intent\": \"...\", \"sentiment\": \"...\"}\n\nUtterance: "
)


def _rate(flags: list[float], n_boot: int) -> dict[str, float]:
    if not flags:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    mean, lo, hi = _bootstrap_mean(np.asarray(flags, dtype=float), n_boot, 42)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(flags)}


def analyze_llm_stability(cfg: ExperimentConfig, split: str | None = "full") -> dict[str, Any]:
    n_boot = int(cfg.section("statistics").get("bootstrap_samples", 2000))
    records = sorted(iter_references(cfg, split=split), key=lambda r: r["utterance_id"])[:SUBSET]
    out: dict[str, Any] = {"items": len(records), "repeats": REPEATS, "models": {}}
    for model in MODELS:
        out["models"][model] = {}
        for temp in TEMPERATURES:
            prompts = [PROMPT + rec["reference_text"] for rec in records for _ in range(REPEATS)]
            tags = [f"{rec['utterance_id']}_r{r}" for rec in records for r in range(REPEATS)]
            answers = map_batches(cfg, prompts, model, temp, tags)
            by_item: list[list[dict[str, Any]]] = []
            for i in range(len(records)):
                chunk = [a for a in answers[i * REPEATS : (i + 1) * REPEATS] if a.get("intent")]
                by_item.append(chunk)
            usable = [c for c in by_item if len(c) >= 2]
            string_flags, sentiment_flags, pairs = [], [], []
            for chunk in usable:
                intents = [normalize_intent(str(a["intent"])) for a in chunk]
                sents = [str(a.get("sentiment", "")).lower() for a in chunk]
                combos = list(itertools.combinations(range(len(chunk)), 2))
                string_flags.append(float(np.mean([intents[i] != intents[j] for i, j in combos])))
                sentiment_flags.append(float(np.mean([sents[i] != sents[j] for i, j in combos])))
                pairs += [("intent", *sorted((intents[i], intents[j]))) for i, j in combos if intents[i] != intents[j]]
            verdict = judge_pairs(cfg, pairs)
            semantic_flags = []
            for chunk in usable:
                intents = [normalize_intent(str(a["intent"])) for a in chunk]
                combos = list(itertools.combinations(range(len(chunk)), 2))
                semantic_flags.append(
                    float(np.mean([intents[i] != intents[j] and not verdict.get(("intent", *sorted((intents[i], intents[j]))), True) for i, j in combos]))
                )
            out["models"][model][str(temp)] = {
                "items_usable": len(usable),
                "intent_string_disagreement": _rate(string_flags, n_boot),
                "intent_semantic_disagreement": _rate(semantic_flags, n_boot),
                "sentiment_disagreement": _rate(sentiment_flags, n_boot),
            }
            logger.info("llm stability %s T=%s usable=%s", model, temp, len(usable))
    path = cfg.results_dir / "llm_stability.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    logger.info("Wrote %s", path)
    return out
