"""Semantic equivalence of open intent and topic labels (E4).

String disagreement is not semantic failure. This module scores every string-disagreeing pair three ways:
a lexical overlap, an LLM judge (pinned model, temperature 0, fixed prompt), and a human sample file.
Pairs come from three sources so the paraphrase rate can be compared across them:
repeated calls on identical text (noise floor), reference vs clean transcript, and clean vs 0 dB transcript.
"""

from __future__ import annotations

import csv
import json
import re
from typing import Any

import numpy as np

from src.cache import raw_response_path
from src.config_loader import ExperimentConfig
from src.generate_dataset import iter_references
from src.llm_client import JUDGE_MODEL, map_batches
from src.logging_utils import setup_logging
from src.stability_baseline import DEFAULT_REPEATS, _labels, _repeat_path, _source_text
from src.statistical_analysis import _bootstrap_mean

logger = setup_logging()

BATCH = 15
HUMAN_SAMPLE = 100
STOP = {"a", "an", "the", "to", "of", "for", "and", "or", "on", "in", "with", "about", "my", "your", "their", "is", "be", "from"}

INTENT_PROMPT = (
    "Each item is a pair of short phrases that a speech-analytics API produced for the same customer call. "
    "For each pair decide whether both phrases name the SAME caller goal (a paraphrase or a different level of detail "
    "about the same goal counts as same). Different goals, or one phrase naming a goal the other does not, are different.\n"
    "Return JSON: {\"answers\": [{\"id\": <id>, \"same\": true|false}, ...]}\n\nPairs:\n"
)
TOPIC_PROMPT = (
    "Each item is a pair of topic lists that a speech-analytics API produced for the same customer call. "
    "Decide whether both lists describe the SAME subject matter (synonyms and different granularity of the same subjects count as same; "
    "a subject present in one list and absent in the other counts as different).\n"
    "Return JSON: {\"answers\": [{\"id\": <id>, \"same\": true|false}, ...]}\n\nPairs:\n"
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP}


def lexical_overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _pair_key(kind: str, a: str, b: str) -> tuple[str, str, str]:
    left, right = sorted((a, b))
    return kind, left, right


def judge_pairs(cfg: ExperimentConfig, pairs: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], bool]:
    """Judge unique (kind, a, b) pairs. Returns same/different per pair; pairs the judge failed on are dropped."""
    unique = sorted(set(pairs))
    prompts, index = [], []
    for kind in ("intent", "topic"):
        todo = [p for p in unique if p[0] == kind]
        for start in range(0, len(todo), BATCH):
            chunk = todo[start : start + BATCH]
            body = "\n".join(json.dumps({"id": i, "a": a, "b": b}) for i, (_, a, b) in enumerate(chunk))
            prompts.append((INTENT_PROMPT if kind == "intent" else TOPIC_PROMPT) + body)
            index.append(chunk)
    logger.info("judge batches: %s (%s unique pairs)", len(prompts), len(unique))
    out: dict[tuple[str, str, str], bool] = {}
    for chunk, answer in zip(index, map_batches(cfg, prompts, JUDGE_MODEL, 0.0)):
        for item in answer.get("answers", []):
            try:
                out[chunk[int(item["id"])]] = bool(item["same"])
            except (KeyError, IndexError, ValueError, TypeError):
                continue
    return out


def _label_text(labels: dict[str, Any], kind: str) -> str:
    return labels["intent"] if kind == "intent" else "|".join(sorted(labels["topics"]))


def collect_pairs(cfg: ExperimentConfig, split: str | None = "full", repeats: int = DEFAULT_REPEATS) -> dict[str, list[dict[str, Any]]]:
    """For each comparison group, one row per item with the intent and topic strings on both sides."""
    groups: dict[str, list[dict[str, Any]]] = {"noise_floor_reference": [], "reference_vs_clean": [], "clean_vs_harsh_same_transcript": []}
    for rec in iter_references(cfg, split=split):
        uid = rec["utterance_id"]
        ref = _labels(cfg, raw_response_path(cfg, "read", f"{uid}_reference"))
        clean = _labels(cfg, raw_response_path(cfg, "read", f"{uid}_transcript_clean"))
        harsh = _labels(cfg, raw_response_path(cfg, "read", f"{uid}_transcript_harsh"))
        rep1 = _labels(cfg, _repeat_path(cfg, uid, "reference", 1))
        for name, left, right in (
            ("noise_floor_reference", ref, rep1),
            ("reference_vs_clean", ref, clean),
            ("clean_vs_harsh_same_transcript", clean, harsh),
        ):
            same_text = _source_text(cfg, rec, "transcript_clean") == _source_text(cfg, rec, "transcript_harsh")
            if name == "clean_vs_harsh_same_transcript" and not same_text:
                continue
            if left and right:
                groups[name].append({"uid": uid, "left": left, "right": right})
    return groups


def analyze_semantic(cfg: ExperimentConfig, split: str | None = "full", repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    n_boot = int(cfg.section("statistics").get("bootstrap_samples", 2000))
    groups = collect_pairs(cfg, split, repeats)
    wanted: list[tuple[str, str, str]] = []
    for rows in groups.values():
        for row in rows:
            for kind in ("intent", "topic"):
                a, b = _label_text(row["left"], kind), _label_text(row["right"], kind)
                if a != b:
                    wanted.append(_pair_key(kind, a, b))
    verdict = judge_pairs(cfg, wanted)

    result: dict[str, Any] = {"judge_model": JUDGE_MODEL, "judge_pairs": len(set(wanted)), "groups": {}}
    sample_rows: list[dict[str, Any]] = []
    for name, rows in groups.items():
        block: dict[str, Any] = {"items": len(rows)}
        for kind in ("intent", "topic"):
            semantic, same_flags, string_dis, unjudged, overlaps = [], [], [], 0, []
            for row in rows:
                a, b = _label_text(row["left"], kind), _label_text(row["right"], kind)
                if a == b:
                    string_dis.append(0.0)
                    semantic.append(0.0)
                    continue
                string_dis.append(1.0)
                overlaps.append(lexical_overlap(a, b))
                same = verdict.get(_pair_key(kind, a, b))
                if same is None:
                    unjudged += 1
                    continue
                semantic.append(0.0 if same else 1.0)
                same_flags.append(1.0 if same else 0.0)
                sample_rows.append({"group": name, "kind": kind, "uid": row["uid"], "a": a, "b": b, "judge_same": int(same)})
            block[kind] = {
                "string_disagreement": float(np.mean(string_dis)) if string_dis else float("nan"),
                "semantic_disagreement": _ci(semantic, n_boot),
                "string_disagree_pairs": int(sum(string_dis)),
                "unjudged_pairs": unjudged,
                "paraphrase_fraction": _ci(same_flags, n_boot),
                "mean_lexical_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
            }
        result["groups"][name] = block

    _write_human_sample(cfg, sample_rows)
    result["human_agreement"] = human_agreement(cfg)
    out = cfg.results_dir / "semantic_match.json"
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    logger.info("Wrote %s", out)
    return result


def _ci(values: list[float], n_boot: int) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean, lo, hi = _bootstrap_mean(np.asarray(values, dtype=float), n_boot, 42)
    return {"mean": mean, "lo": lo, "hi": hi}


def _write_human_sample(cfg: ExperimentConfig, rows: list[dict[str, Any]]) -> None:
    """Seeded sample of string-disagreeing intent pairs for hand labeling. Never overwrites labels already entered."""
    path = cfg.results_dir / "human_label_sample.csv"
    if path.is_file():
        return
    intents = [r for r in rows if r["kind"] == "intent"]
    if not intents:
        return
    pick = np.random.default_rng(42).choice(len(intents), size=min(HUMAN_SAMPLE, len(intents)), replace=False)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "group", "uid", "a", "b", "judge_same", "human_same"])
        writer.writeheader()
        for n, i in enumerate(sorted(pick)):
            writer.writerow({**{k: intents[int(i)][k] for k in ("group", "uid", "a", "b", "judge_same")}, "pair_id": n, "human_same": ""})
    logger.info("Wrote %s: fill human_same with 1 (same goal) or 0, then rerun", path)


def cohen_kappa(x: list[int], y: list[int]) -> float:
    n = len(x)
    observed = sum(a == b for a, b in zip(x, y)) / n
    expected = sum((x.count(v) / n) * (y.count(v) / n) for v in (0, 1))
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def human_agreement(cfg: ExperimentConfig) -> dict[str, Any]:
    """Judge-vs-human agreement and the human paraphrase rate. Empty until the sample is labeled."""
    path = cfg.results_dir / "human_label_sample.csv"
    if not path.is_file():
        return {"labeled": 0}
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_same"].strip() in ("0", "1")]
    if not rows:
        return {"labeled": 0}
    human = [int(r["human_same"]) for r in rows]
    judge = [int(r["judge_same"]) for r in rows]
    return {
        "labeled": len(rows),
        "human_paraphrase_fraction": sum(human) / len(human),
        "judge_human_accuracy": sum(a == b for a, b in zip(human, judge)) / len(human),
        "judge_human_kappa": cohen_kappa(human, judge),
    }
