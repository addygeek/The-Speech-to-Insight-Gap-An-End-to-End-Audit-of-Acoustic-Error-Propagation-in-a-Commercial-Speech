"""Generate synthetic reference utterances with critical spans."""

from __future__ import annotations

import json
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any, Iterator

import httpx

from src.config_loader import ExperimentConfig
from src.logging_utils import setup_logging
from src.text_normalize import normalize_text

logger = setup_logging()

NAMES = ["Jordan Lee", "Maria Santos", "Alex Chen", "Sam Patel", "Taylor Brooks"]
PLACES = ["Chicago", "Austin", "Denver", "Seattle", "Boston"]
DATES = ["March 12", "April 3", "June 21", "October 5"]
TIMES = ["9:30 AM", "2:15 PM", "11:00 AM"]
AMOUNTS = ["$45.99", "$120.00", "$8.50", "$250.00", "$19.40", "$75.00"]

# Extra clauses keep each utterance in the 25–40 word band so clean TTS is long
# enough to produce recognition errors under noise. Critical spans stay in the core sentence.
DETAIL_CLAUSES = [
    "I already tried the website twice and it would not save the change under my account.",
    "Please send the updated confirmation to the same email address you have on file.",
    "I need this settled before the end of the business day because I travel tomorrow morning.",
    "The last agent I spoke with asked me to call back if the charge was still showing.",
    "I can wait on the line while you check the record, and I do have my account number ready.",
    "This is the second time I have contacted support about the same problem this week.",
    "Please do not transfer me again unless the next person can actually change the booking.",
    "I am calling from my mobile phone because I am away from my home office right now.",
]

TEMPLATES: list[dict[str, Any]] = [
    {
        "scenario": "customer_support",
        "template": "Hi, my name is {name}. I do not want to cancel my order for {amount}.",
        "expected_intent": "retain_order",
        "expected_sentiment": "negative",
        "critical_spans": ["{name}", "do not", "{amount}"],
        "critical_span_types": ["name", "negation", "amount"],
    },
    {
        "scenario": "requests_cancellations",
        "template": "Please cancel my order for {amount} placed yesterday.",
        "expected_intent": "cancellation",
        "expected_sentiment": "neutral",
        "critical_spans": ["cancel", "{amount}"],
        "critical_span_types": ["action", "amount"],
    },
    {
        "scenario": "appointment_scheduling",
        "template": "I need to book an appointment with Dr. {name} on {date} at {time}.",
        "expected_intent": "scheduling",
        "expected_sentiment": "neutral",
        "critical_spans": ["{name}", "{date}", "{time}"],
        "critical_span_types": ["name", "date", "time"],
    },
    {
        "scenario": "travel_booking",
        "template": "Can you change my flight to {place} on {date}?",
        "expected_intent": "change_booking",
        "expected_sentiment": "neutral",
        "critical_spans": ["{place}", "{date}"],
        "critical_span_types": ["place", "date"],
    },
    {
        "scenario": "banking_payments",
        "template": "I was charged {amount} twice on my card ending in 4421.",
        "expected_intent": "dispute_charge",
        "expected_sentiment": "negative",
        "critical_spans": ["{amount}", "4421"],
        "critical_span_types": ["amount", "number"],
    },
    {
        "scenario": "delivery_tracking",
        "template": "Where is my package going to {place}? It should arrive by {date}.",
        "expected_intent": "track_delivery",
        "expected_sentiment": "neutral",
        "critical_spans": ["{place}", "{date}"],
        "critical_span_types": ["place", "date"],
    },
    {
        "scenario": "technical_support",
        "template": "My internet has been down since {time} and I cannot work from {place}.",
        "expected_intent": "technical_issue",
        "expected_sentiment": "negative",
        "critical_spans": ["{time}", "cannot", "{place}"],
        "critical_span_types": ["time", "negation", "place"],
    },
    {
        "scenario": "complaints_compliments",
        "template": "Thank you {name}, the support team in {place} was excellent today.",
        "expected_intent": "compliment",
        "expected_sentiment": "positive",
        "critical_spans": ["{name}", "{place}"],
        "critical_span_types": ["name", "place"],
    },
]


def _pad_to_word_band(text: str, rng: random.Random, min_words: int, max_words: int) -> str:
    clauses = DETAIL_CLAUSES.copy()
    rng.shuffle(clauses)
    for clause in clauses:
        if len(text.split()) >= min_words:
            break
        if len(text.split()) + len(clause.split()) > max_words:
            continue
        text = f"{text} {clause}"
    if len(text.split()) < min_words:
        ticket = rng.randint(100000, 999999)
        extra = f"My ticket number is {ticket} and I need a written confirmation today."
        if len(text.split()) + len(extra.split()) <= max_words + 4:
            text = f"{text} {extra}"
    return text


def _fill_template(
    tpl: dict[str, Any],
    rng: random.Random,
    min_words: int = 25,
    max_words: int = 40,
) -> dict[str, Any]:
    slots = {
        "name": rng.choice(NAMES),
        "place": rng.choice(PLACES),
        "date": rng.choice(DATES),
        "time": rng.choice(TIMES),
        "amount": rng.choice(AMOUNTS),
    }
    text = _pad_to_word_band(tpl["template"].format(**slots), rng, min_words, max_words)
    critical_spans = [s.format(**slots) if "{" in s else s for s in tpl["critical_spans"]]
    uid = str(uuid.uuid4())[:8]
    return {
        "utterance_id": f"utt_{uid}",
        "reference_text": text,
        "scenario": tpl["scenario"],
        "expected_intent": tpl.get("expected_intent", ""),
        "expected_sentiment": tpl.get("expected_sentiment", ""),
        "critical_spans": critical_spans,
        "critical_span_types": tpl.get("critical_span_types", []),
        "generation_template_id": tpl["template"][:40],
        "dataset_split": "pilot",
        "word_count": len(text.split()),
    }


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("expected object", text, 0)
    return parsed


def _grok_model(cfg: ExperimentConfig) -> str:
    configured = str(cfg.section("dataset").get("llm_model", "")).strip()
    if configured:
        return configured
    models = os.environ.get("GROK_MODELS", "grok-4.7-high")
    return models.split(",")[0].strip() or "grok-4.7-high"


def _post_anthropic_messages(url: str, key: str, model: str, prompt: str) -> str:
    resp = httpx.post(
        url,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90.0,
    )
    resp.raise_for_status()
    body = resp.json()
    blocks = body.get("content") or []
    texts = [str(block.get("text", "")) for block in blocks if isinstance(block, dict)]
    return "\n".join(t for t in texts if t).strip()


def _llm_paraphrase(record: dict[str, Any], cfg: ExperimentConfig) -> dict[str, Any] | None:
    provider = str(cfg.section("dataset").get("llm_provider", "openrouter")).strip().lower()
    model = _grok_model(cfg) if provider == "grok" else str(
        cfg.section("dataset").get("llm_model", "openai/gpt-4o-mini")
    )
    prompt = (
        "Paraphrase the following customer utterance in 25 to 40 words. "
        "Preserve names, dates, amounts, negation, and intent. Return JSON with keys "
        "reference_text and critical_spans (list of substring spans still present).\n\n"
        f"Original: {record['reference_text']}\n"
        f"Critical spans: {record['critical_spans']}"
    )
    try:
        if provider in ("openai", "openrouter"):
            if provider == "openrouter":
                key = os.environ.get("OPENROUTER_API_KEY", "").strip()
                base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
            else:
                key = os.environ.get("OPENAI_API_KEY", "").strip()
                base = "https://api.openai.com/v1"
            if not key:
                return None
            resp = httpx.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        elif provider == "grok":
            key = os.environ.get("GROK_API_KEY", "").strip()
            base = os.environ.get(
                "GROK_BASE_URL",
                "https://anthropic-middleware.parthsolankipd.workers.dev/grok",
            ).rstrip("/")
            if not key:
                return None
            content = _post_anthropic_messages(f"{base}/v1/messages", key, model, prompt)
        else:
            key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not key:
                return None
            content = _post_anthropic_messages(
                "https://api.anthropic.com/v1/messages", key, model, prompt
            )
        parsed = _extract_json_object(content)
        new_text = parsed.get("reference_text", "").strip()
        if not new_text:
            return None
        spans = parsed.get("critical_spans", record["critical_spans"])
        out = dict(record)
        out["reference_text"] = new_text
        out["critical_spans"] = spans
        out["generation_template_id"] = "llm_paraphrase"
        return out
    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("LLM paraphrase failed: %s", exc)
        return None


def generate_records(cfg: ExperimentConfig, count: int, split: str = "pilot") -> list[dict[str, Any]]:
    rng = random.Random(cfg.random_seed)
    mode = cfg.section("dataset").get("generation_mode", "templates")
    min_words = int(cfg.section("dataset").get("min_words", 25))
    max_words = int(cfg.section("dataset").get("max_words", 40))
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    templates = TEMPLATES.copy()
    rng.shuffle(templates)
    idx = 0
    while len(records) < count:
        tpl = templates[idx % len(templates)]
        idx += 1
        rec = _fill_template(tpl, rng, min_words=min_words, max_words=max_words)
        rec["dataset_split"] = split
        norm = normalize_text(rec["reference_text"])
        if norm in seen:
            continue
        seen.add(norm)
        if mode in ("llm", "hybrid") and len(records) % 3 == 2:
            paraphrased = _llm_paraphrase(rec, cfg)
            if paraphrased:
                pn = normalize_text(paraphrased["reference_text"])
                if pn not in seen:
                    paraphrased["dataset_split"] = split
                    records.append(paraphrased)
                    seen.add(pn)
                    continue
        records.append(rec)
    return records


def load_references(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_references(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def iter_references(cfg: ExperimentConfig, split: str | None = None) -> Iterator[dict[str, Any]]:
    for rec in load_references(cfg.references_jsonl):
        if split is None or rec.get("dataset_split") == split:
            yield rec


def run_generate_dataset(cfg: ExperimentConfig, count: int, split: str = "pilot") -> int:
    records = generate_records(cfg, count=count, split=split)
    save_references(cfg.references_jsonl, records)
    logger.info("Wrote %d utterances to %s", len(records), cfg.references_jsonl)
    return len(records)
