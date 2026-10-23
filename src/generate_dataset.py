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
