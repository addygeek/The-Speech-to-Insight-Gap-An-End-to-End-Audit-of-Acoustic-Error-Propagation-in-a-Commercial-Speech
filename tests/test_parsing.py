import json
from pathlib import Path

from src.config_loader import load_config
from src.parse_responses import extract_sentiment, extract_topics, extract_transcript, parse_intelligence

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def test_parse_listen_fixture():
    cfg = load_config()
    payload = json.loads((FIXTURES / "listen_response.json").read_text(encoding="utf-8"))
    assert "cancel" in extract_transcript(payload).lower()
    label, score = extract_sentiment(payload)
    assert label == "neutral"
    assert score is not None
    intel = parse_intelligence(payload, cfg)
    assert intel["topics"]


def test_parse_read_fixture():
    cfg = load_config()
    payload = json.loads((FIXTURES / "read_response.json").read_text(encoding="utf-8"))
    label, _ = extract_sentiment(payload)
    assert label == "negative"
    topics = extract_topics(payload)
    assert topics
