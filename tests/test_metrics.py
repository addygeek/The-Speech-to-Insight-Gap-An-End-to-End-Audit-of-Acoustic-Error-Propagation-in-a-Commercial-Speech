from src.calculate_critical_errors import critical_span_error_rate
from src.calculate_disagreement import sentiment_disagreement, topic_disagreement
from src.calculate_wer import compute_wer
from src.config_loader import load_config
from src.text_normalize import normalize_intent, normalize_text


def test_normalize_negation_preserved():
    policy = {"lowercase": True, "remove_punctuation": True, "collapse_whitespace": True}
    assert "do not" in normalize_text("I do not want to cancel.", policy)


def test_wer_identical_zero():
    cfg = load_config()
    r = compute_wer("hello world", "hello world", cfg)
    assert r.wer == 0.0


def test_critical_span_error():
    cfg = load_config()
    rate, err, total = critical_span_error_rate(
        "do not cancel order",
        "cancel order",
        ["do not"],
        cfg,
    )
    assert err == 1
    assert total == 1
    assert rate == 1.0


def test_sentiment_disagreement():
    assert sentiment_disagreement("positive", "negative")


def test_intent_synonym():
    cfg = load_config()
    syn = cfg.section("intent_normalization").get("synonym_map", {})
    assert normalize_intent("Cancel my order", syn) == "cancellation"
