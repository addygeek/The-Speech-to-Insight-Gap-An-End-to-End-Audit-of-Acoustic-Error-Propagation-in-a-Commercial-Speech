import numpy as np

from src.action_policy import action_of
from src.extra_conditions import make_babble, reverb, telephone
from src.perturb_audio import mix_at_snr
from src.semantic_match import cohen_kappa, lexical_overlap
from src.stability_baseline import _pair_stats, _per_item_pairwise


def _label(intent, topics=(), sentiment="neutral"):
    return {"intent": intent, "topics": set(topics), "sentiment": sentiment}


def test_pair_stats_identical_is_zero_disagreement():
    a = _label("check order", ["delivery"])
    assert _pair_stats(a, dict(a)) == {"intent_disagree": 0.0, "topic_disagree": 0.0, "topic_jaccard": 1.0, "sentiment_disagree": 0.0}


def test_pair_stats_detects_change():
    s = _pair_stats(_label("a", ["x"]), _label("b", ["x", "y"], "negative"))
    assert s["intent_disagree"] == 1.0 and s["topic_disagree"] == 1.0 and s["sentiment_disagree"] == 1.0
    assert s["topic_jaccard"] == 0.5


def test_per_item_pairwise_needs_two_copies():
    assert _per_item_pairwise([_label("a")]) is None
    assert _per_item_pairwise([_label("a"), _label("a"), _label("b")])["intent_disagree"] == 2 / 3


def test_lexical_overlap_ignores_stopwords():
    assert lexical_overlap("check the order", "check order") == 1.0
    assert lexical_overlap("cancel order", "update address") == 0.0


def test_cohen_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


def test_conditions_keep_length_and_snr():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(24000)
    others = [rng.standard_normal(20000) for _ in range(3)]
    noisy = mix_at_snr(sig, make_babble(len(sig), others, rng), -5.0)
    assert len(noisy) == len(sig)
    assert abs(10 * np.log10(np.mean(sig**2) / np.mean((noisy - sig) ** 2)) + 5.0) < 0.1
    assert len(reverb(sig, 24000, 0.8, rng)) == len(sig)
    narrow, sr = telephone(sig, 24000)
    assert sr == 8000 and len(narrow) == 8000


def test_action_uses_route_and_negative_sentiment():
    routes = {"cancel order": "requests_cancellations"}
    assert action_of(_label("cancel order", sentiment="negative"), routes) == ("requests_cancellations", True)
    assert action_of(_label("unknown phrase"), routes) is None
