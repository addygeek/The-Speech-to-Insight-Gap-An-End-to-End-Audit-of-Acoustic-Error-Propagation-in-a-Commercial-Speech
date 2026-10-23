"""Spoken forms of times and currency count as present."""

from src.span_rescore import span_present_relaxed


def test_currency_cents_match_the_dollar_amount() -> None:
    assert span_present_relaxed("$250.00", "The fee is 250 dollars today.")


def test_clock_time_matches_a_zero_padded_transcript() -> None:
    assert span_present_relaxed("9:30 AM", "Your slot is 0930am tomorrow.")


def test_missing_amount_is_still_missing() -> None:
    assert not span_present_relaxed("$250.00", "The fee was waived.")
