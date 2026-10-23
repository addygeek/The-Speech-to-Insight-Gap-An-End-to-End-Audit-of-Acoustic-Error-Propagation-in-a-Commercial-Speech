"""Second critical-span check that keeps times and currency a reader would count.

The primary metric strips punctuation and then asks for a substring. That turns
``$250.00`` into ``25000`` and ``9:30 AM`` into ``930 am``. This module asks
whether any spoken form of the same span is present. It does not replace the
primary metric.
"""

from __future__ import annotations

import re

from src.text_normalize import normalize_text

_TIME = re.compile(r"(\d{1,2})\s*:\s*(\d{2})\s*([ap]\.?m\.?)?", re.IGNORECASE)
_MONEY = re.compile(r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|\$?\s*\d+(?:\.\d+)?")


def _digit_forms(raw: str) -> set[str]:
    forms: set[str] = set()
    digits = re.sub(r"\D", "", raw)
    if digits:
        forms.add(digits)
        stripped = digits.lstrip("0")
        forms.add(stripped or "0")
    return forms


def span_forms(span: object) -> set[str]:
    raw = "" if span is None else str(span).strip()
    forms = {normalize_text(raw)}
    forms |= _digit_forms(raw)
    for match in _MONEY.finditer(raw):
        token = match.group(0)
        number = token.replace("$", "").replace(",", "").strip()
        if not number:
            continue
        forms.add(normalize_text(number))
        forms |= _digit_forms(number)
        if "." in number:
            whole, _frac = number.split(".", 1)
            forms.add(whole.lstrip("0") or "0")
    time_match = _TIME.search(raw)
    if time_match:
        hour, minute, meridiem = time_match.group(1), time_match.group(2), time_match.group(3) or ""
        meridiem = re.sub(r"[^a-z]", "", meridiem.lower())
        hours = {str(int(hour)), hour.zfill(2)}
        for hour_form in hours:
            forms.add(f"{hour_form}{minute}{meridiem}")
            forms.add(f"{hour_form}{minute} {meridiem}".strip())
            forms.add(f"{hour_form} {minute} {meridiem}".strip())
    return {form for form in forms if form}


def _contains(text_norm: str, form: str) -> bool:
    compact = text_norm.replace(" ", "")
    target = form.replace(" ", "")
    if target.isdigit():
        return re.search(rf"(?<!\d){re.escape(target)}(?!\d)", compact) is not None
    return form in text_norm or target in compact


def span_present_relaxed(span: object, text: str) -> bool:
    norm = normalize_text(text)
    return any(_contains(norm, form) for form in span_forms(span))


def relaxed_span_error_rate(
    reference: str,
    hypothesis: str,
    critical_spans: list[object],
) -> tuple[float, int, int]:
    if not critical_spans:
        return 0.0, 0, 0
    errors = 0
    for span in critical_spans:
        if span_present_relaxed(span, reference) != span_present_relaxed(span, hypothesis):
            errors += 1
    total = len(critical_spans)
    return errors / total, errors, total
