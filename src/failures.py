"""Append API failures to results/failures.csv."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

FAILURE_FIELDS = [
    "timestamp_utc",
    "stage",
    "utterance_id",
    "condition",
    "http_status",
    "error",
    "request_id",
]


def log_failure(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FAILURE_FIELDS})


def failure_row(
    stage: str,
    utterance_id: str,
    condition: str = "",
    http_status: str = "",
    error: str = "",
    request_id: str = "",
) -> dict[str, str]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "utterance_id": utterance_id,
        "condition": condition,
        "http_status": http_status,
        "error": error,
        "request_id": request_id,
    }
