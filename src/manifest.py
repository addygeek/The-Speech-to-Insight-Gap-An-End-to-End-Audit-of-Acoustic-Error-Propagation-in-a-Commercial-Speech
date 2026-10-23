"""Resumable experiment manifest (utterance × condition × stage)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

MANIFEST_FIELDS = [
    "utterance_id",
    "stage",
    "condition",
    "status",
    "timestamp_utc",
    "request_id",
    "error",
    "meta",
]


class Stage(str, Enum):
    DATASET = "dataset"
    TTS = "tts"
    PERTURB = "perturb"
    VALIDATE_AUDIO = "validate_audio"
    LISTEN = "listen"
    READ = "read"
    PARSE = "parse"
    METRICS = "metrics"
    STATS = "stats"


class Status(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ManifestRow:
    utterance_id: str
    stage: str
    condition: str
    status: str
    timestamp_utc: str
    request_id: str = ""
    error: str = ""
    meta: str = ""


def _stage_key(stage: Stage | str) -> str:
    if isinstance(stage, Stage):
        return stage.value
    return str(stage)


def _status_key(status: Status | str) -> str:
    if isinstance(status, Status):
        return status.value
    return str(status)


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: dict[tuple[str, str, str], ManifestRow] = {}
        if path.is_file():
            self._load()

    def _key(self, utterance_id: str, stage: Stage | str, condition: str) -> tuple[str, str, str]:
        return utterance_id, _stage_key(stage), condition

    def _load(self) -> None:
        with self.path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mr = ManifestRow(
                    utterance_id=row["utterance_id"],
                    stage=row["stage"],
                    condition=row.get("condition", ""),
                    status=row["status"],
                    timestamp_utc=row.get("timestamp_utc", ""),
                    request_id=row.get("request_id", ""),
                    error=row.get("error", ""),
                    meta=row.get("meta", ""),
                )
                self._rows[self._key(mr.utterance_id, mr.stage, mr.condition)] = mr

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for mr in sorted(
                self._rows.values(),
                key=lambda r: (r.utterance_id, r.stage, r.condition),
            ):
                writer.writerow(
                    {
                        "utterance_id": mr.utterance_id,
                        "stage": mr.stage,
                        "condition": mr.condition,
                        "status": mr.status,
                        "timestamp_utc": mr.timestamp_utc,
                        "request_id": mr.request_id,
                        "error": mr.error,
                        "meta": mr.meta,
                    }
                )

    def get(self, utterance_id: str, stage: Stage | str, condition: str = "") -> ManifestRow | None:
        return self._rows.get(self._key(utterance_id, stage, condition))

    def is_success(self, utterance_id: str, stage: Stage | str, condition: str = "") -> bool:
        row = self.get(utterance_id, stage, condition)
        return row is not None and row.status == Status.SUCCESS.value

    def should_skip(self, resume: bool, utterance_id: str, stage: Stage | str, condition: str = "") -> bool:
        return resume and self.is_success(utterance_id, stage, condition)

    def record(
        self,
        utterance_id: str,
        stage: Stage | str,
        status: Status | str,
        condition: str = "",
        request_id: str = "",
        error: str = "",
        meta: str = "",
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        stage_s = _stage_key(stage)
        self._rows[self._key(utterance_id, stage_s, condition)] = ManifestRow(
            utterance_id=utterance_id,
            stage=stage_s,
            condition=condition,
            status=_status_key(status),
            timestamp_utc=ts,
            request_id=request_id,
            error=error,
            meta=meta,
        )

    def iter_stage(self, stage: Stage | str) -> Iterable[ManifestRow]:
        stage_s = _stage_key(stage)
        for row in self._rows.values():
            if row.stage == stage_s:
                yield row

    def count_success(self, stage: Stage | str) -> int:
        return sum(1 for r in self.iter_stage(stage) if r.status == Status.SUCCESS.value)
