"""Worker cap and resume checks for parallel Deepgram calls."""

from pathlib import Path

from src.cache import complete_response, save_json
from src.config_loader import load_config
from src.deepgram_read import _read_params
from src.request_pool import WorkerGate


def test_read_params_include_language() -> None:
    params = _read_params(load_config())
    assert params["language"] == "en"
    assert params["sentiment"] == "true"


def test_gate_shrinks_after_repeated_429() -> None:
    gate = WorkerGate(start=6, reduced=3, shrink_after=3)
    assert gate.limit == 6
    gate.note_status(200)
    gate.note_status(429)
    gate.note_status(429)
    assert gate.limit == 6
    gate.note_status(429)
    assert gate.limit == 3


def test_complete_response_skips_only_intact_json(tmp_path: Path) -> None:
    ready = tmp_path / "utt_example_clean.json"
    save_json(
        ready,
        {
            "meta": {"utterance_id": "utt_example", "condition": "clean"},
            "request_id": "dg-example",
            "response": {"results": {}},
            "timestamp_utc": "2026-10-23T00:00:00+00:00",
        },
    )
    assert complete_response(ready, live=True)

    broken = tmp_path / "utt_example_mild.json"
    broken.write_text("{not json", encoding="utf-8")
    assert not complete_response(broken, live=True)

    dry = tmp_path / "utt_example_strong.json"
    save_json(dry, {"dry_run": True, "response": {}, "timestamp_utc": "2026-10-23T00:00:00+00:00"})
    assert not complete_response(dry, live=True)
    assert not dry.exists()
