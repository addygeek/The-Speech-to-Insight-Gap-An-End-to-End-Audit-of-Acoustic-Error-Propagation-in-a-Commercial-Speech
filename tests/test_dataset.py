from src.config_loader import load_config
from src.generate_dataset import generate_records
from src.validate_dataset import validate_records


def test_generate_records_unique():
    cfg = load_config()
    cfg.raw.setdefault("dataset", {})["generation_mode"] = "templates"
    records = generate_records(cfg, count=10, split="pilot")
    assert len(records) == 10
    report = validate_records(records)
    assert report["valid"]


def test_critical_spans_present():
    cfg = load_config()
    cfg.raw.setdefault("dataset", {})["generation_mode"] = "templates"
    records = generate_records(cfg, count=5, split="pilot")
    for rec in records:
        assert rec["critical_spans"]
        assert len(rec["critical_spans"]) == len(rec["critical_span_types"])
        assert 25 <= len(rec["reference_text"].split()) <= 44
