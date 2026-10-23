"""Pilot acceptance checks before full run."""

from __future__ import annotations

import json

from src.config_loader import ExperimentConfig
from src.logging_utils import setup_logging
from src.manifest import Manifest, Stage

logger = setup_logging()


def run_validate_results(cfg: ExperimentConfig, force: bool = False) -> bool:
    manifest = Manifest(cfg.manifest_csv)
    errors: list[str] = []

    if manifest.count_success(Stage.LISTEN) < 1:
        errors.append("no successful listen requests")
    if manifest.count_success(Stage.READ) < 1:
        errors.append("no successful read requests")
    if not (cfg.results_dir / "intelligence.csv").is_file():
        errors.append("missing results/intelligence.csv")
    if not (cfg.results_dir / "metrics.csv").is_file():
        errors.append("missing results/metrics.csv (run stats first)")

    ok = len(errors) == 0 or force
    report = {"valid": ok, "errors": errors, "forced": force}
    summary_path = cfg.results_dir / "summary.json"
    summary: dict = {}
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
    summary["pilot_validation"] = report
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if ok:
        logger.info("Pilot validation passed")
    else:
        logger.error("Pilot validation failed: %s", errors)
    return ok
