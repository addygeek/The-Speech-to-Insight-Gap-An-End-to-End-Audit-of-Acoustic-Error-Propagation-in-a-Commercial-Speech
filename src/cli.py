"""CLI entrypoint for the Speech-to-Insight Gap pipeline."""

from __future__ import annotations

import argparse
import sys

from src.budget import BudgetTracker
from src.budget_estimator import run_preflight_cost
from src.cache import ensure_dirs
from src.config_loader import load_config
from src.deepgram_listen import run_listen
from src.deepgram_read import run_read
from src.deepgram_tts import run_tts
from src.generate_dataset import run_generate_dataset
from src.generate_paper_assets import run_paper_assets
from src.logging_utils import setup_logging
from src.manifest import Manifest, Stage, Status
from src.perturb_audio import run_perturb
from src.statistical_analysis import run_metrics, run_statistics
from src.validate_audio import run_validate_audio
from src.validate_dataset import run_validate_dataset
from src.validate_results import run_validate_results

logger = setup_logging()


def _apply_overrides(cfg, args: argparse.Namespace):
    if args.dry_run is not None:
        cfg.raw.setdefault("run", {})["dry_run"] = args.dry_run
    if args.no_resume:
        cfg.raw.setdefault("run", {})["resume"] = False
    return cfg


def cmd_run_pipeline(cfg, args: argparse.Namespace, split: str, count: int) -> None:
    ensure_dirs(cfg)
    mf = Manifest(cfg.manifest_csv)
    budget = BudgetTracker(cfg)
    run_generate_dataset(cfg, count=count, split=split)
    run_validate_dataset(cfg)
    run_tts(cfg, mf, budget, split=split, max_requests=args.max_requests)
    run_perturb(cfg, mf, split=split)
    run_validate_audio(cfg, mf, split=split)
    run_listen(cfg, mf, budget, split=split, max_requests=args.max_requests)
    run_read(cfg, mf, budget, split=split, max_requests=args.max_requests)
    run_metrics(cfg)
    run_statistics(cfg)
    run_preflight_cost(cfg, target_utterances=cfg.full_utterances, split=split)


def _common_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default="config/experiment.yaml")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
    p.add_argument("--live", dest="dry_run", action="store_false")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--force", action="store_true")
    return p

