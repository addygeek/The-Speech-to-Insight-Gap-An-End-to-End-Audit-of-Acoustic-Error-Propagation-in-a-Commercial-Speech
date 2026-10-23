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
from src.generate_dataset import load_references, run_generate_dataset
from src.generate_paper_assets import run_paper_assets
from src.extended_analysis import run_extended
from src.logging_utils import setup_logging
from src.manifest import Manifest, Stage, Status
from src.perturb_audio import run_perturb
from src.statistical_analysis import run_metrics, run_statistics
from src.validate_audio import run_validate_audio
from src.validate_dataset import run_validate_dataset
from src.validate_results import run_validate_results
from src.write_findings import write_findings

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


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    p = argparse.ArgumentParser(prog="speech-to-insight", parents=[common])
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate-dataset", parents=[common])
    g.add_argument("--count", type=int, default=None)
    g.add_argument("--split", default="pilot")
    sub.add_parser("validate-dataset", parents=[common])
    sub.add_parser("tts", parents=[common])
    perturb_cmd = sub.add_parser("perturb", parents=[common])
    perturb_cmd.add_argument("--split", default=None)
    sub.add_parser("validate-audio", parents=[common])
    listen_cmd = sub.add_parser("listen", parents=[common])
    listen_cmd.add_argument("--split", default=None)
    read_cmd = sub.add_parser("read", parents=[common])
    read_cmd.add_argument("--split", default=None)
    sub.add_parser("metrics", parents=[common])
    sub.add_parser("stats", parents=[common])
    sub.add_parser("paper-assets", parents=[common])
    pf = sub.add_parser("preflight-cost", parents=[common])
    pf.add_argument("--target", type=int, default=None)
    pf.add_argument("--split", default="pilot")
    sub.add_parser("run-pilot", parents=[common])
    rf = sub.add_parser("run-full", parents=[common])
    rf.add_argument("--validate", action="store_true")
    sub.add_parser("validate-results", parents=[common])
    return p


def resolve_split(cfg, requested: str | None) -> str | None:
    if requested:
        return requested
    splits = {row.get("dataset_split") for row in load_references(cfg.references_jsonl)}
    splits.discard(None)
    splits.discard("")
    if len(splits) == 1:
        return next(iter(splits))
    if not splits:
        return "pilot"
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _apply_overrides(load_config(args.config), args)
    ensure_dirs(cfg)
    mf = Manifest(cfg.manifest_csv)
    budget = BudgetTracker(cfg)

    if args.command == "generate-dataset":
        split = args.split or "pilot"
        count = args.count or (cfg.pilot_utterances if split == "pilot" else cfg.full_utterances)
        run_generate_dataset(cfg, count=count, split=split)
        run_validate_dataset(cfg)
        mf.record("global", Stage.DATASET, Status.SUCCESS, meta=f"split={split},n={count}")
        mf.save()
    elif args.command == "validate-dataset":
        run_validate_dataset(cfg)
    elif args.command == "tts":
        run_tts(cfg, mf, budget, max_requests=args.max_requests)
    elif args.command == "perturb":
        run_perturb(cfg, mf, split=resolve_split(cfg, args.split))
    elif args.command == "validate-audio":
        run_validate_audio(cfg, mf)
    elif args.command == "listen":
        budget.load(cfg.results_dir / "cost_report.json")
        run_listen(cfg, mf, budget, split=resolve_split(cfg, args.split), max_requests=args.max_requests)
    elif args.command == "read":
        budget.load(cfg.results_dir / "cost_report.json")
        run_read(cfg, mf, budget, split=resolve_split(cfg, args.split), max_requests=args.max_requests)
    elif args.command == "metrics":
        run_metrics(cfg)
    elif args.command == "stats":
        run_statistics(cfg)
    elif args.command == "paper-assets":
        run_paper_assets(cfg)
        run_extended(cfg)
        write_findings(cfg)
    elif args.command == "preflight-cost":
        run_preflight_cost(cfg, target_utterances=args.target or cfg.full_utterances, split=args.split)
    elif args.command == "run-pilot":
        cmd_run_pipeline(cfg, args, split="pilot", count=cfg.pilot_utterances)
    elif args.command == "run-full":
        if args.validate and not run_validate_results(cfg, force=args.force):
            return 1
        cmd_run_pipeline(cfg, args, split="full", count=cfg.full_utterances)
    elif args.command == "validate-results":
        return 0 if run_validate_results(cfg, force=args.force) else 1
    else:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
