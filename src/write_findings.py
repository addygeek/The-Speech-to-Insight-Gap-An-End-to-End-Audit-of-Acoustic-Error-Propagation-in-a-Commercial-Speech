"""Write the results narrative from measured tables only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config_loader import ExperimentConfig
from src.logging_utils import setup_logging

logger = setup_logging()


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return f"{number:.{digits}f}"


def _load_summary(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def write_findings(cfg: ExperimentConfig) -> Path:
    metrics_path = cfg.results_dir / "metrics.csv"
    summary = _load_summary(cfg.results_dir / "summary.json")
    cost = _load_summary(cfg.results_dir / "cost_report.json")
    stats = summary.get("statistics") or {}
    lines = [
        "# Measured findings",
        "",
        "Every number below is computed from results/metrics.csv and results/summary.json.",
        "Missing files are reported as missing. No value is filled in by hand.",
        "",
    ]
    if not metrics_path.is_file():
        lines.append("metrics.csv is not present yet, so there are no findings to report.")
        out = cfg.results_dir / "findings.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    frame = pd.read_csv(metrics_path)
    n_utt = int(frame["utterance_id"].nunique()) if "utterance_id" in frame else 0
    lines.append(f"Rows: {len(frame)}. Unique utterances: {n_utt}.")
    lines.append(f"Exploratory flag: {stats.get('exploratory', 'n/a')}.")
    lines.append("")
    lines.append("## By acoustic condition")
    lines.append("")
    by_cond = stats.get("by_condition") or {}
    for cond, block in sorted(by_cond.items()):
        ci = block.get("wer_ci95") or [None, None]
        lines.append(
            f"- {cond}: n={block.get('n')}, mean WER={_fmt(block.get('wer_mean'))} "
            f"(95% bootstrap interval {_fmt(ci[0])} to {_fmt(ci[1])}), "
            f"mean CER={_fmt(block.get('cer_mean'))}, "
            f"critical-span error rate={_fmt(block.get('critical_span_error_rate'))}, "
            f"sentiment disagreement={_fmt(block.get('sentiment_disagreement_ref_vs_transcript'))}, "
            f"intent disagreement={_fmt(block.get('intent_disagreement_ref_vs_transcript'))}, "
            f"topic-set change={_fmt(block.get('topic_set_changed_ref_vs_transcript'))}, "
            f"topic Jaccard={_fmt(block.get('topic_jaccard_mean'))}."
        )
    lines.append("")
    lines.append("## Association")
    lines.append("")
    assoc = (stats.get("associations") or {}).get("wer_vs_sentiment_disagreement")
    if assoc:
        lines.append(
            f"Spearman correlation of utterance WER with sentiment disagreement: "
            f"rho={_fmt(assoc.get('spearman_rho'))}, p={_fmt(assoc.get('p_value'), 4)}, n={assoc.get('n')}."
        )
    else:
        lines.append("No rank correlation was computed. The sample was constant or smaller than the cutoff.")
    lines.append("")
    lines.append("## Cost record")
    lines.append("")
    if cost:
        lines.append(f"Recorded estimate total USD: {_fmt(cost.get('spent_usd'), 2)}.")
        pre = cost.get("preflight") or {}
        est = pre.get("estimated_usd") or {}
        if est:
            lines.append(
                f"Preflight split: speech synthesis {_fmt(est.get('tts'), 2)}, "
                f"transcription {_fmt(est.get('stt'), 2)}, text analysis {_fmt(est.get('read'), 2)}."
            )
    else:
        lines.append("cost_report.json is not present.")
    lines.append("")
    lines.append("## What this does not show")
    lines.append("")
    lines.append(
        "These figures describe this synthetic English set and this vendor configuration. "
        "Disagreement with the reference-text analysis is not a human ground-truth error. "
        "A higher word error rate is not, by itself, a measure of semantic damage."
    )
    out = cfg.results_dir / "findings.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_tex(cfg, by_cond, assoc, n_utt, len(frame))
    logger.info("Wrote %s", out)
    return out


def _write_tex(cfg: ExperimentConfig, by_cond: dict, assoc: dict | None, n_utt: int, n_rows: int) -> None:
    parts = [
        "\\paragraph{Measured summary.}",
        f"The metrics table contains {n_rows} rows from {n_utt} utterances.",
    ]
    for cond, block in sorted(by_cond.items()):
        parts.append(
            f"Under {cond}, mean WER is {_fmt(block.get('wer_mean'))} "
            f"(n={block.get('n')}), and the sentiment-disagreement rate versus the reference text is "
            f"{_fmt(block.get('sentiment_disagreement_ref_vs_transcript'))}."
        )
    if assoc:
        parts.append(
            f"The Spearman correlation between WER and sentiment disagreement is {_fmt(assoc.get('spearman_rho'))}."
        )
    path = cfg.results_dir / "results_section.tex"
    path.write_text(" ".join(parts) + "\n", encoding="utf-8")
