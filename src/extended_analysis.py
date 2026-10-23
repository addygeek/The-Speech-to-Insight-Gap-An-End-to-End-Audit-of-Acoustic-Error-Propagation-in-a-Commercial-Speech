"""Figures and secondary measurements computed from the saved result tables."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import wilcoxon

from src.calculate_critical_errors import span_present
from src.config_loader import ExperimentConfig
from src.generate_dataset import load_references
from src.logging_utils import setup_logging
from src.span_rescore import relaxed_span_error_rate, span_present_relaxed
from src.text_normalize import normalize_text

logger = setup_logging()

CONDITION_ORDER = ("clean", "mild", "strong", "harsh")
CONDITION_LABEL = {
    "clean": "Clean",
    "mild": "15 dB",
    "strong": "8 dB",
    "harsh": "0 dB",
}
CONDITION_COLOR = {
    "clean": "#1B4F72",
    "mild": "#148F77",
    "strong": "#B9770E",
    "harsh": "#922B21",
}
SCENARIO_LABEL = {
    "appointment_scheduling": "Appointment",
    "technical_support": "Technical",
    "customer_support": "Support",
    "banking_payments": "Banking",
    "requests_cancellations": "Cancellation",
    "delivery_tracking": "Delivery",
    "travel_booking": "Travel",
    "complaints_compliments": "Complaints",
}


def _conditions(df: pd.DataFrame) -> list[str]:
    present = set(df["acoustic_condition"].astype(str))
    return [name for name in CONDITION_ORDER if name in present]


def _style(root: Path) -> None:
    regular = root / "paper" / "fonts" / "OpenSans-Regular.ttf"
    bold = root / "paper" / "fonts" / "OpenSans-Bold.ttf"
    if regular.is_file():
        font_manager.fontManager.addfont(str(regular))
        font_manager.fontManager.addfont(str(bold))
        plt.rcParams["font.family"] = "Open Sans"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _bootstrap_mean(values: np.ndarray, n_samples: int, seed: int) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _token_jaccard(left: str, right: str) -> float:
    a = set(normalize_text(left).split())
    b = set(normalize_text(right).split())
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _tex(text: str) -> str:
    escaped = str(text)
    for char, repl in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        escaped = escaped.replace(char, repl)
    return escaped


def _short(text: str, words: int = 16) -> str:
    parts = re.sub(r"\s+", " ", str(text)).strip().split()
    if len(parts) <= words:
        return " ".join(parts)
    return " ".join(parts[:words]) + " ..."


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _paired(frame: pd.DataFrame, left: str, right: str) -> dict[str, float]:
    wide = frame.pivot(index="utterance_id", columns="acoustic_condition", values="wer")
    if left not in wide.columns or right not in wide.columns:
        return {}
    pair = wide[[left, right]].dropna()
    delta = pair[right].to_numpy() - pair[left].to_numpy()
    mean, lo, hi = _bootstrap_mean(delta, 2000, 42)
    nonzero = delta[delta != 0]
    wilcox = {"statistic": float("nan"), "p_value": float("nan")}
    if len(nonzero) >= 10 and np.unique(np.sign(nonzero)).size > 1:
        stat, pval = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
        wilcox = {"statistic": float(stat), "p_value": float(pval)}
    return {
        "n": int(len(pair)),
        "mean_delta": mean,
        "ci95": [lo, hi],
        "fraction_equal": float(np.mean(delta == 0)),
        "fraction_higher": float(np.mean(delta > 0)),
        "fraction_lower": float(np.mean(delta < 0)),
        "wilcoxon_nonzero_n": int(len(nonzero)),
        "wilcoxon": wilcox,
    }


def _intent_overlap(intel: pd.DataFrame, conditions: list[str]) -> dict[str, dict[str, object]]:
    ref = intel[intel["text_source"] == "reference"][
        ["utterance_id", "intent_normalized"]
    ].drop_duplicates("utterance_id")
    out: dict[str, dict[str, object]] = {}
    for cond in conditions:
        rows = intel[intel["text_source"] == f"transcript_{cond}"][
            ["utterance_id", "intent_normalized"]
        ]
        merged = ref.merge(rows, on="utterance_id", suffixes=("_ref", "_hyp"))
        if merged.empty:
            continue
        scores = [
            _token_jaccard(str(a), str(b))
            for a, b in zip(merged["intent_normalized_ref"], merged["intent_normalized_hyp"], strict=True)
        ]
        exact = [
            normalize_text(str(a)) != normalize_text(str(b))
            for a, b in zip(merged["intent_normalized_ref"], merged["intent_normalized_hyp"], strict=True)
        ]
        disagree = [score for score, flag in zip(scores, exact, strict=True) if flag]
        out[cond] = {
            "n": int(len(scores)),
            "exact_disagree_rate": float(np.mean(exact)) if exact else float("nan"),
            "mean_token_jaccard": float(np.mean(scores)) if scores else float("nan"),
            "disagree_n": int(len(disagree)),
            "disagree_mean_token_jaccard": float(np.mean(disagree)) if disagree else float("nan"),
            "disagree_zero_overlap": float(np.mean([score == 0 for score in disagree])) if disagree else float("nan"),
            "disagree_overlap_at_least_half": (
                float(np.mean([score >= 0.5 for score in disagree])) if disagree else float("nan")
            ),
            "scores_on_disagreement": [round(float(score), 4) for score in disagree],
        }
    return out


def _span_tables(cfg: ExperimentConfig, transcripts: pd.DataFrame) -> dict[str, object]:
    refs = {row["utterance_id"]: row for row in load_references(cfg.references_jsonl)}
    by_condition: dict[str, dict[str, float]] = {}
    by_type: dict[str, dict[str, dict[str, int]]] = {}
    for cond, group in transcripts.groupby("acoustic_condition"):
        rates: list[float] = []
        type_counts: dict[str, dict[str, int]] = {}
        for _, row in group.iterrows():
            rec = refs.get(row["utterance_id"])
            if not rec:
                continue
            spans = list(rec.get("critical_spans") or [])
            kinds = list(rec.get("critical_span_types") or [])
            rate, _, _ = relaxed_span_error_rate(rec["reference_text"], str(row["transcript"]), spans)
            rates.append(rate)
            for span, kind in zip(spans, kinds, strict=False):
                bucket = type_counts.setdefault(str(kind), {"n": 0, "raw_missing": 0, "relaxed_missing": 0})
                bucket["n"] += 1
                if span_present_relaxed(span, rec["reference_text"]) != span_present_relaxed(span, str(row["transcript"])):
                    bucket["relaxed_missing"] += 1
                if span_present(span, rec["reference_text"], cfg) != span_present(span, str(row["transcript"]), cfg):
                    bucket["raw_missing"] += 1
        by_condition[str(cond)] = {
            "relaxed_mean": float(np.mean(rates)) if rates else float("nan"),
            "n": int(len(rates)),
        }
        by_type[str(cond)] = type_counts
    return {"by_condition": by_condition, "by_type": by_type}


def _examples(transcripts: pd.DataFrame, metrics: pd.DataFrame, focus: str) -> list[dict[str, object]]:
    if focus not in set(transcripts["acoustic_condition"]):
        return []
    wide = transcripts.pivot(index="utterance_id", columns="acoustic_condition", values="transcript")
    if "clean" not in wide.columns or focus not in wide.columns:
        return []
    changed = wide[wide["clean"].fillna("") != wide[focus].fillna("")].copy()
    wer = metrics.pivot(index="utterance_id", columns="acoustic_condition", values="wer")
    changed["delta"] = wer[focus] - wer["clean"]
    changed = changed.sort_values("delta", ascending=False).head(6)
    scenario = metrics.drop_duplicates("utterance_id").set_index("utterance_id")["scenario"]
    rows = []
    for uid, row in changed.iterrows():
        rows.append(
            {
                "utterance_id": str(uid),
                "scenario": str(scenario.get(uid, "")),
                "wer_delta": None if pd.isna(row["delta"]) else float(row["delta"]),
                "clean": _short(row["clean"]),
                "focus": _short(row[focus]),
            }
        )
    return rows


def _write_scenario_table(path: Path, metrics: pd.DataFrame) -> None:
    conditions = _conditions(metrics)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Word error, intent disagreement, and critical-span rate by scenario and condition.}",
        "\\label{tab:scenario}",
        "\\small",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Scenario & Condition & WER & Intent & Spans \\\\",
        "\\midrule",
    ]
    for scenario, group in metrics.groupby("scenario"):
        n = int(group["utterance_id"].nunique())
        label = SCENARIO_LABEL.get(str(scenario), str(scenario))
        first = True
        for cond in conditions:
            sub = group[group["acoustic_condition"] == cond]
            if sub.empty:
                continue
            wer = f"{sub['wer'].mean():.3f}"
            intent = f"{sub['intent_disagreement_ref_vs_transcript'].mean():.3f}"
            spans = f"{sub['critical_span_error_rate'].mean():.3f}"
            name = f"{_tex(label)} ($n{{=}}{n}$)" if first else ""
            lines.append(f"{name} & {CONDITION_LABEL[cond]} & {wer} & {intent} & {spans} \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_rates(path: Path, summary: dict[str, dict[str, tuple[float, float, float]]], conditions: list[str]) -> None:
    panels = [
        ("Acoustic rates", [("wer", "WER"), ("cer", "CER"), ("relaxed_span", "Span, rescored")]),
        ("Label rates", [("sentiment", "Sentiment"), ("intent", "Intent"), ("topic", "Topic set")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, (title, series) in zip(axes, panels, strict=True):
        for index, (key, label) in enumerate(series):
            for offset, cond in enumerate(conditions):
                mean, lo, hi = summary[cond][key]
                y = index + (offset - (len(conditions) - 1) / 2) * 0.16
                xerr = None if np.isnan(lo) else [[mean - lo], [hi - mean]]
                ax.errorbar(
                    mean,
                    y,
                    xerr=xerr,
                    fmt="o",
                    color=CONDITION_COLOR[cond],
                    markersize=4.5,
                    capsize=2,
                    label=CONDITION_LABEL[cond] if index == 0 else None,
                )
        ax.set_yticks(range(len(series)))
        ax.set_yticklabels([label for _, label in series])
        ax.set_xlabel("Rate")
        ax.set_title(title)
        ax.set_xlim(left=-0.02)
    axes[1].legend(frameon=False, loc="lower right")
    _save(fig, path)


def _plot_ecdf(path: Path, metrics: pd.DataFrame, conditions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for cond in conditions:
        values = np.sort(metrics.loc[metrics["acoustic_condition"] == cond, "wer"].to_numpy())
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", color=CONDITION_COLOR[cond], label=CONDITION_LABEL[cond])
    ax.set_xlabel("Word error rate")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0, max(0.2, float(metrics["wer"].quantile(0.98))))
    ax.legend(frameon=False)
    _save(fig, path)


def _plot_scenario(path: Path, metrics: pd.DataFrame, conditions: list[str]) -> None:
    scenarios = [name for name in SCENARIO_LABEL if name in set(metrics["scenario"])]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    y = np.arange(len(scenarios))
    width = 0.8 / max(len(conditions), 1)
    for index, cond in enumerate(conditions):
        means = []
        for scenario in scenarios:
            sub = metrics[(metrics["scenario"] == scenario) & (metrics["acoustic_condition"] == cond)]
            means.append(float(sub["wer"].mean()) if len(sub) else 0.0)
        ax.barh(
            y + (index - (len(conditions) - 1) / 2) * width,
            means,
            height=width * 0.92,
            color=CONDITION_COLOR[cond],
            label=CONDITION_LABEL[cond],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIO_LABEL[name] for name in scenarios])
    ax.set_xlabel("Mean word error rate")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.invert_yaxis()
    _save(fig, path)


def _plot_errors(path: Path, metrics: pd.DataFrame, conditions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.arange(len(conditions))
    subs, dels, ins = [], [], []
    for cond in conditions:
        sub = metrics[metrics["acoustic_condition"] == cond]
        subs.append(float(sub["substitutions_rate"].mean()))
        dels.append(float(sub["deletions_rate"].mean()))
        ins.append(float(sub["insertions_rate"].mean()))
    ax.bar(x, subs, color="#1B4F72", label="Substitution")
    bottom = np.array(subs)
    ax.bar(x, dels, bottom=bottom, color="#B9770E", label="Deletion")
    ax.bar(x, ins, bottom=bottom + np.array(dels), color="#148F77", label="Insertion")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABEL[name] for name in conditions])
    ax.set_ylabel("Mean rate")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, path)


def _plot_spans(path: Path, by_type: dict[str, dict[str, int]], condition: str) -> None:
    kinds = [name for name, bucket in by_type.items() if bucket["n"]]
    kinds.sort(key=lambda name: by_type[name]["raw_missing"] / by_type[name]["n"], reverse=True)
    raw = [by_type[name]["raw_missing"] / by_type[name]["n"] for name in kinds]
    relaxed = [by_type[name]["relaxed_missing"] / by_type[name]["n"] for name in kinds]
    y = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.barh(y - 0.18, raw, height=0.36, color="#7F8C8D", label="Punctuation stripped")
    ax.barh(y + 0.18, relaxed, height=0.36, color="#1B4F72", label="Time and currency forms")
    ax.set_yticks(y)
    ax.set_yticklabels(kinds)
    ax.set_xlabel(f"Flagged fraction ({CONDITION_LABEL.get(condition, condition)})")
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    _save(fig, path)


def _plot_intent(path: Path, overlap: dict[str, dict[str, object]], conditions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bins = np.linspace(0, 1, 11)
    for cond in conditions:
        scores = overlap.get(cond, {}).get("scores_on_disagreement") or []
        if not scores:
            continue
        ax.hist(
            scores,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=CONDITION_COLOR[cond],
            label=CONDITION_LABEL[cond],
        )
    ax.set_xlabel("Token Jaccard of disagreeing intent strings")
    ax.set_ylabel("Disagreements")
    ax.legend(frameon=False)
    _save(fig, path)


def _plot_jaccard(path: Path, metrics: pd.DataFrame, conditions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    data = [
        metrics.loc[metrics["acoustic_condition"] == cond, "topic_jaccard_ref_vs_transcript"].dropna().to_numpy()
        for cond in conditions
    ]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, cond in zip(parts["bodies"], conditions, strict=True):
        body.set_facecolor(CONDITION_COLOR[cond])
        body.set_alpha(0.75)
        body.set_edgecolor("none")
    parts["cmedians"].set_color("#111318")
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels([CONDITION_LABEL[name] for name in conditions])
    ax.set_ylabel("Topic Jaccard, reference vs transcript")
    ax.set_ylim(-0.05, 1.05)
    _save(fig, path)


def _plot_sentiment(path: Path, intel: pd.DataFrame, conditions: list[str]) -> None:
    labels = ["positive", "neutral", "negative"]
    colors = {"positive": "#148F77", "neutral": "#7F8C8D", "negative": "#922B21"}
    series: list[tuple[str, pd.Series]] = []
    ref = intel[intel["text_source"] == "reference"]
    series.append(("Reference", ref["sentiment_label"].astype(str).str.lower()))
    audio = intel[intel["pipeline_path"] == "audio_native"]
    for cond in conditions:
        series.append(
            (
                CONDITION_LABEL[cond],
                audio.loc[audio["acoustic_condition"] == cond, "sentiment_label"].astype(str).str.lower(),
            )
        )
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.arange(len(series))
    bottom = np.zeros(len(series))
    for label in labels:
        heights = [float((values == label).mean()) if len(values) else 0.0 for _, values in series]
        ax.bar(x, heights, bottom=bottom, color=colors[label], label=label.capitalize())
        bottom = bottom + np.array(heights)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in series])
    ax.set_ylabel("Share of items")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, path)


def _plot_delta(path: Path, metrics: pd.DataFrame, conditions: list[str]) -> None:
    wide = metrics.pivot(index="utterance_id", columns="acoustic_condition", values="wer")
    noisy = [cond for cond in conditions if cond != "clean" and cond in wide.columns and "clean" in wide.columns]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.arange(len(noisy))
    lower, equal, higher = [], [], []
    for cond in noisy:
        delta = (wide[cond] - wide["clean"]).dropna()
        lower.append(float((delta < 0).mean()))
        equal.append(float((delta == 0).mean()))
        higher.append(float((delta > 0).mean()))
    ax.bar(x, equal, color="#D5D8DC", label="Same WER")
    ax.bar(x, higher, bottom=equal, color="#922B21", label="Higher than clean")
    ax.bar(x, lower, bottom=np.array(equal) + np.array(higher), color="#1B4F72", label="Lower than clean")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABEL[name] for name in noisy])
    ax.set_ylabel("Fraction of utterances")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, path)


def run_extended(cfg: ExperimentConfig) -> dict[str, object]:
    root = Path.cwd()
    _style(root)
    metrics = pd.read_csv(cfg.results_dir / "metrics.csv")
    intel = pd.read_csv(cfg.results_dir / "intelligence.csv")
    transcripts = pd.read_csv(cfg.results_dir / "transcripts.csv")
    conditions = _conditions(metrics)
    spans = _span_tables(cfg, transcripts)
    overlap = _intent_overlap(intel, conditions)
    paired = {cond: _paired(metrics, "clean", cond) for cond in conditions if cond != "clean"}
    rate_summary: dict[str, dict[str, tuple[float, float, float]]] = {}
    for cond in conditions:
        group = metrics[metrics["acoustic_condition"] == cond]
        relaxed = float(spans["by_condition"].get(cond, {}).get("relaxed_mean", float("nan")))
        rate_summary[cond] = {
            "wer": _bootstrap_mean(group["wer"].to_numpy(dtype=float), 2000, 42),
            "cer": _bootstrap_mean(group["cer"].to_numpy(dtype=float), 2000, 42),
            "relaxed_span": (relaxed, float("nan"), float("nan")),
            "sentiment": _bootstrap_mean(
                group["sentiment_disagreement_ref_vs_transcript"].astype(float).to_numpy(), 2000, 42
            ),
            "intent": _bootstrap_mean(
                group["intent_disagreement_ref_vs_transcript"].astype(float).to_numpy(), 2000, 42
            ),
            "topic": _bootstrap_mean(
                group["topic_set_changed_ref_vs_transcript"].astype(float).to_numpy(), 2000, 42
            ),
        }
    focus = "harsh" if "harsh" in conditions else "strong"
    payload: dict[str, object] = {
        "conditions": conditions,
        "rates": {
            cond: {key: {"mean": vals[0], "ci95": [vals[1], vals[2]]} for key, vals in block.items()}
            for cond, block in rate_summary.items()
        },
        "paired_wer_vs_clean": paired,
        "span_relaxed": spans["by_condition"],
        "span_by_type": spans["by_type"],
        "intent_overlap": {
            cond: {key: value for key, value in block.items() if key != "scores_on_disagreement"}
            for cond, block in overlap.items()
        },
        "examples": _examples(transcripts, metrics, focus),
        "focus_condition": focus,
    }
    (cfg.results_dir / "extended_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    figures = cfg.figures_dir
    _plot_rates(figures / "fig4_rates.pdf", rate_summary, conditions)
    _plot_ecdf(figures / "fig5_wer_ecdf.pdf", metrics, conditions)
    _plot_scenario(figures / "fig6_scenario_wer.pdf", metrics, conditions)
    _plot_errors(figures / "fig7_error_composition.pdf", metrics, conditions)
    _plot_spans(figures / "fig8_span_types.pdf", spans["by_type"][focus], focus)
    _plot_intent(figures / "fig9_intent_overlap.pdf", overlap, conditions)
    _plot_jaccard(figures / "fig10_topic_jaccard.pdf", metrics, conditions)
    _plot_sentiment(figures / "fig11_sentiment.pdf", intel, conditions)
    _plot_delta(figures / "fig12_paired_delta.pdf", metrics, conditions)
    _write_scenario_table(cfg.tables_dir / "table_scenario.tex", metrics)
    logger.info("Wrote extended summary and figures for %s", ", ".join(conditions))
    return payload
