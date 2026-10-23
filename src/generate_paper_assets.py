"""Generate LaTeX tables and figures from measured results only."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config_loader import ExperimentConfig
from src.logging_utils import setup_logging

logger = setup_logging()


def _stub_table(name: str, reason: str) -> str:
    return (
        f"% {name}: {reason}\n"
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{name} (data not available)}}\n"
        f"\\label{{tab:{name}}}\n\\end{{table}}\n"
    )


def write_tables(cfg: ExperimentConfig) -> None:
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.results_dir / "metrics.csv"

    if not metrics_path.is_file():
        for i, reason in enumerate(["missing metrics.csv"] * 3, start=1):
            (cfg.tables_dir / f"table{i}_{'dataset' if i==1 else 'wer' if i==2 else 'disagreement'}.tex").write_text(
                _stub_table(f"table{i}", reason), encoding="utf-8"
            )
        return

    df = pd.read_csv(metrics_path)
    n_utt = df["utterance_id"].nunique()
    conds = ", ".join(sorted(df["acoustic_condition"].unique()))
    table1 = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Dataset and acoustic conditions (measured).}\n\\label{tab:dataset}\n"
        "\\begin{tabular}{lr}\n\\toprule\nMetric & Value \\\\\n\\midrule\n"
        f"Utterances & {n_utt} \\\\\n"
        f"Metric rows & {len(df)} \\\\\n"
        f"Conditions & {conds} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    wer_by = df.groupby("acoustic_condition")["wer"].agg(["mean", "std", "count"]).reset_index()
    lines = ["\\begin{table}[t]\n\\centering\n\\caption{WER by acoustic condition.}\n\\label{tab:wer}\n"]
    lines.append("\\begin{tabular}{lrrr}\n\\toprule\nCondition & Mean WER & Std & N \\\\\n\\midrule\n")
    for _, row in wer_by.iterrows():
        lines.append(
            f"{row['acoustic_condition']} & {row['mean']:.3f} & {row['std']:.3f} & {int(row['count'])} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    table2 = "".join(lines)

    sent_col = "sentiment_disagreement_ref_vs_transcript"
    if sent_col in df.columns:
        dis = df.groupby("acoustic_condition")[sent_col].mean().reset_index()
        lines = [
            "\\begin{table}[t]\n\\centering\n"
            "\\caption{Sentiment disagreement (reference vs transcript).}\n\\label{tab:disagreement}\n"
            "\\begin{tabular}{lr}\n\\toprule\nCondition & Rate \\\\\n\\midrule\n"
        ]
        for _, row in dis.iterrows():
            lines.append(f"{row['acoustic_condition']} & {row[sent_col]:.3f} \\\\\n")
        lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        table3 = "".join(lines)
    else:
        table3 = _stub_table("table3_disagreement", "sentiment column missing")

    (cfg.tables_dir / "table1_dataset.tex").write_text(table1, encoding="utf-8")
    (cfg.tables_dir / "table2_wer.tex").write_text(table2, encoding="utf-8")
    (cfg.tables_dir / "table3_disagreement.tex").write_text(table3, encoding="utf-8")
    logger.info("Wrote tables to %s", cfg.tables_dir)


def write_figures(cfg: ExperimentConfig) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.results_dir / "metrics.csv"
    pipeline_fig = cfg.figures_dir / "fig1_pipeline.pdf"
    if not pipeline_fig.exists():
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Reference text → TTS → acoustic conditions → Listen/Read → metrics",
            ha="center",
            va="center",
        )
        fig.savefig(pipeline_fig, bbox_inches="tight")
        plt.close(fig)

    if not metrics_path.is_file():
        logger.warning("Skipping empirical figures: no metrics.csv")
        return

    df = pd.read_csv(metrics_path)
    fig, ax = plt.subplots()
    df.boxplot(column="wer", by="acoustic_condition", ax=ax)
    ax.set_title("WER by acoustic condition")
    ax.set_xlabel("Condition")
    fig.savefig(cfg.figures_dir / "fig2_wer_by_condition.pdf", bbox_inches="tight")
    plt.close(fig)

    sent_col = "sentiment_disagreement_ref_vs_transcript"
    if sent_col in df.columns:
        fig, ax = plt.subplots()
        ax.scatter(df["wer"], df[sent_col].astype(int), alpha=0.6)
        ax.set_xlabel("WER")
        ax.set_ylabel("Sentiment disagreement (0/1)")
        ax.set_title("WER vs downstream disagreement")
        fig.savefig(cfg.figures_dir / "fig3_wer_vs_disagreement.pdf", bbox_inches="tight")
        plt.close(fig)


def run_paper_assets(cfg: ExperimentConfig) -> None:
    write_tables(cfg)
    write_figures(cfg)
